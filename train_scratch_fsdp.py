"""
Train Qwen3-8B / Qwen3-AttnRes-8B from scratch on FineWeb-Edu using FSDP.

Requires FSDP for 7B+ models that don't fit in single-GPU DDP.
Uses gradient checkpointing to reduce activation memory.

Usage:
    # Baseline
    torchrun --nproc_per_node=8 train_scratch_fsdp.py --mode baseline --steps 20000

    # Delta-block AttnRes
    torchrun --nproc_per_node=8 train_scratch_fsdp.py --mode delta_block --steps 20000
"""

import argparse
import functools
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Attention-Residuals"))

from modeling_qwen3_attnres import Qwen3AttnResConfig, Qwen3AttnResForCausalLM, Qwen3AttnResDecoderLayer
from transformers import AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="delta_block",
                   choices=["baseline", "block", "delta_block", "delta", "full"],
                   help="AttnRes mode")
    # Qwen3-8B defaults
    p.add_argument("--hidden_size", type=int, default=4096)
    p.add_argument("--num_layers", type=int, default=36)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--num_kv_heads", type=int, default=8)
    p.add_argument("--intermediate_size", type=int, default=12288)
    p.add_argument("--num_blocks", type=int, default=4,
                   help="Number of AttnRes blocks")
    p.add_argument("--gate_type", default="bias",
                   choices=["bias", "sigmoid_scalar", "sigmoid_vector", "learnable_alpha"])
    p.add_argument("--null_source", action="store_true")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_name", default="default")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=2, help="per-GPU micro batch")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--wandb_project", default="residual")
    p.add_argument("--wandb_entity", default="wdlctc_abr")
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                   help="Enable gradient checkpointing (default: on for 8B)")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--compile_model", action="store_true",
                   help="torch.compile the model for faster training")
    p.add_argument("--shard_grad_op", action="store_true",
                   help="Use SHARD_GRAD_OP (ZeRO-2) instead of FULL_SHARD (ZeRO-3) for faster training")
    return p.parse_args()


def cosine_with_warmup(step, warmup, total, lr_min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cos


def token_stream(dataset_name, config_name, tokenizer, seq_len, rank, world_size, seed):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, name=config_name, split="train",
                      streaming=True)
    ds = ds.shuffle(seed=seed + rank, buffer_size=10_000)
    ds = ds.skip(rank)
    buf = []
    for sample in ds:
        text = sample.get("text") or sample.get("content") or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        buf.extend(ids)
        while len(buf) >= seq_len + 1:
            chunk = buf[:seq_len + 1]
            buf = buf[world_size * seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)


def build_model(args, device):
    """Build model from scratch."""
    common = dict(
        vocab_size=151936,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_len * 2,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        head_dim=args.hidden_size // args.num_heads,
    )

    if args.mode == "baseline":
        config = Qwen3Config(**common)
        model = Qwen3ForCausalLM(config)
    else:
        config = Qwen3AttnResConfig(
            attnres_num_blocks=args.num_blocks,
            attnres_mode=args.mode,
            attnres_gate_type=args.gate_type,
            attnres_use_null_source=args.null_source,
            **common,
        )
        model = Qwen3AttnResForCausalLM(config)

    return model


def save_fsdp_checkpoint(model, tokenizer, save_dir, rank):
    """Save FSDP model checkpoint using full state dict."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        # Save as HF-compatible checkpoint
        unwrapped = model._fsdp_wrapped_module if hasattr(model, '_fsdp_wrapped_module') else model.module
        unwrapped.save_pretrained(save_dir, state_dict=state_dict)
        tokenizer.save_pretrained(save_dir)


def main():
    args = parse_args()
    if args.no_gradient_checkpointing:
        args.gradient_checkpointing = False

    if args.run_name is None:
        args.run_name = f"scratch-{args.mode}-d{args.hidden_size}-L{args.num_layers}-{args.steps//1000}k"
    if args.out_dir is None:
        args.out_dir = f"./output/scratch-{args.mode}-d{args.hidden_size}-L{args.num_layers}-{args.steps//1000}k"

    # ── distributed ──
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    # ── W&B ──
    use_wandb = False
    if is_main:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.run_name, config=vars(args))
            use_wandb = True
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")

    # ── model ──
    if is_main:
        print(f"Building {args.mode} model from scratch (FSDP)...")

    model = build_model(args, device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if is_main:
        print(f"Model: {n_params:.1f}M params | mode={args.mode} | d={args.hidden_size} L={args.num_layers}")
        if args.mode != "baseline":
            n_attnres = sum(p.numel() for n, p in model.named_parameters() if "res_" in n)
            print(f"AttnRes params: {n_attnres/1e3:.1f}K")

    # ── gradient checkpointing ──
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main:
            print("Gradient checkpointing enabled")

    # ── torch.compile ──
    if args.compile_model:
        model = torch.compile(model)
        if is_main:
            print("torch.compile enabled")

    # ── FSDP wrapping ──
    # Wrap at the decoder layer level
    layer_cls = Qwen3AttnResDecoderLayer if args.mode != "baseline" else Qwen3DecoderLayer
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP if args.shard_grad_op else ShardingStrategy.FULL_SHARD,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    if is_main:
        print(f"FSDP wrapped with {world_size} GPUs")

    # ── optimizer ──
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    lr_min_ratio = args.lr_min / args.lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, args.warmup, args.steps, lr_min_ratio),
    )

    # ── data ──
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    stream = token_stream(args.dataset, args.dataset_name, tokenizer,
                          args.seq_len, rank, world_size, args.seed)

    val_stream = None
    if args.eval_every > 0:
        val_stream = token_stream(args.dataset, args.dataset_name, tokenizer,
                                  args.seq_len, rank, world_size, args.seed + 9999)

    @torch.no_grad()
    def evaluate(val_iter, n_steps):
        model.eval()
        total_loss = 0.0
        count = 0
        for _ in range(n_steps):
            batch_chunks = []
            for _ in range(args.batch_size):
                try:
                    chunk = next(val_iter)
                    batch_chunks.append(chunk[:-1])
                except StopIteration:
                    break
            if not batch_chunks:
                break
            input_ids = torch.stack(batch_chunks).to(device)
            labels = input_ids
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            total_loss += out.loss.item()
            count += 1
        model.train()
        avg = torch.tensor(total_loss / max(count, 1), device=device)
        dist.all_reduce(avg, op=dist.ReduceOp.AVG)
        return avg.item()

    # ── training ──
    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()

    val_iter = iter(val_stream) if val_stream is not None else None

    global_step = 0
    accum_step = 0
    accum_loss = 0.0
    t0 = time.time()
    tokens_seen = 0

    # Global batch size = batch_size * grad_accum * world_size
    global_batch_tokens = args.batch_size * args.grad_accum * world_size * args.seq_len
    if is_main:
        print(f"Global batch: {args.batch_size}×{args.grad_accum}×{world_size} = "
              f"{args.batch_size * args.grad_accum * world_size} seqs = "
              f"{global_batch_tokens // 1024}K tokens")
        print(f"Training for {args.steps} steps = "
              f"{args.steps * global_batch_tokens / 1e9:.1f}B tokens")
        print("Starting training...")

    batch_buf = []
    for chunk in stream:
        if global_step >= args.steps:
            break

        batch_buf.append(chunk[:-1])
        if len(batch_buf) < args.batch_size:
            continue

        input_ids = torch.stack(batch_buf).to(device)
        labels = input_ids
        batch_buf = []

        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss / args.grad_accum
        loss.backward()

        accum_loss += loss.item()
        accum_step += 1
        tokens_seen += args.seq_len * args.batch_size

        if accum_step < args.grad_accum:
            continue

        model.clip_grad_norm_(args.max_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        accum_step = 0

        if global_step % args.log_every == 0:
            loss_t = torch.tensor(accum_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)

            if is_main:
                elapsed = time.time() - t0
                tok_sec = tokens_seen * world_size / elapsed
                avg_loss = loss_t.item()
                lr_now = scheduler.get_last_lr()[0]
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {global_step:6d} | loss {avg_loss:.4f} | "
                      f"lr {lr_now:.2e} | "
                      f"{tok_sec/1e3:.1f}k tok/s | {mem_gb:.1f}GB")

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "train/tok_per_s": tok_sec,
                        "train/mem_gb": mem_gb,
                    }, step=global_step)

                tokens_seen = 0
                t0 = time.time()
        accum_loss = 0.0

        if global_step % args.save_every == 0:
            ckpt_dir = os.path.join(args.out_dir, f"step-{global_step}")
            save_fsdp_checkpoint(model, tokenizer, ckpt_dir, rank)
            if is_main:
                print(f"Saved checkpoint → {ckpt_dir}")
            dist.barrier()

        if args.eval_every > 0 and global_step % args.eval_every == 0 and val_iter is not None:
            val_loss = evaluate(val_iter, args.eval_steps)
            if is_main:
                import math as _math
                val_ppl = _math.exp(val_loss) if val_loss < 20 else float('inf')
                print(f"step {global_step:6d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                if use_wandb:
                    import wandb
                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=global_step)

    # ── final ──
    if args.eval_every > 0 and val_iter is not None:
        val_loss = evaluate(val_iter, args.eval_steps)
        if is_main:
            import math as _math
            val_ppl = _math.exp(val_loss) if val_loss < 20 else float('inf')
            print(f"FINAL   | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
            if use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=global_step)

    final_dir = os.path.join(args.out_dir, "final")
    save_fsdp_checkpoint(model, tokenizer, final_dir, rank)
    if is_main:
        print(f"Training done. Final model → {final_dir}")
        if use_wandb:
            import wandb
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
