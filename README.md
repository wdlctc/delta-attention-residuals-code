# Delta Attention Residuals

Official code for **"Delta Attention Residuals: Per-Sublayer Sources for Cross-Layer Information Flow"**.

Cheng Luo, Zefan Cai, Junjie Hu

[[Paper]](https://github.com/wdlctc/delta-attention-residuals-arxiv)

## Overview

Delta Attention Residuals replace cumulative hidden states with per-sublayer deltas as routing sources for cross-layer connectivity. The key insight: routing over *what changed* rather than *what accumulated* yields 3x sharper routing and consistently better perplexity across all tested scales (220M--8B).

Two variants:
- **Delta AttnRes**: per-sublayer deltas (2L sources), best quality
- **Delta Block**: block-level deltas (~L/B sources), practical default with minimal overhead

## Repository Structure

```
Attention-Residuals/
  modeling_qwen3_attnres.py   # Core model: Qwen3 + Delta Attention Residuals
train_scratch.py              # From-scratch training (DDP, up to ~1B)
train_scratch_fsdp.py         # From-scratch training (FSDP, 7B+)
train_finetune.py             # Fine-tuning pretrained models
eval_downstream.py            # Downstream evaluation (lm-eval-harness)
run_8b_delta_block.sh         # Launch script for 8B training
```

## Quick Start

### Requirements

```bash
pip install torch transformers datasets wandb
```

### Training from scratch (220M--1B, DDP)

```bash
# Baseline
torchrun --standalone --nproc_per_node=8 train_scratch.py --mode baseline

# Delta Block (recommended)
torchrun --standalone --nproc_per_node=8 train_scratch.py --mode delta_block --compile_model

# Delta AttnRes (per-sublayer)
torchrun --standalone --nproc_per_node=8 train_scratch.py --mode delta --compile_model
```

### Training from scratch (7B+, FSDP)

```bash
torchrun --standalone --nproc_per_node=8 train_scratch_fsdp.py \
    --mode delta_block \
    --hidden_size 4096 --num_layers 36 --num_heads 32 --num_kv_heads 8 \
    --intermediate_size 12288 \
    --batch_size 4 --grad_accum 2 \
    --compile_model --shard_grad_op \
    --steps 10000
```

### Fine-tuning pretrained models

```bash
torchrun --standalone --nproc_per_node=4 train_finetune.py \
    --base_model Qwen/Qwen3-0.6B \
    --mode delta_block \
    --lr 5e-5 --lr_attnres 5e-3 \
    --steps 20000
```

## Results & W&B Runs

The exact W&B run for every paper experiment is listed in [`WANDB_RUNS.md`](./WANDB_RUNS.md) (training/validation curves, configs, and system metrics). Project: <https://wandb.ai/wdlctc_abr/attention-residual-h100>.

### From-Scratch Training (10K steps, FineWeb-Edu)

| Scale | Baseline | AttnRes | Delta Block | Delta AttnRes |
|-------|----------|---------|-------------|---------------|
| 220M  | 38.71    | 37.39   | **37.08**   | **36.83**     |
| 533M  | 32.00    | 31.75   | **31.16**   | **31.05**     |
| 1044M | 29.70    | 31.76   | **29.19**   | **29.13**     |

### Fine-tuning Qwen3-0.6B (downstream avg accuracy)

| Baseline FT | AttnRes | Delta Block |
|-------------|---------|-------------|
| 55.0%       | 54.1%   | **55.6%**   |

## Citation

```bibtex
@article{luo2026delta,
  title={Delta Attention Residuals},
  author={Luo, Cheng and Cai, Zefan and Hu, Junjie},
  year={2026}
}
```

## License

MIT
