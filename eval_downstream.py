"""
Evaluate models on downstream benchmarks via lm-evaluation-harness.

Supports both baseline Qwen3 and AttnRes models. Uses the standard
lm-eval framework for reproducible results.

Usage:
    # Evaluate pretrained Qwen3-0.6B (baseline reference)
    python eval_downstream.py --model_path Qwen/Qwen3-0.6B --mode baseline

    # Evaluate fine-tuned Delta AttnRes
    python eval_downstream.py --model_path output/ft-Qwen3-0.6B-delta_block-10k/final --mode delta_block

    # Evaluate from-scratch model
    python eval_downstream.py --model_path output/matrix-delta_block-d1280-L36-10k/final --mode delta_block

    # Quick test (fewer tasks)
    python eval_downstream.py --model_path Qwen/Qwen3-0.6B --mode baseline --tasks hellaswag,arc_easy

    # All benchmarks for paper table
    python eval_downstream.py --model_path Qwen/Qwen3-0.6B --mode baseline --tasks paper
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Attention-Residuals"))


PAPER_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "boolq",
    "mmlu",
    "lambada_openai",
]

# Concise task set for quick sanity checks
QUICK_TASKS = ["hellaswag", "arc_easy", "piqa", "lambada_openai"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Model name or path")
    p.add_argument("--mode", required=True,
                   choices=["baseline", "block", "block_v", "full", "full_v",
                            "delta", "delta_block", "delta_block_v", "delta_v",
                            "first_layer", "pre_gated"])
    p.add_argument("--tasks", default="paper",
                   help="Comma-separated task list, or 'paper' / 'quick'")
    p.add_argument("--num_fewshot", type=int, default=0,
                   help="Number of few-shot examples (0 for zero-shot)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default=None,
                   help="Directory to save results JSON")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of examples per task (for debugging)")
    return p.parse_args()


def load_model_and_tokenizer(model_path, mode, device):
    """Load model based on mode."""
    from transformers import AutoTokenizer

    if mode == "baseline":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device})
    else:
        from modeling_qwen3_attnres import Qwen3AttnResForCausalLM
        model = Qwen3AttnResForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": device})

    model.eval()

    # Try loading tokenizer from model path, fall back to Qwen3-0.6B
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    return model, tokenizer


def eval_with_lm_eval(model, tokenizer, tasks, num_fewshot, batch_size, device, limit=None):
    """Run lm-evaluation-harness on loaded model."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )

    return results


def eval_manual(model, tokenizer, tasks, device, limit=None):
    """Fallback: manual evaluation without lm-eval-harness."""
    from datasets import load_dataset
    import math

    results = {}

    if "hellaswag" in tasks:
        ds = load_dataset("Rowan/hellaswag", split="validation")
        n = limit or len(ds)
        correct = 0
        for sample in ds.select(range(min(n, len(ds)))):
            ctx = sample["ctx"]
            endings = sample["endings"]
            label = int(sample["label"])
            scores = []
            for ending in endings:
                text = ctx + " " + ending
                input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=2048)["input_ids"].to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids).logits
                ctx_ids = tokenizer(ctx, return_tensors="pt")["input_ids"]
                ctx_len = ctx_ids.size(1)
                shift_logits = logits[:, ctx_len-1:-1, :]
                shift_labels = input_ids[:, ctx_len:]
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                scores.append(token_log_probs.mean().item())
            if scores.index(max(scores)) == label:
                correct += 1
        results["hellaswag"] = {"acc": correct / min(n, len(ds))}
        print(f"  HellaSwag: {results['hellaswag']['acc']:.4f}")

    if "arc_easy" in tasks:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        n = limit or len(ds)
        correct = 0
        for sample in ds.select(range(min(n, len(ds)))):
            question = sample["question"]
            choices = sample["choices"]["text"]
            labels = sample["choices"]["label"]
            answer = sample["answerKey"]
            scores = []
            for choice in choices:
                text = f"Question: {question}\nAnswer: {choice}"
                input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=512)["input_ids"].to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids).logits
                q_ids = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt")["input_ids"]
                q_len = q_ids.size(1)
                shift_logits = logits[:, q_len-1:-1, :]
                shift_labels = input_ids[:, q_len:]
                if shift_labels.numel() == 0:
                    scores.append(float("-inf"))
                    continue
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                scores.append(token_log_probs.mean().item())
            pred_idx = scores.index(max(scores))
            if labels[pred_idx] == answer:
                correct += 1
        results["arc_easy"] = {"acc": correct / min(n, len(ds))}
        print(f"  ARC-Easy: {results['arc_easy']['acc']:.4f}")

    if "arc_challenge" in tasks:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        n = limit or len(ds)
        correct = 0
        for sample in ds.select(range(min(n, len(ds)))):
            question = sample["question"]
            choices = sample["choices"]["text"]
            labels = sample["choices"]["label"]
            answer = sample["answerKey"]
            scores = []
            for choice in choices:
                text = f"Question: {question}\nAnswer: {choice}"
                input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=512)["input_ids"].to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids).logits
                q_ids = tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt")["input_ids"]
                q_len = q_ids.size(1)
                shift_logits = logits[:, q_len-1:-1, :]
                shift_labels = input_ids[:, q_len:]
                if shift_labels.numel() == 0:
                    scores.append(float("-inf"))
                    continue
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                scores.append(token_log_probs.mean().item())
            pred_idx = scores.index(max(scores))
            if labels[pred_idx] == answer:
                correct += 1
        results["arc_challenge"] = {"acc": correct / min(n, len(ds))}
        print(f"  ARC-Challenge: {results['arc_challenge']['acc']:.4f}")

    if "piqa" in tasks:
        ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
        n = limit or len(ds)
        correct = 0
        for sample in ds.select(range(min(n, len(ds)))):
            goal = sample["goal"]
            sols = [sample["sol1"], sample["sol2"]]
            label = sample["label"]
            scores = []
            for sol in sols:
                text = f"{goal} {sol}"
                input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=512)["input_ids"].to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids).logits
                g_ids = tokenizer(goal, return_tensors="pt")["input_ids"]
                g_len = g_ids.size(1)
                shift_logits = logits[:, g_len-1:-1, :]
                shift_labels = input_ids[:, g_len:]
                if shift_labels.numel() == 0:
                    scores.append(float("-inf"))
                    continue
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                scores.append(token_log_probs.mean().item())
            if scores.index(max(scores)) == label:
                correct += 1
        results["piqa"] = {"acc": correct / min(n, len(ds))}
        print(f"  PIQA: {results['piqa']['acc']:.4f}")

    if "winogrande" in tasks:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation",
                          trust_remote_code=True)
        n = limit or len(ds)
        correct = 0
        for sample in ds.select(range(min(n, len(ds)))):
            sentence = sample["sentence"]
            opt1 = sample["option1"]
            opt2 = sample["option2"]
            label = int(sample["answer"]) - 1  # 1-indexed → 0-indexed
            scores = []
            for opt in [opt1, opt2]:
                text = sentence.replace("_", opt)
                input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=512)["input_ids"].to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids).logits
                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                scores.append(token_log_probs.mean().item())
            if scores.index(max(scores)) == label:
                correct += 1
        results["winogrande"] = {"acc": correct / min(n, len(ds))}
        print(f"  WinoGrande: {results['winogrande']['acc']:.4f}")

    if "boolq" in tasks:
        ds = load_dataset("google/boolq", split="validation")
        n = limit or len(ds)
        correct = 0
        for sample in ds.select(range(min(n, len(ds)))):
            passage = sample["passage"]
            question = sample["question"]
            label = sample["answer"]  # True/False
            scores = []
            for ans in ["Yes", "No"]:
                text = f"{passage}\nQuestion: {question}?\nAnswer: {ans}"
                input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                      max_length=2048)["input_ids"].to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_ids).logits
                # Score the answer token
                scores.append(logits[0, -1].detach())
            # Compare log-prob of "Yes" vs "No" answer tokens
            yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = tokenizer.encode("No", add_special_tokens=False)[0]
            prompt = f"{passage}\nQuestion: {question}?\nAnswer:"
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=2048)["input_ids"].to(device)
            with torch.no_grad():
                logits = model(input_ids=prompt_ids).logits[0, -1]
            pred = logits[yes_id] > logits[no_id]
            if pred.item() == label:
                correct += 1
        results["boolq"] = {"acc": correct / min(n, len(ds))}
        print(f"  BoolQ: {results['boolq']['acc']:.4f}")

    if "lambada_openai" in tasks or "lambada" in tasks:
        ds = load_dataset("lambada", split="test")
        n = limit or len(ds)
        correct = 0
        for sample in ds.select(range(min(n, len(ds)))):
            text = sample["text"]
            words = text.strip().split()
            if len(words) < 2:
                continue
            last_word = words[-1]
            context = " ".join(words[:-1])
            input_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)
            target_ids = tokenizer(" " + last_word, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0, -1]
                pred_id = logits.argmax().item()
            if len(target_ids) > 0 and pred_id == target_ids[0]:
                correct += 1
        results["lambada_openai"] = {"acc": correct / min(n, len(ds))}
        print(f"  LAMBADA: {results['lambada_openai']['acc']:.4f}")

    return results


def main():
    args = parse_args()

    # Resolve task list
    if args.tasks == "paper":
        tasks = PAPER_TASKS
    elif args.tasks == "quick":
        tasks = QUICK_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    print(f"Loading {args.mode} model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.mode, args.device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params | mode={args.mode}")
    print(f"Tasks: {', '.join(tasks)}")
    print()

    # Try lm-eval-harness first, fall back to manual
    try:
        import lm_eval
        print("Using lm-evaluation-harness...")
        results = eval_with_lm_eval(
            model, tokenizer, tasks, args.num_fewshot,
            args.batch_size, args.device, args.limit)

        # Print summary
        print("\n" + "=" * 60)
        print(f"RESULTS ({args.mode}) — {args.model_path}")
        print("=" * 60)
        for task_name, task_results in results["results"].items():
            metrics = {k: v for k, v in task_results.items()
                       if not k.startswith("_") and isinstance(v, (int, float))}
            for metric, value in metrics.items():
                print(f"  {task_name}/{metric}: {value:.4f}")

    except ImportError:
        print("lm-eval not installed, using manual evaluation...")
        print("(Install with: pip install lm-eval>=0.4.0)")
        print()
        results = {"results": eval_manual(model, tokenizer, tasks, args.device, args.limit)}

    # Save results
    output_dir = args.output_dir or os.path.dirname(args.model_path)
    if output_dir and output_dir != args.model_path:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"eval_downstream_{args.mode}.json")
        with open(out_file, "w") as f:
            json.dump({
                "model_path": args.model_path,
                "mode": args.mode,
                "tasks": tasks,
                "num_fewshot": args.num_fewshot,
                "results": results.get("results", results),
            }, f, indent=2, default=str)
        print(f"\nResults saved → {out_file}")


if __name__ == "__main__":
    main()
