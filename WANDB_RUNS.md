# Weights & Biases Runs

This file maps each experiment in the [paper](https://github.com/wdlctc/delta-attention-residuals-arxiv) to the exact W&B runs used to produce its numbers. All runs are under the `wdlctc_abr` entity; each link opens the run page with full training/validation curves, configs, and system metrics.

W&B project landing page: <https://wandb.ai/wdlctc_abr/attention-residual-h100>

> **Note on naming.** Internal mode names in the code map to paper names as follows: `baseline` = Baseline, `block` = AttnRes, `full` = Full AttnRes, `delta` = Delta AttnRes, `delta_block` = Delta Block.

---

## Table 1 — Main results (10K steps, FineWeb-Edu)

### 220M ($d{=}768$, $L{=}12$)

| Method | Val PPL | W&B Run |
|---|---|---|
| Baseline       | 38.71 | [`irh3j7wf`](https://wandb.ai/wdlctc_abr/residual/runs/irh3j7wf)  (`baseline-bs2ga2-d768-L12-10k`) |
| AttnRes        | 37.39 | [`woal1e00`](https://wandb.ai/wdlctc_abr/residual/runs/woal1e00)  (`block-B4-bs2ga2-d768-L12-10k`) |
| Full AttnRes   | 37.30 | [`oiv6w41g`](https://wandb.ai/wdlctc_abr/residual/runs/oiv6w41g)  (`full-N12-bs2ga2-d768-L12-10k`) |
| Delta Block    | 37.08 | [`7xdjieki`](https://wandb.ai/wdlctc_abr/residual/runs/7xdjieki)  (`delta-block-d768-L12-10k-fixed`) |
| Delta AttnRes  | 36.83 | [`mxcrhyuv`](https://wandb.ai/wdlctc_abr/residual/runs/mxcrhyuv)  (`delta-d768-L12-10k-final`) |

### 533M ($d{=}1024$, $L{=}24$)

| Method | Val PPL | W&B Run |
|---|---|---|
| Baseline       | 32.00 | [`cph11fzm`](https://wandb.ai/wdlctc_abr/residual/runs/cph11fzm)  (`baseline-d1024-L24-10k-lr6e4-v3`) |
| AttnRes        | 31.75 | [`2dbtu34n`](https://wandb.ai/wdlctc_abr/residual/runs/2dbtu34n)  (`block-d1024-L24-10k-lr6e4-v3`) |
| Full AttnRes   | 31.68 | [`4rkk7jqj`](https://wandb.ai/wdlctc_abr/residual/runs/4rkk7jqj)  (`full-N24-d1024-L24-10k-lr6e4-v3`) |
| Delta Block    | 31.16 | [`f9cjqypq`](https://wandb.ai/wdlctc_abr/residual/runs/f9cjqypq)  (`delta_block-d1024-L24-10k-lr6e4-v3`) |
| Delta AttnRes  | 31.05 | [`fsdf0y3h`](https://wandb.ai/wdlctc_abr/residual/runs/fsdf0y3h)  (`delta-d1024-L24-10k-lr6e4-v3`) |

### 1044M ($d{=}1280$, $L{=}36$)

| Method | Val PPL | W&B Run |
|---|---|---|
| Baseline       | 29.70 | [`nlzsi4dy`](https://wandb.ai/wdlctc_abr/residual/runs/nlzsi4dy)  (`baseline-d1280-L36-10k-lr6e4`) |
| AttnRes        | 31.76 | [`czu4e462`](https://wandb.ai/wdlctc_abr/residual/runs/czu4e462)  (`block-d1280-L36-10k-lr6e4`) |
| Full AttnRes   | 33.36 | [`b2jjvzri`](https://wandb.ai/wdlctc_abr/residual/runs/b2jjvzri)  (`full-N36-d1280-L36-10k-lr6e4`) |
| Delta Block    | 29.19 | [`zh6w1o9d`](https://wandb.ai/wdlctc_abr/residual/runs/zh6w1o9d)  (`delta_block-d1280-L36-10k-lr6e4`) |
| Delta AttnRes  | 29.13 | [`16gcajbs`](https://wandb.ai/wdlctc_abr/residual/runs/16gcajbs)  (`delta-d1280-L36-10k-lr6e4`) |

---

## Table 2 — From-scratch Qwen3-0.6B ($d{=}1024$, $L{=}28$, $N{=}28$, 10K steps, lr $6{\times}10^{-4}$)

| Method | Val PPL | W&B Run |
|---|---|---|
| Baseline    | 32.22 | [`2pe87xex`](https://wandb.ai/wdlctc_abr/residual/runs/2pe87xex)  (`baseline-qwen06b-10k`) |
| AttnRes     | 32.38 | [`d8bbxko5`](https://wandb.ai/wdlctc_abr/residual/runs/d8bbxko5)  (`block-N28-qwen06b-10k`) |
| Delta Block | 31.45 | [`vnvlxy3w`](https://wandb.ai/wdlctc_abr/residual/runs/vnvlxy3w)  (`dblock-N28-qwen06b-10k`) |

---

## Table 3 — Scaling to 8B ($d{=}4096$, $L{=}36$, 10K steps, FSDP on 8×H100)

| Method | Val PPL | W&B Run |
|---|---|---|
| Baseline    | 17.43 | [`6l151369`](https://wandb.ai/wdlctc_abr/residual/runs/6l151369)  (`scratch-baseline-8B-10k`) |
| AttnRes     | 18.58 | [`at7xzn5l`](https://wandb.ai/wdlctc_abr/residual/runs/at7xzn5l)  (`scratch-block-8B-10k`) |
| Delta Block | 16.00 | [`23hux20r`](https://wandb.ai/wdlctc_abr/residual/runs/23hux20r)  (`scratch-delta_block-8B-10k`) |

Tok/s and memory in the paper are reported from dedicated benchmark runs (`bench-baseline-8B`, `bench-block-8B`, `bench-delta_block-8B`) in the same project.

---

## Table 4 — Delta Block block-size ablation (10K steps)

### 220M ($L{=}12$)

| Config | Val PPL | W&B Run |
|---|---|---|
| Baseline           | 38.71 | [`irh3j7wf`](https://wandb.ai/wdlctc_abr/residual/runs/irh3j7wf) |
| Delta Block $B{=}1$  | 37.44 | [`i1sx3124`](https://wandb.ai/wdlctc_abr/residual/runs/i1sx3124)  (`ablation-dblock-B1-d768-L12-10k-lr6e4`) |
| Delta Block $B{=}2$  | 37.08 | [`fmuj604s`](https://wandb.ai/wdlctc_abr/residual/runs/fmuj604s)  (`ablation-dblock-B2-d768-L12-10k-lr6e4`) |
| Delta Block $B{=}4$  | 36.98 | [`wmgg7byx`](https://wandb.ai/wdlctc_abr/residual/runs/wmgg7byx)  (`ablation-dblock-B4-d768-L12-10k-lr6e4`) |
| Delta Block $B{=}6$  | 36.92 | [`pl1hfocm`](https://wandb.ai/wdlctc_abr/residual/runs/pl1hfocm)  (`ablation-dblock-B6-d768-L12-10k-lr6e4`) |
| Delta Block $B{=}12$ | 37.34 | [`bp6qpukt`](https://wandb.ai/wdlctc_abr/residual/runs/bp6qpukt)  (`ablation-dblock-B12-d768-L12-10k-lr6e4`) |
| Delta AttnRes      | 36.75 | [`yibyyl95`](https://wandb.ai/wdlctc_abr/residual/runs/yibyyl95)  (`ablation-delta-full-d768-L12-10k-lr6e4`) |

### 533M ($L{=}24$)

| Config | Val PPL | W&B Run |
|---|---|---|
| Baseline           | 32.00 | [`cph11fzm`](https://wandb.ai/wdlctc_abr/residual/runs/cph11fzm) |
| Delta Block $B{=}2$  | 31.23 | [`xuznr576`](https://wandb.ai/wdlctc_abr/residual/runs/xuznr576)  (`ablation-dblock-B2-d1024-L24-10k-v3`) |
| Delta Block $B{=}6$  | 31.19 | [`kqqn9ia7`](https://wandb.ai/wdlctc_abr/residual/runs/kqqn9ia7)  (`ablation-dblock-B6-d1024-L24-10k-v5`) |
| Delta Block $B{=}12$ | 31.22 | [`8ahsydnu`](https://wandb.ai/wdlctc_abr/residual/runs/8ahsydnu)  (`ablation-dblock-B12-d1024-L24-10k-v2`) |
| Delta Block $B{=}24$ | 31.18 | [`yozwx3fm`](https://wandb.ai/wdlctc_abr/residual/runs/yozwx3fm)  (`ablation-dblock-B24-d1024-L24-10k-v2`) |
| Delta AttnRes      | 31.05 | [`fsdf0y3h`](https://wandb.ai/wdlctc_abr/residual/runs/fsdf0y3h) |

---

## Table 5 — Fine-tuning Qwen3-0.6B on FineWeb-Edu

| Method | W&B Run |
|---|---|
| Baseline    | [`wgm9tkku`](https://wandb.ai/wdlctc_abr/residual/runs/wgm9tkku)  (`ft-baseline-0.6B-20k`) |
| AttnRes     | [`kiagouy8`](https://wandb.ai/wdlctc_abr/residual/runs/kiagouy8)  (`ft-block-0.6B-20k`) |
| Delta Block | [`te3jah3u`](https://wandb.ai/wdlctc_abr/residual/runs/te3jah3u)  (`ft-delta-block-dual-0.6B-20k`) |

Downstream 0-shot accuracies (paper: Delta Block 55.6%, Baseline 55.0%, AttnRes 54.1%) are evaluated with `eval_downstream.py` on the final checkpoint of each run.

---

## Reproducing

Each row's training command is reflected in the run's `args` panel on W&B. The corresponding training scripts in this repo are:
- **From-scratch (Tables 1–2, 4):** `train_scratch.py` with `--mode {baseline,block,full,delta,delta_block}`
- **From-scratch 8B (Table 3):** `train_scratch_fsdp.py` (see `run_8b_delta_block.sh`)
- **Fine-tuning (Table 5):** `train_finetune.py`
- **Downstream eval (Table 5):** `eval_downstream.py`
