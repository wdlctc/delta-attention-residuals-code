#!/bin/bash
# Pretrain Qwen3-8B with delta-block AttnRes on 8xH100
# FSDP SHARD_GRAD_OP (ZeRO-2) + gradient checkpointing + torch.compile + tf32
# Global batch = 4 * 2 * 8 = 64 seqs = 128K tokens/step
# 10K steps = ~1.3B tokens

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

torchrun --standalone --nproc_per_node=8 \
    train_scratch_fsdp.py \
    --mode delta_block \
    --hidden_size 4096 \
    --num_layers 36 \
    --num_heads 32 \
    --num_kv_heads 8 \
    --intermediate_size 12288 \
    --num_blocks 4 \
    --seq_len 2048 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 3e-4 \
    --lr_min 3e-5 \
    --warmup 500 \
    --steps 10000 \
    --save_every 2000 \
    --eval_every 500 \
    --log_every 10 \
    --compile_model \
    --shard_grad_op \
    --run_name "scratch-delta_block-8B-10k" \
    --out_dir "./output/scratch-delta_block-8B-10k" \
    2>&1 | tee run_8b_delta_block.log
