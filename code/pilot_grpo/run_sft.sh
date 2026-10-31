#!/usr/bin/env bash
# SFT baseline for the pilot: same model, LoRA and data as run_grpo.sh, but trained to
# output "<answer>x.xx</answer>" directly with cross-entropy (VQAThinker Table 7 comparison).
# Global batch 64 clips, 2 epochs, so it sees exactly the same clips as the GRPO run.
#
# Smoke test first:  MAX_STEPS=5 bash run_sft.sh
set -euo pipefail
B=${B:-/share/panjiayi/chengqiming/AQA}
P=$B/pilot
export MODELSCOPE_CACHE=$B/cache/modelscope HF_HOME=$B/cache/hf

EXTRA=()
[[ -n "${MAX_STEPS:-}" ]] && EXTRA+=(--max_steps "$MAX_STEPS" --eval_steps "$MAX_STEPS" --save_steps "$MAX_STEPS")

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
ENABLE_AUDIO_OUTPUT=0 \
swift sft \
    --model $B/models/Qwen2.5-Omni-7B \
    --dataset $P/data/sft_train.jsonl \
    --val_dataset $P/data/sft_val.jsonl \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --max_length 4096 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 8 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --seed 1337 \
    --report_to tensorboard \
    --output_dir $P/output/sft \
    "${EXTRA[@]}"
