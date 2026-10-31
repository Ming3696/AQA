#!/usr/bin/env bash
# GRPO pilot: Qwen2.5-Omni-7B + LoRA on NISQA train, VQAThinker-style rewards.
# 8 x A800. vLLM runs colocated (one engine per GPU) and sleeps during the training step.
#
# Batch layout per optimizer step (on-policy, one rollout per step):
#   per GPU: 8 completions x 4 accumulation = 32 completions = 8 clips x 4 samples
#   global : 64 clips -> 11,020 clips / 64 = ~172 steps per epoch, 2 epochs = ~344 steps
# The ranking reward compares each clip with the other 7 clips on the same GPU.
#
# Smoke test first:  MAX_STEPS=5 bash run_grpo.sh
set -euo pipefail
B=${B:-/share/panjiayi/chengqiming/AQA}
P=$B/pilot
HERE=$(cd "$(dirname "$0")" && pwd)
export MODELSCOPE_CACHE=$B/cache/modelscope HF_HOME=$B/cache/hf   # home disk is full

EXTRA=()
[[ -n "${MAX_STEPS:-}" ]] && EXTRA+=(--max_steps "$MAX_STEPS" --eval_steps "$MAX_STEPS" --save_steps "$MAX_STEPS")

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
ENABLE_AUDIO_OUTPUT=0 \
swift rlhf \
    --rlhf_type grpo \
    --model $B/models/Qwen2.5-Omni-7B \
    --external_plugins $HERE/aqa_rewards.py \
    --reward_funcs aqa_format aqa_reg aqa_rank \
    --reward_weights 1 1 1 \
    --dataset $P/data/grpo_train.jsonl \
    --val_dataset $P/data/grpo_val.jsonl \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --sleep_level 1 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 1.0 \
    --beta 0.04 \
    --max_completion_length 512 \
    --max_length 4096 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate ${LR:-1e-5} \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 8 \
    --logging_steps 1 \
    --log_completions true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --seed 1337 \
    --report_to tensorboard \
    --output_dir ${OUT:-$P/output/grpo} \
    "${EXTRA[@]}"
