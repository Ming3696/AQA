#!/usr/bin/env bash
# Rerun the GRPO pilot with the decimal-penalty fix (aqa_rewards.py, 2026-10-27), then evaluate.
#   Run A: same settings as v4 (lr 1e-5)  -> isolates the bug fix
#   Run B: lr 5e-5                       -> does a stronger update help?
# Each run ~2.7 h on 8 GPUs, eval ~10 min. Total ~6 h.  Usage (inside tmux):  bash rerun_fixed.sh
set -uo pipefail
B=${B:-/share/panjiayi/chengqiming/AQA}
cd "$(dirname "$0")"
grep -q 'rstrip("0")' aqa_rewards.py || { echo "aqa_rewards.py is the old version: upload the fixed one first"; exit 1; }
N="--nisqa-root $B/dataset/NISQA/NISQA_Corpus"

for cfg in "1e-5 grpo_fix" "5e-5 grpo_fix_lr5e-5"; do
  set -- $cfg; LR=$1; NAME=$2; OUT=$B/pilot/output/$NAME
  echo "=== $NAME (lr $LR) $(date)"
  LR=$LR OUT=$OUT bash run_grpo.sh > $NAME.log 2>&1 || { echo "training failed, see $NAME.log"; continue; }
  CKPT=$(ls -td $OUT/v*/checkpoint-* | grep -v merged | head -1)
  ENABLE_AUDIO_OUTPUT=1 swift export --adapters $CKPT --merge_lora true >> $NAME.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python eval_omni.py --model ${CKPT}-merged --prompt grpo $N \
    --out-dir results/q25o_$NAME >> $NAME.log 2>&1
  cat results/q25o_$NAME/summary.csv
done
echo "=== all done $(date)"
