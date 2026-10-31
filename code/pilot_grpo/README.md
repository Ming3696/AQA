# GRPO pilot: Qwen2.5-Omni-7B on NISQA

Goal: check that VQAThinker-style rewards teach an audio LLM to predict MOS, before we pool
datasets and move to Qwen3-Omni. One question decides the next step:
**does GRPO beat both the zero-shot model and plain SFT on the NISQA test sets?**

| File | What it does |
|---|---|
| `prompts.py` | System text and the two prompts (GRPO with `<think>`, SFT score-only). Shared by everything below |
| `make_nisqa_jsonl.py` | NISQA train / val → ms-swift jsonl (GRPO and SFT versions) |
| `aqa_rewards.py` | ms-swift plugin: `aqa_format`, `aqa_reg` (Gaussian, σ = 0.5), `aqa_rank` (pairwise fidelity, same-subset pairs only) |
| `run_grpo.sh` | GRPO, LoRA r64, 8 GPUs, vLLM colocated, 4 samples per clip, 64 clips per step, 2 epochs (~344 steps) |
| `run_sft.sh` | SFT baseline, same model / LoRA / data / epochs |
| `eval_omni.py` | vLLM evaluation on NISQA tests and any CSV set; same metrics as the baselines |

Reward details and the reasons for each choice are in the docstring of `aqa_rewards.py` and
in `knowledge/literature/VQAThinker_paper_and_code_explained.md` §4.3 and §7.

Tested here on mock data only: the rewards (format, regression, ranking incl. the
same-subset rule and the clip grouping), the data prep and the answer parser. Nothing has
run on a GPU yet, so do the smoke tests in step 4 before the full runs.

## 1. Environment (once)

Same vLLM build as the Qwen3-Omni env (driver 535 needs the CUDA 12.9 wheels), plus ms-swift.
A separate env, so the working `q3o` env stays untouched.

```bash
export B=/share/panjiayi/chengqiming/AQA
conda create -p $B/envs/swift python=3.11 -y
conda activate $B/envs/swift
pip install uv
uv pip install vllm==0.25.1 "torchcodec==0.16.0+cu129" \
  --extra-index-url https://wheels.vllm.ai/0.25.1/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
uv pip install "ms-swift==4.6.0" "trl==0.29.1" qwen-omni-utils==0.0.9 audioread librosa soundfile tensorboard

# check: expect 4.6.0, 0.25.1+cu129, a transformers version below 5.20, True
python -c "import swift, vllm, transformers, torch; print(swift.__version__, vllm.__version__, transformers.__version__, torch.cuda.is_available())"
```

Copy this folder to `$B/code/pilot_grpo` (next to `$B/code/baselines`, which `eval_omni.py` imports from).

## 2. Model and data

```bash
cd $B/code/pilot_grpo
modelscope download --model Qwen/Qwen2.5-Omni-7B --local_dir $B/models/Qwen2.5-Omni-7B
python make_nisqa_jsonl.py --nisqa-root $B/dataset/NISQA/NISQA_Corpus --out $B/pilot/data
```
Expect 10,000 + 1,020 training clips, 0 missing.

## 3. Zero-shot row (the "before" number, and a check that vLLM runs Qwen2.5-Omni)

```bash
CUDA_VISIBLE_DEVICES=0 python eval_omni.py --model $B/models/Qwen2.5-Omni-7B --prompt grpo \
  --nisqa-root $B/dataset/NISQA/NISQA_Corpus --out-dir results/q25o_zeroshot --limit 20
# if that works, rerun without --limit
```
`summary.csv` also reports `n_strict_format` (answers that used `<answer>` tags) and
`n_distinct_scores` (the score-collapse problem we saw with Qwen3-Omni).

## 4. Smoke tests (5 steps each)

```bash
MAX_STEPS=5 bash run_grpo.sh 2>&1 | tee smoke_grpo.log
MAX_STEPS=5 bash run_sft.sh  2>&1 | tee smoke_sft.log
```
In the GRPO log, check that each step prints `rewards/aqa_format/mean`, `rewards/aqa_reg/mean`
and `rewards/aqa_rank/mean`, and that `completions.jsonl` under `$B/pilot/output/grpo/v*/`
shows real reasoning about the audio. If it runs out of memory, lower
`--vllm_gpu_memory_utilization` to 0.3 or `--per_device_train_batch_size` to 4 (and raise
`--gradient_accumulation_steps` to 8 so a step still covers 64 clips).

## 5. Full runs

```bash
nohup bash run_grpo.sh > grpo.log 2>&1 &
# afterwards (both use all 8 GPUs)
nohup bash run_sft.sh  > sft.log  2>&1 &
tensorboard --logdir $B/pilot/output --port 6006   # optional
```
What healthy GRPO training looks like:
- `aqa_format` near 1.0 within the first few dozen steps;
- `aqa_reg` and `aqa_rank` rising steadily;
- `frac_reward_zero_std` low. A high value means all 4 samples of a clip get the same reward,
  so there is nothing to learn from; raise `--temperature` or `--num_generations` if it stays high;
- completion length stable. A collapse to very short reasoning is a warning sign.

## 6. Evaluate a checkpoint

```bash
CKPT=$B/pilot/output/grpo/vX-XXXX/checkpoint-N
ENABLE_AUDIO_OUTPUT=1 swift export --adapters $CKPT --merge_lora true    # writes ${CKPT}-merged
CUDA_VISIBLE_DEVICES=0 python eval_omni.py --model ${CKPT}-merged --prompt grpo \
  --nisqa-root $B/dataset/NISQA/NISQA_Corpus --out-dir results/q25o_grpo
# SFT checkpoint: same, with --prompt sft --out-dir results/q25o_sft
```
Evaluate the final checkpoint and one from the middle of training. Each eval needs one GPU,
so several can run side by side.

## 7. Go / no-go

| Outcome on NISQA FOR / P501 / LIVETALK | Next step |
|---|---|
| GRPO ≥ ~0.85 SRCC and ≥ SFT | Scale up: pooled speech set, then OOD sets, then Qwen3-Omni |
| GRPO ≈ SFT, both well above zero-shot | Rewards work, but RL isn't adding much yet; check the ranking reward's batch size and the reasoning quality before scaling |
| GRPO well below SFT | Debug the rewards on the logged completions before adding data |

For reference: SHEET (trained on 8 datasets incl. NISQA) gets 0.91 / 0.94 / 0.89, and
Qwen3-Omni zero-shot gets 0.66 / 0.38 / 0.55.
