#!/usr/bin/env python3
"""Evaluate Qwen2.5-Omni (base, or a merged GRPO / SFT checkpoint) as a MOS predictor.

Same sets, metrics and output files as ../baselines/eval_qwen3omni_zeroshot.py, but with
the pilot's prompts (prompts.py) and the <answer> parser used by the rewards, so the
zero-shot row and the trained rows are measured identically.

Usage:
    # zero-shot starting point (the "before" row)
    python eval_omni.py --model $B/models/Qwen2.5-Omni-7B --prompt grpo \
        --nisqa-root $B/dataset/NISQA/NISQA_Corpus --out-dir results/q25o_zeroshot

    # after training: merge the LoRA first (see README), then
    python eval_omni.py --model output/grpo/vX/checkpoint-N-merged --prompt grpo \
        --nisqa-root ... --out-dir results/q25o_grpo
    python eval_omni.py --model output/sft/vX/checkpoint-N-merged --prompt sft ...

    # out-of-domain sets: add --csv $B/csv/{ttsds2,hablamos,tcd_voip,svcc23}_test.csv ...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines"))
from eval_qwen3omni_zeroshot import SCORE_RE, load_sets, metrics  # noqa: E402

from aqa_rewards import parse_answer  # noqa: E402
from prompts import GRPO_PROMPT, SFT_PROMPT, SYSTEM  # noqa: E402

PROMPTS = {"grpo": GRPO_PROMPT, "sft": SFT_PROMPT}


def parse(text):
    """<answer>x</answer> first; a base model that ignores the tags may still write 'Score: x'.
    Returns (score or nan, whether the strict <answer> format was used)."""
    s, _ = parse_answer(text)
    if s is not None:
        return s, True
    hits = SCORE_RE.findall(text.split("</think>")[-1])
    if hits and 1.0 <= float(hits[-1]) <= 5.0:
        return float(hits[-1]), False
    return np.nan, False


class OmniJudge:
    def __init__(self, model, tp, max_tokens, max_model_len):
        from transformers import Qwen2_5OmniProcessor
        from vllm import LLM, SamplingParams

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model)
        self.llm = LLM(
            model=model, trust_remote_code=True, gpu_memory_utilization=0.9,
            tensor_parallel_size=tp, max_model_len=max_model_len, max_num_seqs=64,
            limit_mm_per_prompt={"audio": 1, "image": 1, "video": 0}, seed=1234,
        )
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def __call__(self, wav_paths, prompt):
        from qwen_omni_utils import process_mm_info

        inputs = []
        for p in wav_paths:
            # same layout as training: system text, then audio, then the instruction
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "audio", "audio": p},
                                             {"type": "text", "text": prompt}]},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
            inputs.append({"prompt": text, "multi_modal_data": {"audio": audios}})
        outs = self.llm.generate(inputs, sampling_params=self.params)
        return [o.outputs[0].text for o in outs]


def evaluate(judge, sets, prompt, out_dir, batch):
    rows = []
    for name, df in sets:
        texts = []
        for i in range(0, len(df), batch):
            texts += judge(list(df["wav_path"][i:i + batch]), prompt)
            print(f"{name}: {len(texts)}/{len(df)}", flush=True)
        parsed = [parse(t) for t in texts]
        df = df.assign(response=texts, pred=[p[0] for p in parsed], strict=[p[1] for p in parsed])
        df.to_csv(os.path.join(out_dir, f"pred_{name}.csv"), index=False)

        ok = df.dropna(subset=["pred"])
        row = {"set": name, "n_files": len(df), "n_unparsed": len(df) - len(ok),
               "n_strict_format": int(df["strict"].sum()),
               "n_distinct_scores": ok["pred"].nunique()}
        if len(ok) > 2:
            row.update({f"utt_{k}": v for k, v in metrics(ok["pred"], ok["mos"]).items()})
        if ok["system"].notna().all() and ok["system"].nunique() > 2:
            sys_df = ok.groupby("system")[["pred", "mos"]].mean()
            row["n_systems"] = len(sys_df)
            row.update({f"sys_{k}": v for k, v in metrics(sys_df["pred"], sys_df["mos"]).items()})
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + table.round(3).to_string(index=False))
    print(f"\nSaved responses, predictions and summary.csv to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Qwen2.5-Omni folder, base or merged checkpoint")
    ap.add_argument("--nisqa-root")
    ap.add_argument("--csv", nargs="*", help="CSV(s) with wav_path, mos[, system]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", choices=PROMPTS, default="grpo")
    ap.add_argument("--tp", type=int, default=1,
                    help="a 7B model fits on one GPU; use CUDA_VISIBLE_DEVICES to pick it")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, help="first N clips per set (smoke test)")
    args = ap.parse_args()
    if not args.nisqa_root and not args.csv:
        ap.error("give --nisqa-root and/or --csv")
    os.makedirs(args.out_dir, exist_ok=True)

    sets = load_sets(args)
    judge = OmniJudge(args.model, args.tp, args.max_tokens, args.max_model_len)
    evaluate(judge, sets, PROMPTS[args.prompt], args.out_dir, args.batch)
    # vLLM workers can hang on shutdown, and a plain os._exit leaves the engine process
    # alive and holding the GPU. Everything is saved, so stop the children, then exit hard.
    import multiprocessing
    for child in multiprocessing.active_children():
        child.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
