#!/usr/bin/env python3
"""Zero-shot evaluation of Qwen3-Omni as an audio quality judge.

The model listens to each clip, explains what it hears, and ends with a 1-5
score. No fine-tuning. Reports the same SRCC / PLCC / KTAU / MSE table as
eval_sheet_nisqa.py, and keeps every reasoning text in pred_<set>.csv.

Two ways to give it data:
    --nisqa-root  path to NISQA_Corpus (runs FOR / P501 / LIVETALK / NSC)
    --csv         any CSV with columns wav_path, mos  (optional: system)
                  one CSV = one test set; pass several to run several sets

Usage (setup in qwen3omni_baseline.md):
    python eval_qwen3omni_zeroshot.py \
        --model /share/.../models/Qwen3-Omni-30B-A3B-Instruct \
        --nisqa-root /share/panjiayi/chengqiming/AQA/dataset/NISQA/NISQA_Corpus \
        --out-dir results/q3o_instruct --limit 20      # smoke test first

    # the reasoning variant
    python eval_qwen3omni_zeroshot.py --model .../Qwen3-Omni-30B-A3B-Thinking \
        --nisqa-root ... --out-dir results/q3o_thinking --max-tokens 4096

Audio goes through the official qwen_omni_utils loader (librosa, mono,
16 kHz), so anything above 8 kHz is not heard by the model.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

NISQA_SETS = ["FOR", "P501", "LIVETALK", "NSC"]

PROMPTS = {
    # score + explanation, the setting we care about
    "reason": (
        "You are an expert listener in a subjective audio quality test. "
        "Listen to the audio clip and judge its overall perceived quality on the "
        "ACR scale: 1 = bad, 2 = poor, 3 = fair, 4 = good, 5 = excellent. "
        "First describe briefly the quality problems you hear, if any (for example "
        "noise, distortion, coloration, band-limitation, dropouts, unnatural "
        "voice), then end with one line in exactly this form:\n"
        "Score: <a number from 1.0 to 5.0 with one decimal>"
    ),
    # score only, to measure what the explanation adds
    "direct": (
        "You are an expert listener in a subjective audio quality test. "
        "Rate the overall perceived quality of this audio clip on the ACR scale "
        "(1 = bad, 2 = poor, 3 = fair, 4 = good, 5 = excellent). "
        "Reply with only one line: Score: <a number from 1.0 to 5.0 with one decimal>"
    ),
}

SCORE_RE = re.compile(r"score\s*[:：]\s*\**\s*([1-5](?:\.\d+)?)", re.IGNORECASE)
NUM_RE = re.compile(r"(?<![\d.])([1-5](?:\.\d+)?)(?![\d.])")


def parse_score(text):
    """Last 'Score: x' in the answer; falls back to the last 1-5 number."""
    answer = text.split("</think>")[-1]  # Thinking model: ignore the think block
    hits = SCORE_RE.findall(answer)
    if not hits:
        hits = NUM_RE.findall(answer[-200:])
    if not hits:
        return np.nan
    s = float(hits[-1])
    return s if 1.0 <= s <= 5.0 else np.nan


def metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return {
        "SRCC": spearmanr(pred, true)[0],
        "PLCC": pearsonr(pred, true)[0],
        "KTAU": kendalltau(pred, true)[0],
        "MSE": float(np.mean((pred - true) ** 2)),
    }


def load_sets(args):
    """Returns [(name, DataFrame with wav_path, mos, system)]."""
    sets = []
    if args.nisqa_root:
        for name in NISQA_SETS:
            d = os.path.join(args.nisqa_root, f"NISQA_TEST_{name}")
            csv = os.path.join(d, f"NISQA_TEST_{name}_file.csv")
            if not os.path.isfile(csv):
                print(f"[skip] {csv} not found")
                continue
            df = pd.read_csv(csv)
            sets.append((name, pd.DataFrame({
                "wav_path": [os.path.join(d, "deg", f) for f in df["filename_deg"]],
                "mos": df["mos"], "system": df["con"]})))
    for csv in args.csv or []:
        df = pd.read_csv(csv)
        if "system" not in df:
            df["system"] = None
        base = os.path.dirname(os.path.abspath(csv))
        df["wav_path"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in df["wav_path"]]
        sets.append((os.path.splitext(os.path.basename(csv))[0], df[["wav_path", "mos", "system"]]))
    if args.limit:
        sets = [(n, df.head(args.limit)) for n, df in sets]
    return sets


class Qwen3OmniJudge:
    def __init__(self, model, tp, max_tokens, max_model_len):
        from transformers import Qwen3OmniMoeProcessor
        from vllm import LLM, SamplingParams

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model)
        self.llm = LLM(
            model=model, trust_remote_code=True, gpu_memory_utilization=0.9,
            tensor_parallel_size=tp, max_model_len=max_model_len, max_num_seqs=32,
            # image must stay > 0: in vLLM 0.25 setting image and video both to 0 skips
            # the vision tower, which leaves the deepstack buffer on the meta device and
            # crashes the profile run ("Tensor on device meta"). We never send images.
            limit_mm_per_prompt={"audio": 1, "image": 1, "video": 0}, seed=1234,
        )
        # greedy decoding so numbers are reproducible
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def __call__(self, wav_paths, prompt):
        from qwen_omni_utils import process_mm_info

        inputs = []
        for p in wav_paths:
            messages = [{"role": "user", "content": [
                {"type": "audio", "audio": p},
                {"type": "text", "text": prompt},
            ]}]
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
        df = df.assign(response=texts, pred=[parse_score(t) for t in texts])
        df.to_csv(os.path.join(out_dir, f"pred_{name}.csv"), index=False)

        ok = df.dropna(subset=["pred"])
        row = {"set": name, "n_files": len(df), "n_unparsed": len(df) - len(ok)}
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
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    help="HF id or local folder (Instruct or Thinking)")
    ap.add_argument("--nisqa-root")
    ap.add_argument("--csv", nargs="*", help="CSV(s) with wav_path, mos[, system]")
    ap.add_argument("--out-dir", default="results/q3o_instruct")
    ap.add_argument("--prompt", choices=PROMPTS, default="reason")
    ap.add_argument("--tp", type=int, default=None,
                    help="tensor parallel size (default: all visible GPUs)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="512 is enough for Instruct; use ~4096 for Thinking")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=256,
                    help="clips sent to vLLM per call (progress granularity)")
    ap.add_argument("--limit", type=int, help="first N clips per set (smoke test)")
    args = ap.parse_args()
    if not args.nisqa_root and not args.csv:
        ap.error("give --nisqa-root and/or --csv")
    os.makedirs(args.out_dir, exist_ok=True)

    sets = load_sets(args)
    import torch
    tp = args.tp or torch.cuda.device_count()
    print(f"Loading {args.model} with tensor_parallel_size={tp} ...")
    judge = Qwen3OmniJudge(args.model, tp, args.max_tokens, args.max_model_len)
    evaluate(judge, sets, PROMPTS[args.prompt], args.out_dir, args.batch)
    # vLLM's tensor-parallel workers sometimes hang on interpreter shutdown;
    # everything is saved by now, so exit hard.
    os._exit(0)


if __name__ == "__main__":
    main()
