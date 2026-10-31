#!/usr/bin/env python3
"""Build the ms-swift training files for the NISQA GRPO pilot.

Writes, from the NISQA_Corpus folder:
    grpo_train.jsonl   NISQA_TRAIN_SIM + NISQA_TRAIN_LIVE (11,020 clips), prompt only
    grpo_val.jsonl     200 clips (100 VAL_SIM + 100 VAL_LIVE), for the in-training eval
    sft_train.jsonl    same clips as grpo_train, with the target "<answer>x.xx</answer>"
    sft_val.jsonl      same clips as grpo_val, with the target

Every row keeps the human MOS in "mos", the source listening test in "subset" and a
unique "clip_id"; the reward functions read all three. For NISQA all clips come from
the same test protocol, so subset is "NISQA" for every row and the ranking reward may
pair any two clips. When we pool several datasets later, subset becomes the dataset
name and the ranking reward only pairs clips that share it.

Usage:
    python make_nisqa_jsonl.py --nisqa-root $B/dataset/NISQA/NISQA_Corpus --out $B/pilot/data
"""

import argparse
import json
import os

import pandas as pd

from prompts import GRPO_PROMPT, SFT_PROMPT, SYSTEM

TRAIN = ["NISQA_TRAIN_SIM", "NISQA_TRAIN_LIVE"]
VAL = ["NISQA_VAL_SIM", "NISQA_VAL_LIVE"]


def load(root, name):
    d = os.path.join(root, name)
    df = pd.read_csv(os.path.join(d, f"{name}_file.csv"))
    df = pd.DataFrame({
        "wav_path": [os.path.abspath(os.path.join(d, "deg", f)) for f in df["filename_deg"]],
        "mos": df["mos"].astype(float),
        "db": name,
    })
    missing = (~df["wav_path"].map(os.path.isfile)).sum()
    print(f"{name:18s} {len(df):6d} clips, {missing} missing, MOS {df.mos.min():.2f}-{df.mos.max():.2f}")
    return df


def write(df, path, sft):
    with open(path, "w") as f:
        for r in df.itertuples():
            prompt = SFT_PROMPT if sft else GRPO_PROMPT
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "<audio>" + prompt}]
            if sft:
                msgs.append({"role": "assistant", "content": f"<answer>{r.mos:.2f}</answer>"})
            f.write(json.dumps({"messages": msgs, "audios": [r.wav_path], "mos": round(r.mos, 4),
                                "subset": "NISQA", "db": r.db,
                                "clip_id": f"{r.db}/{os.path.basename(r.wav_path)}"}) + "\n")
    print(f"wrote {len(df):6d} rows to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nisqa-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-per-set", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = pd.concat([load(args.nisqa_root, n) for n in TRAIN], ignore_index=True)
    val = pd.concat([load(args.nisqa_root, n).sample(args.val_per_set, random_state=args.seed)
                     for n in VAL], ignore_index=True)
    bad = (~train["wav_path"].map(os.path.isfile)).sum() + (~val["wav_path"].map(os.path.isfile)).sum()
    if bad:
        raise SystemExit(f"{bad} audio files are missing; fix --nisqa-root before training")

    write(train, os.path.join(args.out, "grpo_train.jsonl"), sft=False)
    write(val, os.path.join(args.out, "grpo_val.jsonl"), sft=False)
    write(train, os.path.join(args.out, "sft_train.jsonl"), sft=True)
    write(val, os.path.join(args.out, "sft_val.jsonl"), sft=True)


if __name__ == "__main__":
    main()
