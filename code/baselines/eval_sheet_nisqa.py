#!/usr/bin/env python3
"""Zero-shot evaluation of a SHEET SSL-MOS model on the NISQA test sets.

Runs a pre-trained SHEET model on NISQA_TEST_FOR / P501 / LIVETALK (and NSC if
present), then reports utterance-level and system-level (per condition) SRCC,
PLCC, KTAU and MSE against the human MOS.

Setup (once). Use Python 3.10: s3prl, which SHEET depends on, fails on 3.11+
and on torchaudio >= 2.9.
    conda create -n sheet python=3.10 -y && conda activate sheet
    pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements_sheet.txt

First run downloads two files from Hugging Face (~2.4 GB total): the SHEET
checkpoint and the WavLM-large backbone. Set HF_ENDPOINT=https://hf-mirror.com
if huggingface.co is slow.

Usage:
    python eval_sheet_nisqa.py \
        --nisqa-root /share/panjiayi/chengqiming/AQA/dataset/NISQA/NISQA_Corpus \
        --out-dir results/sheet_default

    # in-domain reference: the SHEET model trained on NISQA only
    python eval_sheet_nisqa.py --nisqa-root ... \
        --model-id nisqa/sslmos-wavlm_large/1337 --out-dir results/sheet_nisqa_only

Note: the model is loaded straight from Hugging Face (unilight/sheet-models)
rather than via torch.hub; the README's v0.2.5 hub tag has a syntax error.
"""

import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import yaml
from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm import tqdm

from s3prl.util.download import set_dir

DEFAULT_ID = (
    "bvcc+somos+singmos+nisqa+tmhint-qi+tencent+pstn+urgent2024-mos/"
    "sslmos-wavlm_large/1337"
)
TEST_SETS = ["FOR", "P501", "LIVETALK", "NSC"]
FS = 16000
MIN_LEN = 1040  # same minimum length as SHEET's hubconf


def load_wav(path, device):
    # soundfile instead of torchaudio.load: newer torchaudio versions moved
    # loading to torchcodec, which is often not installed.
    wav, sr = sf.read(path, dtype="float32", always_2d=True) # frames x channels, sample rate
    wav = torch.from_numpy(wav.mean(axis=1))  # mono
    if sr != FS:
        wav = AF.resample(wav, sr, FS)
    if wav.shape[0] < MIN_LEN:
        pad = (MIN_LEN - wav.shape[0]) // 2 + 1
        wav = F.pad(wav, (pad, pad))
    return wav.to(device)


def load_sheet_model(model_id, device):
    # Same steps as sheet_ssqa() in SHEET's hubconf.py, done directly so we
    # don't depend on torch.hub (GitHub API calls, and the broken v0.2.5 tag).
    from huggingface_hub import hf_hub_download

    # 对应你手动下载 WavLM 的目录，注意包含 download
    cache_dir = "/share/panjiayi/cache/s3prl/download"
    os.makedirs(cache_dir, exist_ok=True)
    set_dir(cache_dir)

    from sheet.models.sslmos import SSLMOS

    conf_path = hf_hub_download("unilight/sheet-models", f"{model_id}/config.yml")
    ckpt_path = hf_hub_download(
        "unilight/sheet-models", f"{model_id}/checkpoint-best.bin"
    )
    with open(conf_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    if cfg["model_type"] != "SSLMOS":
        raise ValueError(f"unsupported model_type {cfg['model_type']}")
    model = SSLMOS(cfg["model_input"], **cfg["model_params"])
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
    return model.to(device).eval(), cfg


def metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    return {
        "SRCC": spearmanr(pred, true)[0],
        "PLCC": pearsonr(pred, true)[0],
        "KTAU": kendalltau(pred, true)[0],
        "MSE": float(np.mean((pred - true) ** 2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nisqa-root", required=True, help="path to the unzipped NISQA_Corpus folder"
    )
    ap.add_argument(
        "--model-id",
        default=DEFAULT_ID,
        help="SHEET model _id, see unilight.github.io/sheet/pretrained-models",
    )
    ap.add_argument("--out-dir", default="results/sheet_default")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only score the first N files per set (quick test)",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading SHEET model {args.model_id} on {args.device} ...")
    model, cfg = load_sheet_model(args.model_id, args.device)
    key = cfg["model_input"]

    @torch.no_grad()
    def predict(wav):
        x = wav.unsqueeze(0)
        inputs = {key: x, key + "_lengths": x.new_tensor([x.size(1)]).long()}
        if cfg["inference_mode"] == "mean_listener":
            out = model.mean_listener_inference(inputs)
        else:
            out = model.mean_net_inference(inputs)
        return float(out["scores"].reshape(-1)[0].cpu())

    rows = []
    for name in TEST_SETS:
        set_dir = os.path.join(args.nisqa_root, f"NISQA_TEST_{name}")
        csv_path = os.path.join(set_dir, f"NISQA_TEST_{name}_file.csv")
        if not os.path.isfile(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        if args.limit:
            df = df.head(args.limit)

        preds = []
        for fn in tqdm(df["filename_deg"], desc=name):
            preds.append(
                predict(load_wav(os.path.join(set_dir, "deg", fn), args.device))
            )
        df["pred"] = preds
        df[["con", "filename_deg", "mos", "pred"]].to_csv(
            os.path.join(args.out_dir, f"pred_{name}.csv"), index=False
        )

        utt = metrics(df["pred"], df["mos"])
        sys_df = df.groupby("con")[["pred", "mos"]].mean()
        sys_m = metrics(sys_df["pred"], sys_df["mos"]) if len(sys_df) > 2 else {}
        rows.append(
            {
                "set": name,
                "n_files": len(df),
                "n_systems": len(sys_df),
                **{f"utt_{k}": v for k, v in utt.items()},
                **{f"sys_{k}": v for k, v in sys_m.items()},
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + table.round(3).to_string(index=False))
    print(f"\nSaved per-file predictions and summary.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
