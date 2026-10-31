#!/usr/bin/env python3
"""Zero-shot evaluation of a SHEET SSL-MOS model on any set of test CSVs.

Replaces eval_sheet_nisqa.py (still works the same with --nisqa-root) and adds:
  * --csv for any test set. Accepts SHEET's own CSVs (wav_path, system_id,
    avg_score or score; made by SHEET's local/data_prep.py; listener-level rows
    are averaged per clip) and the
    format of eval_qwen3omni_zeroshot.py (wav_path, mos[, system]), so both
    baselines read the same files.
  * a "paper" column: for test sets that are in MOS-Bench, the UTT-SRCC/MSE
    printed in MOS-Bench Fig. 2 (arXiv 2411.03715v2) for the same model, so
    the sanity check is a single look at the table.
  * --chunk-sec for long clips (music, general audio).

Setup: same environment as eval_sheet_nisqa.py (Python 3.10, requirements_sheet.txt).

Usage:
    python eval_sheet.py --nisqa-root $R/NISQA/NISQA_Corpus \
        --csv csv/bvcc_test.csv csv/somos_test.csv --out-dir results/sheet_all
    python eval_sheet.py --csv csv/odaq.csv --chunk-sec 20 --out-dir results/sheet_all

Metrics follow SHEET's utils/calculate_metrics.py: utterance level, plus
system level (mean per system) when a system column is present.
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

ALL8_ID = (
    "bvcc+somos+singmos+nisqa+tmhint-qi+tencent+pstn+urgent2024-mos/"
    "sslmos-wavlm_large/1337"
)
NISQA_ID = "nisqa/sslmos-wavlm_large/1337"
NISQA_SETS = ["FOR", "P501", "LIVETALK", "NSC"]
FS = 16000
MIN_LEN = 1040  # same minimum length as SHEET's hubconf

# MOS-Bench Fig. 2 (arXiv 2411.03715v2), utterance level, (SRCC, MSE).
# Keys are the test-set names this script uses: the CSV file name without
# .csv (SHEET's data_prep names), or NISQA_<set> for --nisqa-root.
PAPER = {
    ALL8_ID: {
        "bvcc_test": (0.851, 0.167), "somos_test": (0.663, 0.035),
        "bc19_test": (0.629, 0.398), "vmc23_track1a_test": (0.500, 1.014),
        "vmc23_track1b_test": (0.603, 0.440), "vmc23_track2_test": (0.472, 0.369),
        "singmos_test": (0.537, 0.087), "brspeechmos_test": (0.635, 0.022),
        "hablamos_test": (0.687, 2.115), "ttsds2_test": (0.401, 0.520),
        "NISQA_FOR": (0.916, 0.124), "NISQA_LIVETALK": (0.886, 0.229),
        "NISQA_P501": (0.944, 0.144), "tmhintqi_test": (0.750, 0.444),
        "vmc23_track3_test": (0.761, 0.327), "tcd_voip_test": (0.818, 0.448),
        "vmc24_track3_test": (0.717, 0.592),
    },
    NISQA_ID: {
        "bvcc_test": (0.776, 0.281), "somos_test": (0.355, 0.420),
        "bc19_test": (0.541, 0.809), "vmc23_track1a_test": (0.525, 0.351),
        "vmc23_track1b_test": (0.592, 0.318), "vmc23_track2_test": (0.568, 0.252),
        "singmos_test": (0.406, 0.682), "brspeechmos_test": (0.585, 0.339),
        "hablamos_test": (0.710, 1.294), "ttsds2_test": (0.333, 0.564),
        "NISQA_FOR": (0.932, 0.099), "NISQA_LIVETALK": (0.901, 0.174),
        "NISQA_P501": (0.940, 0.141), "tmhintqi_test": (0.533, 2.325),
        "vmc23_track3_test": (0.494, 1.821), "tcd_voip_test": (0.773, 0.524),
        "vmc24_track3_test": (0.763, 0.525),
    },
}
# SHEET's own CSV names (egs/BENCHMARKS/run_*_test.sh), so either route finds the paper number
_ALIASES = {"bc23a_test": "vmc23_track1a_test", "bc23b_test": "vmc23_track1b_test",
            "svcc23_test": "vmc23_track2_test", "tmhint_qi_s_test": "vmc23_track3_test"}
for _m in PAPER.values():
    for _s in ["FOR", "P501", "LIVETALK"]:
        _m[f"nisqa_{_s}"] = _m[f"NISQA_{_s}"]
    for _a, _k in _ALIASES.items():
        _m[_a] = _m[_k]


def load_wav(path):
    # soundfile instead of torchaudio.load: newer torchaudio versions moved
    # loading to torchcodec, which is often not installed.
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))  # mono
    if sr != FS:
        wav = AF.resample(wav, sr, FS)
    if wav.shape[0] < MIN_LEN:
        pad = (MIN_LEN - wav.shape[0]) // 2 + 1
        wav = F.pad(wav, (pad, pad))
    return wav


def load_sheet_model(model_id, device):
    # Same steps as sheet_ssqa() in SHEET's hubconf.py, done directly so we
    # don't depend on torch.hub (GitHub API calls, and the broken v0.2.5 tag).
    from huggingface_hub import hf_hub_download
    from sheet.models.sslmos import SSLMOS
    
    from s3prl.util.download import set_dir
    # 对应你手动下载 WavLM 的目录，注意包含 download
    cache_dir = "/share/panjiayi/cache/s3prl/download"
    os.makedirs(cache_dir, exist_ok=True)
    set_dir(cache_dir)

    conf_path = hf_hub_download("unilight/sheet-models", f"{model_id}/config.yml")
    ckpt_path = hf_hub_download("unilight/sheet-models", f"{model_id}/checkpoint-best.bin")
    with open(conf_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    if cfg["model_type"] != "SSLMOS":
        raise ValueError(f"unsupported model_type {cfg['model_type']}")
    model = SSLMOS(cfg["model_input"], **cfg["model_params"])
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location="cpu"))
    return model.to(device).eval(), cfg


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
            sets.append((f"NISQA_{name}", pd.DataFrame({
                "wav_path": [os.path.join(d, "deg", f) for f in df["filename_deg"]],
                "mos": df["mos"], "system": df["con"]})))
    for csv in args.csv or []:
        df = pd.read_csv(csv)
        df = df.rename(columns={"avg_score": "mos", "system_id": "system"})
        if "mos" not in df and "score" in df:
            df = df.rename(columns={"score": "mos"})
        if "mos" not in df:
            raise ValueError(f"{csv}: needs a mos, avg_score or score column")
        if "system" not in df:
            df["system"] = None
        if df["wav_path"].duplicated().any():
            # listener-level rows (some SHEET data_prep outputs): average to one MOS per clip
            n = len(df)
            df = df.groupby("wav_path", sort=False).agg(
                mos=("mos", "mean"), system=("system", "first")).reset_index()
            print(f"[{os.path.basename(csv)}] averaged {n} rating rows into {len(df)} clips")
        base = os.path.dirname(os.path.abspath(csv))
        df["wav_path"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in df["wav_path"]]
        sets.append((os.path.splitext(os.path.basename(csv))[0],
                     df[["wav_path", "mos", "system"]].reset_index(drop=True)))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nisqa-root", help="path to the unzipped NISQA_Corpus folder")
    ap.add_argument("--csv", nargs="*", help="test CSV(s): wav_path, avg_score|mos[, system_id|system]")
    ap.add_argument("--model-id", default=ALL8_ID,
                    help="SHEET model _id, see unilight.github.io/sheet/pretrained-models")
    ap.add_argument("--out-dir", default="results/sheet_all")
    ap.add_argument("--chunk-sec", type=float, default=0,
                    help="split clips longer than this into chunks and take the "
                         "length-weighted mean score (0 = whole clip, like SHEET)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None,
                    help="only score the first N files per set (quick test)")
    args = ap.parse_args()
    if not args.nisqa_root and not args.csv:
        ap.error("give --nisqa-root and/or --csv")
    os.makedirs(args.out_dir, exist_ok=True)

    sets = load_sets(args)
    print(f"Loading SHEET model {args.model_id} on {args.device} ...")
    model, cfg = load_sheet_model(args.model_id, args.device)
    key = cfg["model_input"]
    paper = PAPER.get(args.model_id, {})

    @torch.no_grad()
    def predict_one(wav):
        x = wav.to(args.device).unsqueeze(0)
        inputs = {key: x, key + "_lengths": x.new_tensor([x.size(1)]).long()}
        if cfg["inference_mode"] == "mean_listener":
            out = model.mean_listener_inference(inputs)
        else:
            out = model.mean_net_inference(inputs)
        return float(out["scores"].reshape(-1)[0].cpu())

    def predict(wav):
        n = int(args.chunk_sec * FS)
        if n <= 0 or wav.shape[0] <= n:
            return predict_one(wav)
        chunks = [wav[i:i + n] for i in range(0, wav.shape[0], n)]
        chunks = [c for c in chunks if c.shape[0] >= FS] or chunks[:1]  # drop <1 s tails
        w = np.array([c.shape[0] for c in chunks], float)
        return float(np.dot([predict_one(c) for c in chunks], w / w.sum()))

    rows = []
    for name, df in sets:
        if args.limit:
            df = df.head(args.limit).copy()
        preds = []
        for p in tqdm(df["wav_path"], desc=name):
            try:
                preds.append(predict(load_wav(p)))
            except Exception as e:  # keep going; report how many failed
                print(f"[error] {p}: {e}")
                preds.append(np.nan)
        df["pred"] = preds
        df.to_csv(os.path.join(args.out_dir, f"pred_{name}.csv"), index=False)

        ok = df.dropna(subset=["pred"])
        row = {"set": name, "n_files": len(df), "n_failed": len(df) - len(ok)}
        row.update({f"utt_{k}": v for k, v in metrics(ok["pred"], ok["mos"]).items()})
        if ok["system"].notna().all() and ok["system"].nunique() > 2:
            sys_df = ok.groupby("system")[["pred", "mos"]].mean()
            row["n_systems"] = len(sys_df)
            row.update({f"sys_{k}": v for k, v in metrics(sys_df["pred"], sys_df["mos"]).items()})
        if name in paper and not args.limit:
            row["paper_utt_SRCC"], row["paper_utt_MSE"] = paper[name]
            row["diff_SRCC"] = row["utt_SRCC"] - paper[name][0]
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + table.round(3).to_string(index=False))
    if "diff_SRCC" in table:
        bad = table[table["diff_SRCC"].abs() > 0.01]
        print("\nSanity check vs MOS-Bench Fig. 2: " + (
            "all sets within 0.01 SRCC." if bad.empty else
            f"{len(bad)} set(s) off by more than 0.01 SRCC: {', '.join(bad['set'])}"))
    print(f"\nSaved per-file predictions and summary.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
