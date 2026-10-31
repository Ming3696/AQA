"""Reward functions for audio-quality GRPO (ms-swift plugin).

Ported from VQAThinker (grpo_jsonl.py), with two changes for audio:
  * unparseable answers get reward 0 instead of a random score (VQAThinker's random
    fallback injects noise early in training);
  * the ranking reward only pairs clips from the same listening test ("subset" column),
    because MOS from different tests is not on the same scale.

Registered names (use with --reward_funcs):
    aqa_format  1 if the answer is "<think>...</think><answer>x</answer>" with x in [1, 5]
    aqa_reg     Gaussian closeness to MOS, exp(-(s - g)^2 / 2 sigma^2), sigma = 0.5;
                x0.8 if the number has fewer than 2 decimals (pushes fine-grained scores)
    aqa_rank    fidelity reward on pairs of clips (Thurstone model), averaged over the
                other clips of the same subset in the local batch

Load with:  --external_plugins aqa_rewards.py
Env vars:   AQA_RANK_SCOPE=subset (default) | all   pair only same-subset clips, or any two
            AQA_REG_SIGMA=0.5
"""

import math
import os
import re
from collections import defaultdict
from typing import List

ANSWER_RE = re.compile(r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>", re.S)
FORMAT_RE = re.compile(r"^\s*<think>.+?</think>\s*<answer>\s*[-+]?\d+(?:\.\d+)?\s*</answer>\s*$", re.S)
EPS = 1e-8


def parse_answer(text):
    """Returns (score, n_decimals) or (None, 0). Score must be in [1, 5]."""
    m = ANSWER_RE.findall(text or "")
    if not m:
        return None, 0
    raw = m[-1]
    s = float(raw)
    if not 1.0 <= s <= 5.0:
        return None, 0
    return s, len(raw.split(".")[1].rstrip("0")) if "." in raw else 0


def format_rewards(completions):
    return [1.0 if FORMAT_RE.match(c or "") and parse_answer(c)[0] is not None else 0.0
            for c in completions]


def reg_rewards(completions, mos, sigma=0.5):
    out = []
    for c, g in zip(completions, mos):
        s, dec = parse_answer(c)
        if s is None:
            out.append(0.0)
            continue
        r = math.exp(-((s - float(g)) ** 2) / (2 * sigma ** 2))
        out.append(r * (0.8 if dec < 2 else 1.0))
    return out


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rank_rewards(completions, mos, clip_ids, subsets, scope="subset"):
    """VQAThinker's pairwise fidelity reward.

    For completion k of clip i with score s_ik and every other clip j in the batch:
        p   = Phi((s_ik - mu_j) / sqrt(var_i + var_j + eps))   mu, var over each clip's samples
        gt  = 1 if g_i > g_j, 0 if g_i < g_j, 0.5 if equal
        r   = sqrt(p * gt + eps) + sqrt((1 - p) * (1 - gt) + eps)
    averaged over j. Clips with no valid partner get 0 for every sample (no effect on the
    group-normalized advantage).
    """
    scores = [parse_answer(c)[0] for c in completions]
    groups = defaultdict(list)
    for idx, cid in enumerate(clip_ids):
        groups[cid].append(idx)

    stats = {}  # clip -> (mu, var, mos, subset) from its parsed samples
    for cid, idxs in groups.items():
        vals = [scores[i] for i in idxs if scores[i] is not None]
        if not vals:
            continue
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        stats[cid] = (mu, var, float(mos[idxs[0]]), subsets[idxs[0]])

    out = []
    for idx, (s, cid) in enumerate(zip(scores, clip_ids)):
        if s is None or cid not in stats:
            out.append(0.0)
            continue
        _, var_i, g_i, sub_i = stats[cid]
        rs = []
        for cj, (mu_j, var_j, g_j, sub_j) in stats.items():
            if cj == cid or (scope == "subset" and sub_j != sub_i):
                continue
            p = _phi((s - mu_j) / math.sqrt(var_i + var_j + EPS))
            gt = 1.0 if g_i > g_j else 0.0 if g_i < g_j else 0.5
            rs.append(math.sqrt(p * gt + EPS) + math.sqrt((1 - p) * (1 - gt) + EPS))
        out.append(sum(rs) / len(rs) if rs else 0.0)
    return out


def _clip_ids(kwargs, n):
    """Group samples by clip. Uses the dataset's clip_id column, else the audio path.

    Not prompt_id: ms-swift derives it from the message text only, and every clip has
    the same prompt text, so all clips in a batch would share one prompt_id.
    """
    for key in ("clip_id", "audios"):
        ids = kwargs.get(key)
        if ids is not None and len(ids) == n and all(i is not None for i in ids):
            return [str(i) for i in ids]
    raise ValueError("aqa_rank needs a clip_id column (or audios) to group samples by clip")


try:
    from swift.rewards import ORM, orms
except ImportError:  # allows unit tests without ms-swift installed
    ORM, orms = object, {}


class AQAFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        return format_rewards(completions)


class AQAReg(ORM):
    def __call__(self, completions, mos, **kwargs) -> List[float]:
        return reg_rewards(completions, mos, float(os.environ.get("AQA_REG_SIGMA", 0.5)))


class AQARank(ORM):
    def __call__(self, completions, mos, **kwargs) -> List[float]:
        n = len(completions)
        subsets = kwargs.get("subset") or ["all"] * n
        scope = os.environ.get("AQA_RANK_SCOPE", "subset")
        return rank_rewards(completions, mos, _clip_ids(kwargs, n), subsets, scope)


orms["aqa_format"] = AQAFormat
orms["aqa_reg"] = AQAReg
orms["aqa_rank"] = AQARank
