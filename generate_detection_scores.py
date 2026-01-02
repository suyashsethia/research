# generate_dummy_detection_scores.py
# SYNTHETIC / PLACEHOLDER generator for:
#  - RAiDAR-like score (via edit-ratio proxy)
#  - LLM-as-judge p(AI) proxy
#
# These are NOT real experimental outputs. Use only for paper drafting / pipeline testing.
# When you compute real scores, overwrite these columns and remove DUMMY_RESULTS flag.

from __future__ import annotations

import os
import re
import json
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def canonicalize_model(raw: str) -> str:
    """
    Canonicalize model id strings from your filenames/CSVs into stable keys.

    Examples:
      gpt_4o / gpt-4o -> gpt_4o
      gpt_4_5 / gpt-4.5 -> gpt_4_5
      gemini_2_5_flash / gemini-2.5-flash -> gemini_2_5_flash
      mistralai_mistral_7b_instruct_v0_2 / mistralai/Mistral-7B-Instruct-v0.2 -> mistralai_mistral_7b_instruct_v0_2
    """
    s = (raw or "").strip().lower()
    if not s:
        return ""

    # normalize separators
    s = s.replace("/", "_").replace("-", "_").replace(".", "_")
    s = re.sub(r"_+", "_", s).strip("_")

    # map common variants
    if "gpt_4o" in s:
        return "gpt_4o"
    if "gpt_4_5" in s or "gpt45" in s:
        return "gpt_4_5"
    if "gemini_2_5_flash" in s:
        return "gemini_2_5_flash"
    if "gemini_2_5_pro" in s:
        return "gemini_2_5_pro"
    if s in {"qwen_3", "qwen3"} or "qwen_3" in s:
        return "qwen_3"
    if "gemma_3_12b_it" in s:
        return "gemma_3_12b_it"
    if "mistral_7b_instruct_v0_2" in s:
        return "mistralai_mistral_7b_instruct_v0_2"

    return s


def canonicalize_tech(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    # your pipeline uses these exact tokens
    if s in {"zero-shot", "few-shot", "chain-of-thought"}:
        return s
    # try to normalize mild variants
    s = s.replace("_", "-")
    if s in {"zeroshot"}:
        return "zero-shot"
    if s in {"fewshot"}:
        return "few-shot"
    if s in {"cot", "chainofthought", "chain-of-thought"}:
        return "chain-of-thought"
    return s


def canonicalize_lang(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in {"hindi", "bangla", "tamil", "telugu"}:
        return s
    return s


# -----------------------------
# Literature-informed priors (dummy)
# Higher offset => MORE detectable (easier to classify as AI).
# -----------------------------
MODEL_OFFSET = {
    # least detectable (harder)
    "gpt_4_5": -0.06,
    "gpt_4o": -0.05,
    "gemini_2_5_pro": -0.05,
    "gemini_2_5_flash": -0.04,

    # middle
    "qwen_3": 0.00,

    # more detectable (easier)
    "gemma_3_12b_it": +0.03,
    "mistralai_mistral_7b_instruct_v0_2": +0.05,
}

LANG_OFFSET = {
    # harder (typically more fluent / more data)
    "hindi":  -0.02,
    "bangla": -0.02,

    # easier (often more artifacts)
    "tamil":  +0.02,
    "telugu": +0.03,
}

PROMPT_OFFSET = {
    # more generic -> easier to detect
    "zero-shot": +0.02,

    # neutral
    "chain-of-thought": +0.00,

    # more anchored to examples -> harder to detect
    "few-shot": -0.03,
}


def interaction_offset(language: str, model: str, technique: str) -> float:
    """
    Small interaction bumps to create clearer "extreme" buckets.
    """
    bonus = 0.0

    # open/smaller + lower-resource language + zero-shot => more detectable
    if language in {"telugu", "tamil"} and model in {
        "mistralai_mistral_7b_instruct_v0_2", "gemma_3_12b_it"
    } and technique == "zero-shot":
        bonus += 0.02

    # closed/strong + high-resource language + few-shot => less detectable
    if language in {"hindi", "bangla"} and model in {
        "gpt_4_5", "gpt_4o", "gemini_2_5_pro"
    } and technique == "few-shot":
        bonus -= 0.01

    return bonus


def compute_detectability_logit(
    df: pd.DataFrame,
    seed: int,
    ai_logit_base: float = 0.90,
    human_logit_base: float = -0.90,
    noise_std: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - detectability_logit (latent)
      - detectability_prior (additive prior part only)
    """
    rng = np.random.default_rng(seed)

    lang = df["language"].astype(str).map(canonicalize_lang)
    model_raw = df.get("model", pd.Series([""] * len(df))).fillna("").astype(str)
    tech_raw = df.get("prompting_technique", pd.Series([""] * len(df))).fillna("").astype(str)

    model = model_raw.map(canonicalize_model)
    tech = tech_raw.map(canonicalize_tech)

    is_ai = df["source"].astype(str).str.lower().eq("ai").to_numpy()

    # Baseline logit by source (AI tends to be more detectable than human)
    logit = np.where(is_ai, ai_logit_base, human_logit_base).astype(float)

    # Priors apply mainly to AI rows (human has model/tech blank anyway)
    prior = np.zeros(len(df), dtype=float)
    prior += lang.map(LANG_OFFSET).fillna(0.00).to_numpy()
    prior += model.map(MODEL_OFFSET).fillna(0.00).to_numpy()
    prior += tech.map(PROMPT_OFFSET).fillna(0.00).to_numpy()

    # Interaction (only meaningful when fields exist)
    inter = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        if is_ai[i]:
            inter[i] = interaction_offset(lang.iloc[i], model.iloc[i], tech.iloc[i])
    prior += inter

    # Mild length effect: very short articles are easier to flag
    tok = df.get("gpt2_token_count", pd.Series([0] * len(df))).fillna(0).astype(float).to_numpy()
    shortness = np.where(tok > 0, (200.0 - np.minimum(tok, 200.0)) / 200.0, 0.0)  # 0..1
    logit += 0.15 * shortness

    # Add priors + noise
    logit = logit + prior + rng.normal(0.0, noise_std, size=len(df))

    return logit, prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_dir", type=str, default="stats_out")
    ap.add_argument("--ai_csv", type=str, default="ai_samples.csv")
    ap.add_argument("--human_csv", type=str, default="human_samples.csv")
    ap.add_argument("--out_csv", type=str, default="dummy_detection_scores.csv")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--use_human", action="store_true", help="Include human samples if present")
    ap.add_argument("--fixed_n_human", type=int, default=0, help="Optional cap on total human rows (0 = no cap)")
    args = ap.parse_args()

    ai_path = os.path.join(args.stats_dir, args.ai_csv)
    if not os.path.exists(ai_path):
        raise FileNotFoundError(f"Missing {ai_path}")

    ai = pd.read_csv(ai_path)
    ai["source"] = "ai"

    frames = [ai]

    if args.use_human:
        human_path = os.path.join(args.stats_dir, args.human_csv)
        if not os.path.exists(human_path):
            raise FileNotFoundError(f"Missing {human_path}")
        human = pd.read_csv(human_path)
        human["source"] = "human"

        # Optional: cap human to avoid heavy files
        if args.fixed_n_human and len(human) > args.fixed_n_human:
            human = human.sample(n=args.fixed_n_human, random_state=args.seed).reset_index(drop=True)

        frames.append(human)

    df = pd.concat(frames, ignore_index=True)

    # Minimum columns
    if "language" not in df.columns:
        raise ValueError("Missing column: language")
    if "source" not in df.columns:
        raise ValueError("Missing column: source")
    if "gpt2_token_count" not in df.columns:
        # allow missing but strongly recommended
        df["gpt2_token_count"] = 0

    # Latent detectability
    detect_logit, detect_prior = compute_detectability_logit(df, seed=args.seed)
    detectability = sigmoid(detect_logit)  # 0..1; higher => easier detection / more AI-likeness

    rng = np.random.default_rng(args.seed)

    # --- RAiDAR proxy
    # RAiDAR idea: human gets rewritten more (higher edit ratio), AI less (lower edit ratio). :contentReference[oaicite:1]{index=1}
    # Make edit ratio negatively correlated with detectability.
    edit_mean = 0.55 - 0.35 * detectability
    edit_ratio = np.clip(rng.normal(edit_mean, 0.06, size=len(df)), 0.05, 0.85)
    raidar_score = 1.0 - edit_ratio  # higher => more AI-likely

    # --- LLM-as-judge proxy
    # Probability that judge says "AI", correlated with detectability + noise.
    judge_logit = 4.5 * (detectability - 0.50) + rng.normal(0.0, 0.85, size=len(df))
    judge_p_ai = sigmoid(judge_logit)

    # Combined (simple ensemble)
    combined_p_ai = np.clip(0.5 * raidar_score + 0.5 * judge_p_ai, 0.0, 1.0)

    out = df.copy()
    out["DUMMY_RESULTS"] = 1
    out["dummy_priors_version"] = "v2_lit_priors_2025-12-26"

    # keep latent signals for debugging
    out["detectability_logit_dummy"] = detect_logit
    out["detectability_prior_dummy"] = detect_prior
    out["detectability_dummy"] = detectability

    # final dummy scores
    out["raidar_edit_ratio_dummy"] = edit_ratio
    out["raidar_score_dummy"] = raidar_score
    out["judge_p_ai_dummy"] = judge_p_ai
    out["combined_p_ai_dummy"] = combined_p_ai

    out_path = os.path.join(args.stats_dir, args.out_csv)
    out.to_csv(out_path, index=False)

    print(f"[OK] Wrote synthetic scores to: {out_path}")
    print("     Added: raidar_score_dummy, judge_p_ai_dummy, combined_p_ai_dummy")
    print("     NOTE: These are DUMMY results; do not publish as real experimental findings.")


if __name__ == "__main__":
    main()
