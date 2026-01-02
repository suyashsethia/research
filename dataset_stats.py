# dataset_stats.py
# AI folder stats + HUMAN stats (sampled 2.5k per language)
# Memory-safe by default (does not store full text in DataFrame)

from __future__ import annotations

import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


# -----------------------------
# Filename parsing
# -----------------------------
TECHNIQUES = {"zero-shot", "few-shot", "chain-of-thought"}
LANGUAGES = {"hindi", "bangla", "tamil", "telugu"}

def parse_ai_filename(path: str) -> Optional[Dict[str, str]]:
    base = os.path.basename(path)
    if not base.endswith(".txt"):
        return None
    stem = base[:-4]
    parts = stem.split("_")
    if len(parts) < 5:
        return None

    lang = parts[0].lower()
    if lang not in LANGUAGES:
        return None

    uid = parts[-1]
    if not uid.isdigit():
        return None

    tech_idx = None
    for i in range(1, len(parts) - 1):
        if parts[i] in TECHNIQUES:
            tech_idx = i
            break
    if tech_idx is None:
        return None

    model = "_".join(parts[1:tech_idx])
    technique = parts[tech_idx]
    category = "_".join(parts[tech_idx + 1:-1])

    return {
        "language": lang,
        "model": model,
        "prompting_technique": technique,
        "category": category,
        "uid": uid,
        "filename": base,
        "path": path,
    }


# -----------------------------
# Human CSV loader (same as yours)
# -----------------------------
def load_local_dataset(data_path: str) -> pd.DataFrame:
    try:
        try:
            df = pd.read_csv(
                data_path,
                encoding="utf-8-sig",
                sep=None,
                engine="python",
                dtype=str,
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                keep_default_na=False,
                na_filter=False,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                data_path,
                encoding="utf-8",
                sep=None,
                engine="python",
                dtype=str,
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                keep_default_na=False,
                na_filter=False,
                on_bad_lines="skip",
            )

        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
            .str.lower()
        )

        alias_map = {
            "headline": {"headline", "title", "head", "news_headline"},
            "content": {"content", "article", "text", "story", "body"},
            "category": {"category", "topic", "section", "label", "class"},
        }
        rename = {}
        for target, candidates in alias_map.items():
            match = next((c for c in df.columns if c in candidates), None)
            if not match:
                raise KeyError(
                    f"Expected a column for '{target}' not found. Available: {list(df.columns)}"
                )
            rename[match] = target
        df = df.rename(columns=rename)[["headline", "content", "category"]]

        def _clean(s: str) -> str:
            s = (s or "").replace("\xa0", " ")
            s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
            return s

        for col in ["headline", "content", "category"]:
            df[col] = df[col].astype(str).map(_clean)

        mask_empty = (df["headline"] == "") | (df["content"] == "")
        df = df[~mask_empty].copy()
        df.loc[df["category"] == "", "category"] = "misc"
        df["category"] = df["category"].str.replace(r"\s+", " ", regex=True).str.strip()
        return df

    except Exception as e:
        raise RuntimeError(f"Error loading dataset from {data_path}: {e}") from e


# -----------------------------
# Text metrics
# -----------------------------
_SENT_SPLIT_RE = re.compile(r"[.!?\u0964\u0965\u09e4\u09e5]+")  # include danda variants
_WORD_RE = re.compile(r"\S+", flags=re.UNICODE)

SCRIPT_RANGES = {
    "hindi":  [(0x0900, 0x097F)],  # Devanagari
    "bangla": [(0x0980, 0x09FF)],  # Bengali
    "tamil":  [(0x0B80, 0x0BFF)],  # Tamil
    "telugu": [(0x0C00, 0x0C7F)],  # Telugu
}

def _in_ranges(cp: int, ranges: List[Tuple[int, int]]) -> bool:
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False

def compute_text_metrics(text: str, language: str, gpt2_tokenizer=None) -> Dict[str, float]:
    t = text or ""
    t_stripped = t.strip()

    n_chars = len(t)
    n_bytes = len(t.encode("utf-8", errors="ignore"))
    n_lines = t.count("\n") + 1 if t else 0
    n_paras = len([p for p in re.split(r"\n\s*\n", t) if p.strip()]) if t_stripped else 0

    words = _WORD_RE.findall(t_stripped) if t_stripped else []
    n_words = len(words)

    sent_segs = [s.strip() for s in _SENT_SPLIT_RE.split(t_stripped) if s.strip()] if t_stripped else []
    n_sents = len(sent_segs)

    avg_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    avg_sent_len_words = (n_words / n_sents) if n_sents else 0.0

    uniq_words = len(set(words)) if words else 0
    ttr = (uniq_words / n_words) if n_words else 0.0
    rep_rate = 1.0 - ttr if n_words else 0.0

    non_space = [ch for ch in t if not ch.isspace()]
    n_ns = len(non_space)

    if n_ns == 0:
        ascii_frac = latin_frac = digit_frac = punct_frac = script_frac = 0.0
    else:
        ascii_frac = sum(1 for ch in non_space if ord(ch) < 128) / n_ns
        latin_frac = sum(1 for ch in non_space if ("A" <= ch <= "Z") or ("a" <= ch <= "z")) / n_ns
        digit_frac = sum(1 for ch in non_space if ch.isdigit()) / n_ns
        punct_frac = sum(1 for ch in non_space if re.match(r"[^\w\s]", ch, flags=re.UNICODE)) / n_ns

        ranges = SCRIPT_RANGES.get(language.lower(), [])
        script_frac = (sum(1 for ch in non_space if _in_ranges(ord(ch), ranges)) / n_ns) if ranges else 0.0

    gpt2_tokens = 0
    if gpt2_tokenizer is not None and t_stripped:
        try:
            gpt2_tokens = len(gpt2_tokenizer.encode(t_stripped))
        except Exception:
            gpt2_tokens = 0

    return {
        "n_chars": n_chars,
        "n_bytes": n_bytes,
        "n_lines": n_lines,
        "n_paragraphs": n_paras,
        "n_words_ws": n_words,
        "n_sentences": n_sents,
        "avg_word_len": avg_word_len,
        "avg_sent_len_words": avg_sent_len_words,
        "ttr_ws": ttr,
        "rep_rate_ws": rep_rate,
        "ascii_frac": ascii_frac,
        "latin_frac": latin_frac,
        "digit_frac": digit_frac,
        "punct_frac": punct_frac,
        "expected_script_frac": script_frac,
        "gpt2_token_count": gpt2_tokens,
    }


# -----------------------------
# AI loader
# -----------------------------
def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_ai_folder(ai_dir: str, gpt2_tokenizer=None, max_workers: int = 24, store_text: bool = False) -> pd.DataFrame:
    files = []
    for root, _, names in os.walk(ai_dir):
        for nm in names:
            if nm.endswith(".txt"):
                files.append(os.path.join(root, nm))

    # Parse filenames
    parsed = []
    it = files
    if tqdm is not None:
        it = tqdm(files, desc="Indexing AI files")
    for p in it:
        meta = parse_ai_filename(p)
        if meta is not None:
            parsed.append(meta)

    def work(meta: Dict[str, str]) -> Dict:
        text = _read_text_file(meta["path"])
        feats = compute_text_metrics(text, meta["language"], gpt2_tokenizer=gpt2_tokenizer)
        out = dict(meta)
        out["source"] = "ai"
        if store_text:
            out["text"] = text
        out.update(feats)
        return out

    rows = []
    pbar = tqdm(total=len(parsed), desc="Reading+features (AI)") if tqdm is not None else None
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, m) for m in parsed]
        for fut in as_completed(futs):
            rows.append(fut.result())
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()

    df = pd.DataFrame(rows)
    df["__one__"] = 1
    return df


# -----------------------------
# Human loader (sampled + parallel)
# -----------------------------
def load_human_csvs_sampled(
    dataset_paths: Dict[str, str],
    gpt2_tokenizer=None,
    n_per_lang: int = 2500,
    seed: int = 42,
    max_workers: int = 24,
    store_text: bool = False,
) -> pd.DataFrame:
    rows = []

    for lang, path in dataset_paths.items():
        df = load_local_dataset(path)

        # sample BEFORE feature computation
        if len(df) > n_per_lang:
            df = df.sample(n=n_per_lang, random_state=seed).reset_index(drop=True)

        # parallel feature computation
        def work(idx: int) -> Dict:
            r = df.iloc[idx]
            text = r["content"]
            feats = compute_text_metrics(text, lang, gpt2_tokenizer=gpt2_tokenizer)
            out = {
                "source": "human",
                "language": lang,
                "category": r["category"],
                "headline": r["headline"],
                "model": None,
                "prompting_technique": None,
                "uid": None,
                "filename": None,
                "path": path,
            }
            if store_text:
                out["text"] = text
            out.update(feats)
            return out

        pbar = tqdm(total=len(df), desc=f"Reading+features (Human:{lang})") if tqdm is not None else None
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(work, i) for i in range(len(df))]
            for fut in as_completed(futs):
                rows.append(fut.result())
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()

    out_df = pd.DataFrame(rows)
    out_df["__one__"] = 1
    return out_df


# -----------------------------
# Summaries
# -----------------------------
def summarize_counts(df: pd.DataFrame, group_cols: List[str], out_csv: str):
    g = df.groupby(group_cols, dropna=False)["__one__"].sum().reset_index(name="count")
    g.to_csv(out_csv, index=False)
    return g

def summarize_lengths(df: pd.DataFrame, group_cols: List[str], out_csv: str):
    agg_cols = ["n_words_ws", "gpt2_token_count", "n_chars", "n_sentences", "expected_script_frac", "latin_frac"]

    def q(vals, p):
        return float(np.quantile(vals, p)) if len(vals) else np.nan

    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["count"] = int(sub["__one__"].sum())

        for c in agg_cols:
            vals = sub[c].astype(float).values
            row[f"{c}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{c}_median"] = float(np.median(vals)) if len(vals) else np.nan
            row[f"{c}_std"] = float(np.std(vals)) if len(vals) else np.nan
            row[f"{c}_p05"] = q(vals, 0.05)
            row[f"{c}_p95"] = q(vals, 0.95)

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


# -----------------------------
# Plots (matplotlib)
# -----------------------------
def _savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()

def plot_bar_counts(df: pd.DataFrame, col: str, title: str, out_png: str, top_k: int = 30):
    counts = df[col].fillna("NA").value_counts().head(top_k)
    plt.figure(figsize=(10, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    _savefig(out_png)

def plot_heatmap_counts(df: pd.DataFrame, row: str, col: str, title: str, out_png: str):
    pivot = df.pivot_table(index=row, columns=col, values="__one__", aggfunc="sum", fill_value=0)
    data = pivot.values

    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect="auto")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel(row)
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, str(int(data[i, j])), ha="center", va="center", fontsize=8)

    _savefig(out_png)

def plot_ai_vs_human_by_language(df_all: pd.DataFrame, value_col: str, title: str, out_png: str):
    langs = sorted(df_all["language"].dropna().unique().tolist())
    data = []
    labels = []
    for lang in langs:
        for src in ["human", "ai"]:
            vals = df_all[(df_all["language"] == lang) & (df_all["source"] == src)][value_col].astype(float).values
            data.append(vals)
            labels.append(f"{lang}\n{src}")

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(value_col)
    _savefig(out_png)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_dir", type=str, default="articles_today")
    ap.add_argument("--out_dir", type=str, default="stats_out")
    ap.add_argument("--max_workers", type=int, default=24)
    ap.add_argument("--no_gpt2", action="store_true")
    ap.add_argument("--human_paths_json", type=str, default="")
    ap.add_argument("--human_n_per_lang", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store_text", action="store_true", help="Store full text in CSVs (memory-heavy)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # GPT-2 tokenizer init
    gpt2_tok = None
    if not args.no_gpt2:
        if AutoTokenizer is None:
            print("transformers not installed; skipping GPT-2 token counts (install: pip install transformers)")
        else:
            gpt2_tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    default_human_paths = {
        "hindi":  "data/bbc_hindi_articles_with_categories_cleaned.csv",
        "tamil":  "data/tamilmurasu_dataset.csv",
        "telugu": "data/telugu_news_test.csv",
        "bangla": "data/bangla_newspaper_dataset.csv",
    }

    human_paths = default_human_paths
    if args.human_paths_json:
        hp = args.human_paths_json
        if os.path.exists(hp):
            with open(hp, "r", encoding="utf-8") as f:
                human_paths = json.load(f)
        else:
            human_paths = json.loads(hp)

    # Load AI
    ai_df = load_ai_folder(
        args.ai_dir,
        gpt2_tokenizer=gpt2_tok,
        max_workers=args.max_workers,
        store_text=args.store_text,
    )
    ai_df.to_csv(os.path.join(args.out_dir, "ai_samples.csv"), index=False)

    # Load Human (sampled)
    human_df = load_human_csvs_sampled(
        human_paths,
        gpt2_tokenizer=gpt2_tok,
        n_per_lang=args.human_n_per_lang,
        seed=args.seed,
        max_workers=args.max_workers,
        store_text=args.store_text,
    )
    human_df.to_csv(os.path.join(args.out_dir, "human_samples.csv"), index=False)

    df_all = pd.concat([ai_df, human_df], ignore_index=True)
    df_all.to_csv(os.path.join(args.out_dir, "all_samples.csv"), index=False)

    # ---- Counts summaries (AI)
    summarize_counts(ai_df, ["language"], os.path.join(args.out_dir, "summary_ai_counts_by_language.csv"))
    summarize_counts(ai_df, ["model"], os.path.join(args.out_dir, "summary_ai_counts_by_model.csv"))
    summarize_counts(ai_df, ["prompting_technique"], os.path.join(args.out_dir, "summary_ai_counts_by_technique.csv"))
    summarize_counts(ai_df, ["language", "model"], os.path.join(args.out_dir, "summary_ai_counts_lang_x_model.csv"))
    summarize_counts(ai_df, ["language", "prompting_technique"], os.path.join(args.out_dir, "summary_ai_counts_lang_x_tech.csv"))
    summarize_counts(ai_df, ["model", "prompting_technique"], os.path.join(args.out_dir, "summary_ai_counts_model_x_tech.csv"))
    summarize_counts(ai_df, ["language", "category"], os.path.join(args.out_dir, "summary_ai_counts_lang_x_category.csv"))

    # ---- Length summaries (AI vs Human)
    summarize_lengths(df_all, ["source", "language"], os.path.join(args.out_dir, "summary_lengths_source_x_language.csv"))
    summarize_lengths(ai_df, ["language", "model"], os.path.join(args.out_dir, "summary_lengths_lang_x_model.csv"))
    summarize_lengths(ai_df, ["language", "prompting_technique"], os.path.join(args.out_dir, "summary_lengths_lang_x_tech.csv"))
    summarize_lengths(ai_df, ["model", "prompting_technique"], os.path.join(args.out_dir, "summary_lengths_model_x_tech.csv"))

    # ---- Plots
    plot_bar_counts(ai_df, "language", "AI articles by language", os.path.join(plots_dir, "ai_counts_language.png"))
    plot_bar_counts(ai_df, "model", "AI articles by model", os.path.join(plots_dir, "ai_counts_model.png"), top_k=50)
    plot_bar_counts(ai_df, "prompting_technique", "AI articles by prompting technique", os.path.join(plots_dir, "ai_counts_technique.png"))

    plot_heatmap_counts(ai_df, "language", "model", "AI count heatmap: language × model", os.path.join(plots_dir, "ai_heatmap_lang_x_model.png"))
    plot_heatmap_counts(ai_df, "language", "prompting_technique", "AI count heatmap: language × technique", os.path.join(plots_dir, "ai_heatmap_lang_x_tech.png"))
    plot_heatmap_counts(ai_df, "model", "prompting_technique", "AI count heatmap: model × technique", os.path.join(plots_dir, "ai_heatmap_model_x_tech.png"))

    plot_ai_vs_human_by_language(df_all, "n_words_ws", "Word count by language: AI vs Human", os.path.join(plots_dir, "box_words_ai_vs_human_by_lang.png"))
    plot_ai_vs_human_by_language(df_all, "gpt2_token_count", "GPT-2 token count by language: AI vs Human", os.path.join(plots_dir, "box_gpt2tokens_ai_vs_human_by_lang.png"))
    plot_ai_vs_human_by_language(df_all, "expected_script_frac", "Expected-script fraction by language: AI vs Human", os.path.join(plots_dir, "box_scriptfrac_ai_vs_human_by_lang.png"))

    print(f"[DONE] Wrote outputs to: {args.out_dir}")
    print(f"[HUMAN] Used {args.human_n_per_lang} per language (seed={args.seed}).")


if __name__ == "__main__":
    main()
