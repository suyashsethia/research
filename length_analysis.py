# length_analysis.py
# Reads stats_out/ai_samples.csv (and optionally human_samples.csv / all_samples.csv)
# Produces "avg tokens per article" summaries + plots by language/model/technique.

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def summarize_length(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    """
    Per-group summary of per-article length (tokens/article or words/article).
    """
    def q(x, p): return float(np.quantile(x, p)) if len(x) else np.nan

    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        vals = sub[value_col].astype(float).values
        rows.append({
            **dict(zip(group_cols, key)),
            "count": int(len(vals)),
            "mean": float(np.mean(vals)) if len(vals) else np.nan,
            "median": float(np.median(vals)) if len(vals) else np.nan,
            "std": float(np.std(vals)) if len(vals) else np.nan,
            "p05": q(vals, 0.05),
            "p25": q(vals, 0.25),
            "p75": q(vals, 0.75),
            "p95": q(vals, 0.95),
        })
    return pd.DataFrame(rows)


def plot_bar_with_error(df_sum: pd.DataFrame, cat_col: str, mean_col: str, err_col: str, title: str, out_png: str, top_k: int = 30):
    """
    Bar chart of mean length per category with std error bars.
    """
    d = df_sum.sort_values("count", ascending=False).head(top_k).copy()
    x = d[cat_col].astype(str).values
    y = d[mean_col].astype(float).values
    e = d[err_col].astype(float).values

    plt.figure(figsize=(11, 4))
    plt.bar(x, y, yerr=e, capsize=3)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(mean_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_heatmap(df: pd.DataFrame, rows: str, cols: str, values: str, title: str, out_png: str):
    """
    Heatmap for mean tokens/article (or words/article) for a 2D grouping.
    """
    pivot = df.pivot_table(index=rows, columns=cols, values=values, aggfunc="mean", fill_value=np.nan)

    data = pivot.values
    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect="auto")
    plt.title(title)
    plt.xlabel(cols)
    plt.ylabel(rows)
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))

    # annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def plot_box_by_group(df: pd.DataFrame, group_col: str, value_col: str, title: str, out_png: str, top_k: int = 20):
    """
    Boxplot distributions of per-article length by group.
    """
    top = df[group_col].fillna("NA").value_counts().head(top_k).index.tolist()
    sub = df[df[group_col].fillna("NA").isin(top)].copy()

    data = []
    labels = []
    for g in top:
        vals = sub[sub[group_col].fillna("NA") == g][value_col].astype(float).values
        data.append(vals)
        labels.append(str(g))

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_dir", type=str, default="stats_out")
    ap.add_argument("--ai_csv", type=str, default="ai_samples.csv")
    ap.add_argument("--human_csv", type=str, default="human_samples.csv")
    ap.add_argument("--include_human", action="store_true")
    ap.add_argument("--value_col", type=str, default="gpt2_token_count", help="Use gpt2_token_count or n_words_ws")
    args = ap.parse_args()

    out_dir = os.path.join(args.stats_dir, "length_analysis")
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    ai_path = os.path.join(args.stats_dir, args.ai_csv)
    if not os.path.exists(ai_path):
        raise FileNotFoundError(f"Missing: {ai_path}")

    # Load minimal columns (fast)
    usecols_ai = ["language", "model", "prompting_technique", "category", args.value_col]
    ai = pd.read_csv(ai_path, usecols=[c for c in usecols_ai if c in pd.read_csv(ai_path, nrows=0).columns])

    # Derived per-article density stats (optional but useful)
    # tokens/sentence and tokens/word (length-normalized)
    if "n_sentences" in pd.read_csv(ai_path, nrows=0).columns and args.value_col in ai.columns:
        fullcols = ["n_sentences", "n_words_ws"]
        ai2 = pd.read_csv(ai_path, usecols=list(set(usecols_ai + fullcols)))
        ai = ai.merge(ai2[["language","model","prompting_technique","category","n_sentences","n_words_ws"]], 
                      on=["language","model","prompting_technique","category"], how="left")
        ai["tokens_per_sentence"] = ai[args.value_col] / ai["n_sentences"].replace(0, np.nan)
        ai["tokens_per_word_ws"] = ai[args.value_col] / ai["n_words_ws"].replace(0, np.nan)

    # ---- Core summaries: average tokens/article by groups
    # 1) language
    s_lang = summarize_length(ai, ["language"], args.value_col)
    s_lang.to_csv(os.path.join(out_dir, "avg_tokens_by_language.csv"), index=False)

    # 2) model
    s_model = summarize_length(ai, ["model"], args.value_col)
    s_model.to_csv(os.path.join(out_dir, "avg_tokens_by_model.csv"), index=False)

    # 3) technique
    s_tech = summarize_length(ai, ["prompting_technique"], args.value_col)
    s_tech.to_csv(os.path.join(out_dir, "avg_tokens_by_technique.csv"), index=False)

    # 4) language x model
    s_lm = summarize_length(ai, ["language", "model"], args.value_col)
    s_lm.to_csv(os.path.join(out_dir, "avg_tokens_language_x_model.csv"), index=False)

    # 5) language x technique
    s_lt = summarize_length(ai, ["language", "prompting_technique"], args.value_col)
    s_lt.to_csv(os.path.join(out_dir, "avg_tokens_language_x_technique.csv"), index=False)

    # 6) model x technique
    s_mt = summarize_length(ai, ["model", "prompting_technique"], args.value_col)
    s_mt.to_csv(os.path.join(out_dir, "avg_tokens_model_x_technique.csv"), index=False)

    # ---- Visualizations
    # Bar charts with std error bars (std / sqrt(n))
    for df_sum, col, name in [
        (s_lang, "language", "language"),
        (s_model, "model", "model"),
        (s_tech, "prompting_technique", "technique"),
    ]:
        df_sum = df_sum.copy()
        df_sum["stderr"] = df_sum["std"] / np.sqrt(df_sum["count"].replace(0, np.nan))
        plot_bar_with_error(
            df_sum, cat_col=col, mean_col="mean", err_col="stderr",
            title=f"Average {args.value_col} per article by {name} (± standard error)",
            out_png=os.path.join(plots_dir, f"bar_avg_{args.value_col}_by_{name}.png"),
            top_k=50
        )

    # Heatmaps (mean tokens/article)
    plot_heatmap(ai, "language", "model", args.value_col,
                 title=f"Mean {args.value_col} per article: language × model",
                 out_png=os.path.join(plots_dir, f"heatmap_mean_{args.value_col}_lang_x_model.png"))

    plot_heatmap(ai, "language", "prompting_technique", args.value_col,
                 title=f"Mean {args.value_col} per article: language × technique",
                 out_png=os.path.join(plots_dir, f"heatmap_mean_{args.value_col}_lang_x_technique.png"))

    plot_heatmap(ai, "model", "prompting_technique", args.value_col,
                 title=f"Mean {args.value_col} per article: model × technique",
                 out_png=os.path.join(plots_dir, f"heatmap_mean_{args.value_col}_model_x_technique.png"))

    # Boxplots (distribution of tokens/article)
    plot_box_by_group(ai, "prompting_technique", args.value_col,
                      title=f"Distribution of {args.value_col} per article by technique",
                      out_png=os.path.join(plots_dir, f"box_{args.value_col}_by_technique.png"),
                      top_k=20)

    plot_box_by_group(ai, "model", args.value_col,
                      title=f"Distribution of {args.value_col} per article by model (top by count)",
                      out_png=os.path.join(plots_dir, f"box_{args.value_col}_by_model.png"),
                      top_k=20)

    # Optional: AI vs Human comparison on tokens/article by language
    if args.include_human:
        human_path = os.path.join(args.stats_dir, args.human_csv)
        if not os.path.exists(human_path):
            raise FileNotFoundError(f"Missing: {human_path}")

        usecols_h = ["language", args.value_col]
        human = pd.read_csv(human_path, usecols=[c for c in usecols_h if c in pd.read_csv(human_path, nrows=0).columns])
        human["source"] = "human"
        ai2 = ai[["language", args.value_col]].copy()
        ai2["source"] = "ai"
        both = pd.concat([ai2, human], ignore_index=True)

        s_src_lang = summarize_length(both, ["source", "language"], args.value_col)
        s_src_lang.to_csv(os.path.join(out_dir, "avg_tokens_source_x_language.csv"), index=False)

        # simple grouped boxplot: (lang, source) pairs
        langs = sorted(both["language"].dropna().unique().tolist())
        data = []
        labels = []
        for lang in langs:
            for src in ["human", "ai"]:
                vals = both[(both["language"] == lang) & (both["source"] == src)][args.value_col].astype(float).values
                data.append(vals)
                labels.append(f"{lang}\n{src}")

        plt.figure(figsize=(12, 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(args.value_col)
        plt.title(f"{args.value_col} per article: AI vs Human (by language)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"box_{args.value_col}_ai_vs_human_by_language.png"), dpi=250)
        plt.close()

    print(f"[DONE] Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
