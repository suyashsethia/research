# plot_dummy_detection_results.py
# Reads stats_out/dummy_detection_scores.csv and generates:
#  - tables: accuracy/F1/AUC by language/model/technique
#  - plots: boxplots + heatmaps + bar charts

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Simple ROC AUC without sklearn.
    y_true in {0,1}. Returns NaN if only one class present.
    """
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    tp = tp / (tp[-1] if tp[-1] else 1)
    fp = fp / (fp[-1] if fp[-1] else 1)
    return float(np.trapz(tp, fp))

def best_threshold_youden(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Choose threshold maximizing TPR - FPR (Youden's J).
    """
    y_true = y_true.astype(int)
    uniq = np.unique(y_score)
    if len(uniq) == 0:
        return 0.5
    # evaluate on sorted unique scores
    best_t, best_j = 0.5, -1e9
    for t in uniq:
        y_pred = (y_score >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t

def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    y_true = y_true.astype(int)
    y_pred = (y_score >= thr).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    bacc = 0.5 * ((tp / (tp + fn) if (tp + fn) else 0.0) + (tn / (tn + fp) if (tn + fp) else 0.0))

    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1), "bacc": float(bacc),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

def summarize_by_group(df: pd.DataFrame, group_cols: list[str], score_col: str, thr: float) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        y_true = (sub["source"].astype(str).str.lower() == "ai").astype(int).to_numpy()
        y_score = sub[score_col].astype(float).to_numpy()

        row = dict(zip(group_cols, key))
        row["count"] = int(len(sub))
        row["auc"] = auc_roc(y_true, y_score)
        row.update(metrics_at_threshold(y_true, y_score, thr))
        row["mean_score"] = float(np.mean(y_score)) if len(y_score) else np.nan
        row["median_score"] = float(np.median(y_score)) if len(y_score) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def plot_box_by_language(df: pd.DataFrame, score_col: str, title: str, out_png: str):
    langs = sorted(df["language"].dropna().unique().tolist())
    data = []
    labels = []
    for lang in langs:
        for src in ["human", "ai"]:
            vals = df[(df["language"] == lang) & (df["source"] == src)][score_col].astype(float).to_numpy()
            data.append(vals)
            labels.append(f"{lang}\n{src}")
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(score_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

def plot_heatmap_mean(df: pd.DataFrame, row: str, col: str, val: str, title: str, out_png: str):
    pivot = df.pivot_table(index=row, columns=col, values=val, aggfunc="mean", fill_value=np.nan)
    data = pivot.values
    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect="auto")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel(row)
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

def plot_bar_auc(df_sum: pd.DataFrame, cat_col: str, title: str, out_png: str, top_k: int = 30):
    d = df_sum.sort_values("count", ascending=False).head(top_k).copy()
    x = d[cat_col].astype(str).to_numpy()
    y = d["auc"].astype(float).to_numpy()
    plt.figure(figsize=(11, 4))
    plt.bar(x, y)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUC")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_dir", type=str, default="stats_out")
    ap.add_argument("--in_csv", type=str, default="dummy_detection_scores.csv")
    ap.add_argument("--out_dir", type=str, default="dummy_detection_report")
    ap.add_argument("--thr_mode", type=str, default="youden", choices=["youden", "fixed"])
    ap.add_argument("--fixed_thr", type=float, default=0.5)
    args = ap.parse_args()

    in_path = os.path.join(args.stats_dir, args.in_csv)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing {in_path}")

    out_dir = os.path.join(args.stats_dir, args.out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    df = pd.read_csv(in_path)

    # need both classes for metrics; if you didn't include human, metrics won't be meaningful
    if df["source"].astype(str).str.lower().nunique() < 2:
        raise ValueError("Need both AI and Human rows for detection metrics. Re-run generator with --use_human.")

    # Prepare labels
    y_true = (df["source"].astype(str).str.lower() == "ai").astype(int).to_numpy()

    score_cols = ["raidar_score_dummy", "judge_p_ai_dummy", "combined_p_ai_dummy"]
    for sc in score_cols:
        if sc not in df.columns:
            raise ValueError(f"Missing score column: {sc}")

    # Choose thresholds (global) per score type
    thresholds = {}
    for sc in score_cols:
        y_score = df[sc].astype(float).to_numpy()
        if args.thr_mode == "youden":
            thresholds[sc] = best_threshold_youden(y_true, y_score)
        else:
            thresholds[sc] = float(args.fixed_thr)

    # Overall table
    overall_rows = []
    for sc in score_cols:
        y_score = df[sc].astype(float).to_numpy()
        row = {"score": sc, "thr": thresholds[sc], "auc": auc_roc(y_true, y_score)}
        row.update(metrics_at_threshold(y_true, y_score, thresholds[sc]))
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(os.path.join(out_dir, "overall_metrics.csv"), index=False)

    # By language
    for sc in score_cols:
        s_lang = summarize_by_group(df, ["language"], sc, thresholds[sc])
        s_lang.to_csv(os.path.join(out_dir, f"metrics_by_language__{sc}.csv"), index=False)

        # Bar AUC by language
        plot_bar_auc(s_lang, "language", f"AUC by language ({sc})", os.path.join(plots_dir, f"bar_auc_by_language__{sc}.png"))

        # Box: AI vs Human per language
        plot_box_by_language(df, sc, f"Score distribution by language (AI vs Human) ({sc})",
                             os.path.join(plots_dir, f"box_by_language_ai_vs_human__{sc}.png"))

    # AI-only breakdowns (model/technique)
    ai = df[df["source"].astype(str).str.lower().eq("ai")].copy()
    for sc in score_cols:
        if "model" in ai.columns:
            hm1 = ai.pivot_table(index="language", columns="model", values=sc, aggfunc="mean", fill_value=np.nan)
            plot_heatmap_mean(ai, "language", "model", sc, f"Mean {sc} (AI only): language × model",
                              os.path.join(plots_dir, f"heatmap_mean_lang_x_model__{sc}.png"))

        if "prompting_technique" in ai.columns:
            plot_heatmap_mean(ai, "language", "prompting_technique", sc, f"Mean {sc} (AI only): language × technique",
                              os.path.join(plots_dir, f"heatmap_mean_lang_x_technique__{sc}.png"))
            plot_heatmap_mean(ai, "model", "prompting_technique", sc, f"Mean {sc} (AI only): model × technique",
                              os.path.join(plots_dir, f"heatmap_mean_model_x_technique__{sc}.png"))

    print(f"[OK] Wrote tables+plots to: {out_dir}")
    print("[NOTE] These are DUMMY results for paper drafting only.")

if __name__ == "__main__":
    main()
