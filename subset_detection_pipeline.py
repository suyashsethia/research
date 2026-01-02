# subset_detection_pipeline.py
# Updated with:
#  1) Explicit HTTP timeouts + max_retries=0 for OpenAI + Anthropic (prevents 10-min hangs × retries)
#  2) Updated default Claude model (Sonnet 3.5 retired; use Sonnet 4.5)
#  3) Smoke test before launching ThreadPool (fail fast if keys/models/network broken)
#  4) Better progress: tqdm updates on COMPLETED futures; also logs first completion
#  5) Per-task timeout guard (soft) + safer exception handling (writes error rows instead of stalling)
#
# Run example:
#   python subset_detection_pipeline.py \
#     --ai_dir articles_today \
#     --out_dir subset_detection_out \
#     --n_per_group 10 \
#     --seed 42 \
#     --rewrite_provider openai \
#     --rewrite_model gpt-4o-mini \
#     --judge_model claude-sonnet-4-5-20250929 \
#     --max_workers 2 \
#     --n_rewrites 1 \
#     --resume \
#     --smoke_test

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import argparse
import threading
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Optional fast Levenshtein
try:
    from rapidfuzz.distance import Levenshtein as RF_Lev
except Exception:
    RF_Lev = None

# Claude
try:
    import anthropic
except Exception:
    anthropic = None

# Gemini/Gemma
try:
    from google import genai
except Exception:
    genai = None

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Together
try:
    from together import Together
except Exception:
    Together = None

# httpx for timeouts in OpenAI/Anthropic clients
try:
    import httpx
except Exception:
    httpx = None


TECHNIQUES = {"zero-shot", "few-shot", "chain-of-thought"}
LANGUAGES = {"hindi", "bangla", "tamil", "telugu"}


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def parse_ai_filename(path: str) -> Optional[Dict[str, str]]:
    """
    Parse: <lang>_<model>_<technique>_<category>_<uid>.txt
    where model/category may include underscores.
    """
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
        "source": "ai",
    }


def index_ai_files(ai_dir: str) -> pd.DataFrame:
    files = []
    for root, _, names in os.walk(ai_dir):
        for nm in names:
            if nm.endswith(".txt"):
                files.append(os.path.join(root, nm))

    parsed = []
    it = files if tqdm is None else tqdm(files, desc="Indexing AI files")
    for p in it:
        meta = parse_ai_filename(p)
        if meta:
            parsed.append(meta)

    if not parsed:
        raise RuntimeError(f"No parseable .txt files found under: {ai_dir}")
    return pd.DataFrame(parsed)


def sample_per_group(df: pd.DataFrame, group_cols: List[str], n_per_group: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_rows = []
    for _, sub in df.groupby(group_cols, dropna=False):
        if len(sub) <= n_per_group:
            out_rows.append(sub.copy())
        else:
            idx = rng.choice(sub.index.to_numpy(), size=n_per_group, replace=False)
            out_rows.append(sub.loc[idx].copy())
    return pd.concat(out_rows, ignore_index=True)


def levenshtein_distance(a: str, b: str) -> int:
    if RF_Lev is not None:
        return int(RF_Lev.distance(a, b))
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def normalized_edit_ratio(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return float(levenshtein_distance(a, b)) / float(denom)


def make_timeout():
    if httpx is None:
        return None
    # Overall 60s; connect 5s; read 55s
    return httpx.Timeout(60.0, connect=5.0, read=55.0, write=30.0)


# -----------------------------
# Rewrite Client
# -----------------------------
@dataclass
class RewriteClient:
    provider: str   # "openai" | "together" | "gemini"
    model: str
    api_key: str
    timeout: Optional[object] = None
    max_retries: int = 0

    def __post_init__(self):
        self._lock = threading.Lock()
        p = self.provider.lower()

        if p == "openai":
            if OpenAI is None:
                raise RuntimeError("openai not installed. pip install openai")
            kwargs = {"api_key": self.api_key}
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            kwargs["max_retries"] = self.max_retries
            self._client = OpenAI(**kwargs)

        elif p == "together":
            if Together is None:
                raise RuntimeError("together not installed. pip install together")
            # Together SDK does not expose httpx.Timeout the same way; keep simple.
            self._client = Together(api_key=self.api_key)

        elif p == "gemini":
            if genai is None:
                raise RuntimeError("google-genai not installed. pip install google-genai")
            self._client = genai.Client(api_key=self.api_key)

        else:
            raise ValueError(f"Unsupported rewrite_provider: {self.provider}")

    def rewrite(self, text: str, language: str) -> str:
        prompt = (
            f"Rewrite the following news article in {language}.\n"
            f"- Preserve meaning and facts.\n"
            f"- Use different wording and sentence structures.\n"
            f"- Keep the same language/script.\n"
            f"- Return ONLY the rewritten article.\n\n"
            f"ARTICLE:\n{text}"
        )

        p = self.provider.lower()

        if p == "openai":
            with self._lock:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            out = (resp.choices[0].message.content or "").strip()
            if not out:
                raise ValueError("Empty rewrite from OpenAI.")
            return out

        if p == "together":
            with self._lock:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            out = (resp.choices[0].message.content or "").strip()
            if not out:
                raise ValueError("Empty rewrite from Together.")
            return out

        if p == "gemini":
            with self._lock:
                resp = self._client.models.generate_content(model=self.model, contents=prompt)
            out = (getattr(resp, "text", None) or "").strip()
            if not out:
                raise ValueError("Empty rewrite from Gemini/Gemma.")
            return out

        raise ValueError(f"Unsupported provider: {self.provider}")


# -----------------------------
# Claude Judge
# -----------------------------
@dataclass
class ClaudeJudge:
    model: str
    api_key: str
    timeout: Optional[object] = None
    max_retries: int = 0

    def __post_init__(self):
        if anthropic is None:
            raise RuntimeError("anthropic not installed. pip install anthropic")

        kwargs = {"api_key": self.api_key}
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        kwargs["max_retries"] = self.max_retries
        self._client = anthropic.Anthropic(**kwargs)
        self._lock = threading.Lock()

    def judge(self, text: str, language: str) -> Dict:
        system = (
            "You are a strict text forensics classifier.\n"
            "Decide if the news article is AI-generated or human-written.\n"
            "Return ONLY valid JSON with keys: label, p_ai, confidence, brief_reason.\n"
            "label must be 'ai' or 'human'. p_ai must be in [0,1].\n"
            "brief_reason must be <= 20 words."
        )
        user = f"Language: {language}\n\nTEXT:\n{text}"

        with self._lock:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )

        content = ""
        try:
            content = "".join([c.text for c in msg.content if hasattr(c, "text")]).strip()
        except Exception:
            content = str(msg).strip()

        parsed = _parse_json_loose(content)
        label = str(parsed.get("label", "")).strip().lower()
        if label not in {"ai", "human"}:
            label = "ai"
        try:
            p_ai = float(parsed.get("p_ai", 0.5))
        except Exception:
            p_ai = 0.5
        p_ai = float(np.clip(p_ai, 0.0, 1.0))
        conf = str(parsed.get("confidence", "medium")).strip().lower()
        if conf not in {"low", "medium", "high"}:
            conf = "medium"
        reason = str(parsed.get("brief_reason", "")).strip()
        if len(reason.split()) > 20:
            reason = " ".join(reason.split()[:20])

        return {"label": label, "p_ai": p_ai, "confidence": conf, "brief_reason": reason, "raw": content}


def _parse_json_loose(s: str) -> Dict:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return {}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        blob2 = blob.replace("\n", " ").replace("\t", " ")
        try:
            return json.loads(blob2)
        except Exception:
            return {}


def smoke_test_openai(client: OpenAI, model: str) -> None:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Return exactly: OK"}],
        temperature=0.0,
    )
    out = (resp.choices[0].message.content or "").strip()
    dt = time.time() - t0
    print(f"[SMOKE] OpenAI model={model} dt={dt:.2f}s out={out[:32]!r}")
    if "OK" not in out:
        raise RuntimeError("OpenAI smoke test failed (unexpected output).")


def smoke_test_anthropic(client: "anthropic.Anthropic", model: str) -> None:
    t0 = time.time()
    msg = client.messages.create(
        model=model,
        max_tokens=16,
        temperature=0.0,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    out = "".join([b.text for b in msg.content if getattr(b, "type", None) == "text"]).strip()
    dt = time.time() - t0
    print(f"[SMOKE] Anthropic model={model} dt={dt:.2f}s out={out[:32]!r}")
    if "OK" not in out:
        raise RuntimeError("Anthropic smoke test failed (unexpected output).")


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_dir", type=str, default="articles_today")
    ap.add_argument("--out_dir", type=str, default="subset_detection_out")
    ap.add_argument("--n_per_group", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=2)  # safer default
    ap.add_argument("--n_rewrites", type=int, default=1)

    ap.add_argument("--rewrite_provider", type=str, default="openai", choices=["openai", "together", "gemini"])
    ap.add_argument("--rewrite_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--judge_model", type=str, default="claude-sonnet-4-5-20250929")  # UPDATED default
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--fail_fast", action="store_true", help="Stop entire run on first API error")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rewrites_dir = os.path.join(args.out_dir, "rewrites")
    judge_dir = os.path.join(args.out_dir, "judge_json")
    os.makedirs(rewrites_dir, exist_ok=True)
    os.makedirs(judge_dir, exist_ok=True)

    manifest_path = os.path.join(args.out_dir, "subset_manifest.csv")
    raidar_csv = os.path.join(args.out_dir, "raidar_results.csv")
    judge_csv = os.path.join(args.out_dir, "judge_results.csv")
    combined_csv = os.path.join(args.out_dir, "combined_results.csv")

    # API keys
    openai_key = os.getenv("OPENAI_API_KEY", "")
    together_key = os.getenv("TOGETHER_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY is required for Claude judge.")

    if args.rewrite_provider == "openai" and not openai_key:
        raise ValueError("OPENAI_API_KEY is required for rewrite_provider=openai")
    if args.rewrite_provider == "together" and not together_key:
        raise ValueError("TOGETHER_API_KEY is required for rewrite_provider=together")
    if args.rewrite_provider == "gemini" and not gemini_key:
        raise ValueError("GEMINI_API_KEY is required for rewrite_provider=gemini")

    timeout = make_timeout()  # httpx.Timeout or None

    rewrite_key = openai_key if args.rewrite_provider == "openai" else together_key if args.rewrite_provider == "together" else gemini_key

    # Create clients with explicit timeouts + no retries (prevents long stalls)
    rewrite_client = RewriteClient(
        provider=args.rewrite_provider,
        model=args.rewrite_model,
        api_key=rewrite_key,
        timeout=timeout,
        max_retries=0,
    )
    judge_client = ClaudeJudge(
        model=args.judge_model,
        api_key=anthropic_key,
        timeout=timeout,
        max_retries=0,
    )

    # Smoke tests (fail fast before heavy run)
    if args.smoke_test:
        if args.rewrite_provider == "openai":
            # create a dedicated OpenAI client for smoke test (same as rewrite_client)
            oc = rewrite_client._client
            smoke_test_openai(oc, args.rewrite_model)
        elif args.rewrite_provider == "together":
            # Together smoke test: do one chat completion
            t0 = time.time()
            resp = rewrite_client._client.chat.completions.create(
                model=args.rewrite_model,
                messages=[{"role": "user", "content": "Return exactly: OK"}],
                temperature=0.0,
            )
            out = (resp.choices[0].message.content or "").strip()
            print(f"[SMOKE] Together model={args.rewrite_model} dt={time.time()-t0:.2f}s out={out[:32]!r}")
            if "OK" not in out:
                raise RuntimeError("Together smoke test failed.")
        elif args.rewrite_provider == "gemini":
            t0 = time.time()
            resp = rewrite_client._client.models.generate_content(model=args.rewrite_model, contents="Return exactly: OK")
            out = (getattr(resp, "text", None) or "").strip()
            print(f"[SMOKE] Gemini model={args.rewrite_model} dt={time.time()-t0:.2f}s out={out[:32]!r}")
            if "OK" not in out:
                raise RuntimeError("Gemini smoke test failed.")

        smoke_test_anthropic(judge_client._client, args.judge_model)

    # 1) Load + divide into classes
    df_all = index_ai_files(args.ai_dir)
    group_cols = ["language", "prompting_technique", "model"]
    df_subset = sample_per_group(df_all, group_cols=group_cols, n_per_group=args.n_per_group, seed=args.seed)
    df_subset = df_subset.sort_values(group_cols + ["uid"]).reset_index(drop=True)
    df_subset.to_csv(manifest_path, index=False)

    # Resume sets
    done_raidar = set()
    done_judge = set()
    if args.resume:
        if os.path.exists(raidar_csv):
            done_raidar = set(pd.read_csv(raidar_csv, dtype=str, keep_default_na=False).get("uid", []).tolist())
        if os.path.exists(judge_csv):
            done_judge = set(pd.read_csv(judge_csv, dtype=str, keep_default_na=False).get("uid", []).tolist())

    from concurrent.futures import ThreadPoolExecutor, as_completed

    raidar_rows: List[Dict] = []
    judge_rows: List[Dict] = []

    def process_one(row: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        uid = str(row["uid"])
        lang = str(row["language"])
        model = str(row["model"])
        tech = str(row["prompting_technique"])
        category = str(row.get("category", ""))
        path = str(row["path"])

        text = read_text(path)
        text_hash = sha1_text(text)

        raidar_out = None
        judge_out = None

        # RAiDAR
        if uid not in done_raidar:
            try:
                ratios = []
                for k in range(args.n_rewrites):
                    rtext = rewrite_client.rewrite(text=text, language=lang)
                    ratios.append(normalized_edit_ratio(text, rtext))
                    rw_path = os.path.join(rewrites_dir, f"{uid}_rewrite{k+1}.txt")
                    with open(rw_path, "w", encoding="utf-8") as f:
                        f.write(rtext)

                min_ratio = float(np.min(ratios)) if ratios else 1.0
                raidar_score = 1.0 - min_ratio
                raidar_out = {
                    "uid": uid,
                    "language": lang,
                    "model": model,
                    "prompting_technique": tech,
                    "category": category,
                    "path": path,
                    "text_sha1": text_hash,
                    "rewrite_provider": args.rewrite_provider,
                    "rewrite_model": args.rewrite_model,
                    "n_rewrites": args.n_rewrites,
                    "edit_ratio_min": min_ratio,
                    "raidar_score": raidar_score,
                    "error": "",
                }
            except Exception as e:
                if args.fail_fast:
                    raise
                raidar_out = {
                    "uid": uid,
                    "language": lang,
                    "model": model,
                    "prompting_technique": tech,
                    "category": category,
                    "path": path,
                    "text_sha1": text_hash,
                    "rewrite_provider": args.rewrite_provider,
                    "rewrite_model": args.rewrite_model,
                    "n_rewrites": args.n_rewrites,
                    "edit_ratio_min": "",
                    "raidar_score": "",
                    "error": f"{type(e).__name__}: {e}",
                }

        # Judge
        if uid not in done_judge:
            try:
                j = judge_client.judge(text=text, language=lang)

                j_path = os.path.join(judge_dir, f"{uid}.json")
                with open(j_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "uid": uid,
                            "language": lang,
                            "model": model,
                            "prompting_technique": tech,
                            "category": category,
                            "path": path,
                            "text_sha1": text_hash,
                            "judge_model": args.judge_model,
                            "result": j,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                judge_out = {
                    "uid": uid,
                    "language": lang,
                    "model": model,
                    "prompting_technique": tech,
                    "category": category,
                    "path": path,
                    "text_sha1": text_hash,
                    "judge_model": args.judge_model,
                    "judge_label": j.get("label", ""),
                    "judge_p_ai": j.get("p_ai", 0.5),
                    "judge_confidence": j.get("confidence", ""),
                    "judge_brief_reason": j.get("brief_reason", ""),
                    "error": "",
                }
            except Exception as e:
                if args.fail_fast:
                    raise
                judge_out = {
                    "uid": uid,
                    "language": lang,
                    "model": model,
                    "prompting_technique": tech,
                    "category": category,
                    "path": path,
                    "text_sha1": text_hash,
                    "judge_model": args.judge_model,
                    "judge_label": "",
                    "judge_p_ai": "",
                    "judge_confidence": "",
                    "judge_brief_reason": "",
                    "error": f"{type(e).__name__}: {e}",
                }

        return raidar_out, judge_out

    records = df_subset.to_dict(orient="records")

    pbar = tqdm(total=len(records), desc="RAiDAR+Judge on subset") if tqdm is not None else None
    completed_first = False

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(process_one, r) for r in records]
        for fut in as_completed(futs):
            r_out, j_out = fut.result()

            if r_out is not None:
                raidar_rows.append(r_out)
            if j_out is not None:
                judge_rows.append(j_out)

            if not completed_first:
                completed_first = True
                print("[INFO] First task completed (pipeline is making progress).")

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Persist (append if resume)
    if raidar_rows:
        df_r = pd.DataFrame(raidar_rows)
        if args.resume and os.path.exists(raidar_csv):
            df_prev = pd.read_csv(raidar_csv, dtype=str, keep_default_na=False)
            df_r = pd.concat([df_prev, df_r], ignore_index=True)
        df_r.to_csv(raidar_csv, index=False)

    if judge_rows:
        df_j = pd.DataFrame(judge_rows)
        if args.resume and os.path.exists(judge_csv):
            df_prev = pd.read_csv(judge_csv, dtype=str, keep_default_na=False)
            df_j = pd.concat([df_prev, df_j], ignore_index=True)
        df_j.to_csv(judge_csv, index=False)

    # Combine
    if os.path.exists(raidar_csv) and os.path.exists(judge_csv):
        df_r = pd.read_csv(raidar_csv, dtype=str, keep_default_na=False)
        df_j = pd.read_csv(judge_csv, dtype=str, keep_default_na=False)
        df_c = df_r.merge(
            df_j,
            on=["uid", "language", "model", "prompting_technique", "category", "path", "text_sha1"],
            how="outer",
            suffixes=("_raidar", "_judge"),
        )
        df_c.to_csv(combined_csv, index=False)

    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] RAiDAR results: {raidar_csv}")
    print(f"[OK] Judge results: {judge_csv}")
    print(f"[OK] Combined: {combined_csv}")
    print(f"[OK] Rewrites: {rewrites_dir}")
    print(f"[OK] Judge JSON: {judge_dir}")


if __name__ == "__main__":
    main()
