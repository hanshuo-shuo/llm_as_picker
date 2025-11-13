#!/usr/bin/env python3
"""
Analyze whether LLM-selected trajectories are "better" than the rest.

It summarizes per-episode metrics (via trajectory_to_text.summarize_episode),
splits into selected vs non-selected using a provided selected_ids.txt, and
computes:
  - Group summary stats (mean, std, median, p25, p75)
  - Effect sizes (Cohen's d, Cliff's delta) for key metrics
  - Optional correlation with LLM scores if a scores JSONL is provided
  - Optional plots (histograms) if matplotlib is available and --plots-dir set

Example:
  python scripts/analyze_selection_quality.py \
    --input-dir /projects/p33100/siosio/dm_data/task_00_humanoid-stand \
    --selected-ids /home/shv7753/llm_judge/selected_ids.txt \
    --mapping-json /home/shv7753/llm_judge/humanoid_stand_mapping.json \
    --offline \
    --scores-jsonl scores_humanoid_stand.jsonl \
    --output-json report.json \
    --output-txt report.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajectory_to_text import summarize_episode
import re


def read_selected_ids(path: Path) -> List[str]:
    """
    Supports multiple formats:
      - Plain IDs per line:        episode_00012
      - ID + reason (tab-separated): episode_00012\tSome reason...
      - Batch headers starting with '#': ignored
      - Lines with text: will extract the first token matching ^episode_\\d+
    """
    ids: List[str] = []
    ep_re = re.compile(r"(episode_\d+)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            # tab-separated "episode_XXXX\treason"
            if "\t" in s:
                first = s.split("\t", 1)[0].strip()
                if first:
                    ids.append(first)
                    continue
            # fallback: extract episode_XXXX anywhere in the line
            m = ep_re.search(s)
            if m:
                ids.append(m.group(1))
                continue
            # else: if the whole line is just an id, accept it
            if s.startswith("episode_"):
                ids.append(s)
    return ids


def safe_series(x: List[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    return arr[np.isfinite(arr)]


def summary_stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = safe_series(a)
    b = safe_series(b)
    if a.size < 2 or b.size < 2:
        return float("nan")
    mean_diff = np.mean(a) - np.mean(b)
    # pooled std
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    n_a = a.size
    n_b = b.size
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return float("nan")
    return float(mean_diff / pooled)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    # Non-parametric effect size: probability of superiority
    a = safe_series(a)
    b = safe_series(b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Efficient approximation by sorting
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    i = j = more = less = 0
    n_a, n_b = a_sorted.size, b_sorted.size
    while i < n_a and j < n_b:
        if a_sorted[i] > b_sorted[j]:
            more += n_a - i
            j += 1
        elif a_sorted[i] < b_sorted[j]:
            less += n_b - j
            i += 1
        else:
            # equal: advance both
            i += 1
            j += 1
    delta = (more - less) / (n_a * n_b)
    return float(delta)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = safe_series(x)
    y = safe_series(y)
    n = min(x.size, y.size)
    if n < 2:
        return float("nan")
    x = x[:n]
    y = y[:n]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def extract_scalar_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    # direct scalars
    for k in [
        "length",
        "total_reward",
        "avg_reward",
        "success_rate",
        "action_magnitude_mean",
        "action_smoothness_mean",
        "state_change_magnitude_mean",
    ]:
        v = metrics.get(k, None)
        if v is not None and np.isfinite(v):
            out[k] = float(v)
    # nested stats
    def take(name: str):
        sub = metrics.get(name, None)
        if isinstance(sub, dict):
            for kk in ["mean", "std", "min", "max", "start", "end"]:
                v = sub.get(kk, None)
                if v is not None and np.isfinite(v):
                    out[f"{name}.{kk}"] = float(v)
            # derived: end/start ratio, drop
            if "start" in sub and "end" in sub and sub["start"] not in [0, None]:
                try:
                    out[f"{name}.end_over_start"] = float(sub["end"] / (sub["start"] if sub["start"] != 0 else np.nan))
                    out[f"{name}.drop"] = float(sub["start"] - sub["end"])
                except Exception:
                    pass
    for name in ["height_stats", "com_speed_stats", "upright_stats"]:
        take(name)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Analyze selection quality by comparing selected episodes vs non-selected.")
    p.add_argument("--input-dir", type=str, required=True, help="Directory containing episode .npz files.")
    p.add_argument("--selected-ids", type=str, required=True, help="Text file with selected episode IDs (one per line).")
    p.add_argument("--mapping-json", type=str, required=True, help="Mapping JSON for offline summarization.")
    p.add_argument("--offline", action="store_true", help="Run offline; require --mapping-json.")
    p.add_argument("--scores-jsonl", type=str, default="", help="Optional per-episode scores JSONL to correlate with metrics.")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on episodes for faster analysis.")
    p.add_argument("--output-json", type=str, default="report.json", help="Where to write JSON report.")
    p.add_argument("--output-txt", type=str, default="report.txt", help="Where to write human-readable summary.")
    p.add_argument("--plots-dir", type=str, default="", help="Optional directory to save histograms (requires matplotlib).")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    selected_ids = set(read_selected_ids(Path(args.selected_ids)))
    mapping_json = Path(args.mapping_json)
    episode_files = sorted([p for p in input_dir.glob("*.npz") if p.is_file()])
    if args.limit is not None:
        episode_files = episode_files[: args.limit]

    # Optional scores
    id_to_scores: Dict[str, Dict[str, float]] = {}
    if args.scores_jsonl:
        try:
            with open(args.scores_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        eid = rec.get("episode_id", "")
                        sc = rec.get("scores", {})
                        if eid and isinstance(sc, dict):
                            id_to_scores[eid] = {
                                "overall": float(sc.get("overall_learning_value", np.nan)),
                                "dynamic": float(sc.get("dynamic_richness_score", np.nan)),
                                "causal": float(sc.get("causal_coherence_score", np.nan)),
                            }
                    except Exception:
                        continue
        except Exception:
            pass

    # Collect metrics
    rows: List[Dict[str, object]] = []
    for ep in episode_files:
        eid = ep.stem
        try:
            summ = summarize_episode(
                ep,
                task_name_hint="humanoid-stand",
                mapping_json=mapping_json,
                offline=bool(args.offline or mapping_json),
            )
            metrics = summ.get("metrics", {})
            flat = extract_scalar_metrics(metrics)
            row = {
                "episode_id": eid,
                "selected": bool(eid in selected_ids),
                "path": str(ep),
                "metrics": flat,
            }
            if eid in id_to_scores:
                row["scores"] = id_to_scores[eid]
            rows.append(row)
        except Exception as e:
            rows.append({"episode_id": eid, "selected": bool(eid in selected_ids), "path": str(ep), "error": str(e)})

    # Split groups
    sel = [r for r in rows if r.get("selected") and "metrics" in r]
    non = [r for r in rows if not r.get("selected") and "metrics" in r]

    # Determine common metric keys
    keys = set()
    for r in sel[:50] + non[:50]:
        keys.update(list(r.get("metrics", {}).keys()))
    keys = sorted(keys)

    report = {
        "counts": {"selected": len(sel), "non_selected": len(non), "total": len(rows)},
        "metrics": {},
        "correlations": {},
    }

    # Optional plotting
    plot_dir: Optional[Path] = Path(args.plots_dir) if args.plots_dir else None
    can_plot = False
    if plot_dir:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            can_plot = True
            plot_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            can_plot = False

    # Compute per-metric stats and effect sizes
    for k in keys:
        a = safe_series([r["metrics"].get(k, np.nan) for r in sel])
        b = safe_series([r["metrics"].get(k, np.nan) for r in non])
        report["metrics"][k] = {
            "selected": summary_stats(a),
            "non_selected": summary_stats(b),
            "effect_sizes": {
                "cohens_d": cohens_d(a, b),
                "cliffs_delta": cliffs_delta(a, b),
            },
        }
        if can_plot and a.size > 0 and b.size > 0:
            try:
                import matplotlib.pyplot as plt  # type: ignore

                plt.figure(figsize=(6, 4))
                # Use common bin edges and probability density for fair comparison
                num_bins = 30
                vmin = float(np.min([np.min(a), np.min(b)]))
                vmax = float(np.max([np.max(a), np.max(b)]))
                if vmin == vmax:
                    vmax = vmin + 1e-6
                bins = np.linspace(vmin, vmax, num_bins + 1)
                plt.hist(b, bins=bins, density=True, alpha=0.6, label="non_selected")
                plt.hist(a, bins=bins, density=True, alpha=0.6, label="selected")
                plt.title(k)
                plt.ylabel("Probability density")
                plt.legend()
                plt.tight_layout()
                plt.savefig(str(plot_dir / f"{k.replace('.', '_')}.png"))
                plt.close()
            except Exception:
                pass

    # Correlate metrics with overall score if available
    if id_to_scores:
        overall_sel = []
        overall_non = []
        for r in sel:
            sc = r.get("scores")
            if sc and np.isfinite(sc.get("overall", np.nan)):
                overall_sel.append((r, sc["overall"]))
        for r in non:
            sc = r.get("scores")
            if sc and np.isfinite(sc.get("overall", np.nan)):
                overall_non.append((r, sc["overall"]))
        def corr_for(group_rows):
            corr = {}
            if not group_rows:
                return corr
            # Use intersection of keys to avoid sparse metrics
            group_keys = set()
            for r, _ in group_rows[:100]:
                group_keys.update(r["metrics"].keys())
            for k in sorted(group_keys):
                xs = safe_series([r["metrics"].get(k, np.nan) for r, _ in group_rows])
                ys = safe_series([s for _, s in group_rows])
                n = min(xs.size, ys.size)
                if n >= 2:
                    corr[k] = pearson_corr(xs[:n], ys[:n])
            return corr
        report["correlations"]["selected_vs_overall"] = corr_for(overall_sel)
        report["correlations"]["non_selected_vs_overall"] = corr_for(overall_non)

    # Write outputs
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out_txt = Path(args.output_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Selected: {report['counts']['selected']}, Non-selected: {report['counts']['non_selected']}, Total: {report['counts']['total']}\n\n")
        f.write("Per-metric summary (selected vs non-selected) and effect sizes:\n")
        def _fmt4(v):
            try:
                return f"{float(v):.4f}"
            except Exception:
                return "NA"
        def _fmt3(v):
            try:
                return f"{float(v):.3f}"
            except Exception:
                return "NA"
        for k in keys:
            m = report["metrics"][k]
            s = m["selected"]
            n = m["non_selected"]
            eff = m["effect_sizes"]
            f.write(
                f"- {k}: sel(mean={_fmt4(s.get('mean'))}±{_fmt4(s.get('std'))}, n={s.get('count',0)}), "
                f"non(mean={_fmt4(n.get('mean'))}±{_fmt4(n.get('std'))}, n={n.get('count',0)}), "
                f"d={_fmt3(eff.get('cohens_d'))}, cliff={_fmt3(eff.get('cliffs_delta'))}\n"
            )
        if id_to_scores:
            f.write("\nPearson correlations with overall score (per group):\n")
            f.write("- Selected group:\n")
            for k, v in sorted(report["correlations"]["selected_vs_overall"].items()):
                f.write(f"  {k}: {_fmt3(v)}\n")
            f.write("- Non-selected group:\n")
            for k, v in sorted(report["correlations"]["non_selected_vs_overall"].items()):
                f.write(f"  {k}: {_fmt3(v)}\n")

    print(f"Wrote analysis to {out_json} and {out_txt}")


if __name__ == "__main__":
    main()


