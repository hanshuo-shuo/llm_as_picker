#!/usr/bin/env python3
"""
Batch-select best trajectories using an LLM: read episodes in groups (default 10),
present compact summaries + metrics to the model, and ask it to select K winners
per batch with brief reasons.

Example:
  python select_best.py \
    --input-dir /projects/p33100/siosio/dm_data/task_00_humanoid-stand \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --mapping-json /home/shv7753/llm_judge/humanoid_stand_mapping.json \
    --offline \
    --batch-size 10 \
    --select-per-batch 2 \
    --output-jsonl batch_selections.jsonl \
    --selected-txt selected_ids.txt \
    --prompts-jsonl batch_prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import pipeline

from trajectory_to_text import summarize_episode


def extract_json_object(text: str) -> Optional[dict]:
    """
    Extract the first valid JSON object that contains the key 'selected'
    using a simple balanced-brace scan to avoid truncation on nested objects.
    Falls back to generic extraction if nothing is found.
    """
    # Prefer objects that contain "selected"
    s = text.find("{")
    while s != -1:
        depth = 0
        for i in range(s, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[s : i + 1]
                    if '"selected"' in candidate:
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and "selected" in obj:
                                return obj
                        except Exception:
                            pass
                    break
        s = text.find("{", s + 1)

    # Fallback: try to parse first JSON object (may not contain 'selected')
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _compact_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    keep: Dict[str, object] = {}
    for k in [
        "length",
        "total_reward",
        "avg_reward",
        "success_rate",
        "action_magnitude_mean",
        "action_smoothness_mean",
        "state_change_magnitude_mean",
    ]:
        if k in metrics and metrics[k] is not None:
            keep[k] = metrics[k]
    for name in ["height_stats", "com_speed_stats", "upright_stats"]:
        if isinstance(metrics.get(name), dict):
            sub = metrics[name]
            keep[name] = {kk: sub.get(kk) for kk in ["mean", "std", "min", "max", "start", "end"] if kk in sub}
    return keep


def _extract_notes(summary_text: str) -> str:
    for line in summary_text.splitlines():
        if line.startswith("Notes:"):
            return line.replace("Notes:", "").strip()
    return ""


def build_batch_messages(
    candidates: List[Dict[str, object]],
    select_k: int,
) -> List[Dict[str, str]]:
    """
    candidates: list of dicts with keys episode_id, summary_text, metrics
    """
    system_prompt = (
        "You are a top-tier robotics and physics researcher acting as a data evaluation expert. "
        "Given multiple 50-step trajectories, select the most valuable ones for training a world model."
    )
    # Keep the prompt compact to fit context
    lines: List[str] = []
    lines.append(f"Select exactly {select_k} trajectories from the list below.")
    lines.append("For each selected item, return its episode_id and a one-sentence reason.")
    lines.append("Criteria:")
    lines.append("- Dynamic Richness: larger state changes and variability imply higher richness.")
    lines.append("- Causal Coherence: actions plausibly causing the observed changes.")
    lines.append("- Learning Value: usefulness to learn plausible physics and action consequences.")
    lines.append("")
    lines.append("Candidates:")
    # Label candidates 1..N
    for idx, c in enumerate(candidates, 1):
        ep = c["episode_id"]
        metrics = _compact_metrics(c.get("metrics", {}))
        notes = _extract_notes(str(c.get("summary_text", "")))
        # One-line summary header + compact metrics JSON
        lines.append(f"- [{idx}] episode_id={ep}")
        lines.append(f"  notes: {notes if notes else 'N/A'}")
        lines.append(f"  metrics: {json.dumps(metrics, ensure_ascii=False)}")

    user_prompt = "\n".join(lines) + """

Output strictly as JSON:
{
  "selected": [
    {"episode_id": "<id>", "reason": "<one concise sentence>"},
    {"episode_id": "<id>", "reason": "<one concise sentence>"}
  ]
}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def batch_iter(paths: List[Path], batch_size: int) -> Iterable[List[Path]]:
    for i in range(0, len(paths), batch_size):
        yield paths[i : i + batch_size]


def parse_args():
    p = argparse.ArgumentParser(description="Select K best trajectories out of N per batch using an LLM.")
    p.add_argument("--input-dir", type=str, required=True, help="Directory with .npz episodes.")
    p.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--mapping-json", type=str, default="", help="Mapping JSON for offline summarization.")
    p.add_argument("--offline", action="store_true", help="Do not access dm_control; require --mapping-json.")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--select-per-batch", type=int, default=2)
    p.add_argument("--output-jsonl", type=str, default="batch_selections.jsonl")
    p.add_argument("--selected-txt", type=str, default="selected_ids.txt")
    p.add_argument("--prompts-jsonl", type=str, default="")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--task-hint", type=str, default="humanoid-stand")
    p.add_argument("--do-sample", action="store_true", help="Enable sampling; deterministic by default.")
    p.add_argument("--debug", action="store_true", help="Print full prompts and outputs for the first batches.")
    p.add_argument("--debug-batches", type=int, default=1, help="Number of initial batches to print when --debug is on.")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    pipe = pipeline(
        "text-generation",
        model=args.model,
        model_kwargs={"torch_dtype": dtype_map.get(args.dtype, torch.bfloat16)},
        device_map=args.device_map,
    )
    terminators = [pipe.tokenizer.eos_token_id, pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    # Files
    episode_files = sorted([p for p in input_dir.glob("*.npz") if p.is_file()])
    if args.limit is not None:
        episode_files = episode_files[: args.limit]

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_txt = Path(args.selected_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    prompts_file = Path(args.prompts_jsonl) if args.prompts_jsonl else None
    f_prompts = None
    if prompts_file:
        prompts_file.parent.mkdir(parents=True, exist_ok=True)
        f_prompts = open(prompts_file, "w", encoding="utf-8")

    selections: List[dict] = []
    with open(out_jsonl, "w", encoding="utf-8") as f_out, open(out_txt, "w", encoding="utf-8") as f_txt:
        batch_idx = 0
        for batch_paths in batch_iter(episode_files, args.batch_size):
            batch_idx += 1
            # Summarize candidates
            candidates: List[Dict[str, object]] = []
            for ep_path in batch_paths:
                try:
                    summary = summarize_episode(
                        ep_path,
                        task_name_hint=(args.task_hint or None),
                        mapping_json=(Path(args.mapping_json) if args.mapping_json else None),
                        offline=bool(args.offline or args.mapping_json),
                    )
                    candidates.append(
                        {
                            "episode_id": Path(ep_path).stem,
                            "summary_text": summary["summary_text"],
                            "metrics": summary.get("metrics", {}),
                            "path": str(ep_path),
                        }
                    )
                except Exception as e:
                    # Skip failed candidate in this batch but record error
                    candidates.append(
                        {
                            "episode_id": Path(ep_path).stem,
                            "summary_text": "",
                            "metrics": {},
                            "path": str(ep_path),
                            "error": str(e),
                        }
                    )
            # Build prompt
            msgs = build_batch_messages(candidates, select_k=int(args.select_per_batch))
            if args.debug and batch_idx <= int(args.debug_batches):
                try:
                    print(f"\n=== DEBUG Batch {batch_idx} INPUT - System ===")
                    print(msgs[0]["content"])
                    print(f"\n=== DEBUG Batch {batch_idx} INPUT - User ===")
                    print(msgs[1]["content"])
                except Exception:
                    pass
            if f_prompts is not None:
                f_prompts.write(json.dumps({"batch_index": batch_idx, "messages": msgs}, ensure_ascii=False) + "\n")
                f_prompts.flush()
            # Query model
            outputs = pipe(
                msgs,
                max_new_tokens=int(args.max_new_tokens),
                eos_token_id=terminators,
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            content = None
            try:
                content = outputs[0]["generated_text"][-1]["content"]
            except Exception:
                content = str(outputs[0].get("generated_text", ""))
            parsed = extract_json_object(content or "")
            if args.debug and batch_idx <= int(args.debug_batches):
                try:
                    print(f"\n=== DEBUG Batch {batch_idx} OUTPUT - Raw ===")
                    print(content)
                    print(f"\n=== DEBUG Batch {batch_idx} OUTPUT - Parsed ===")
                    print(json.dumps(parsed if parsed is not None else {"error": "parse_failed"}, ensure_ascii=False, indent=2))
                except Exception:
                    pass
            if not parsed or "selected" not in parsed or not isinstance(parsed["selected"], list):
                # Record error for this batch, continue
                rec = {
                    "batch_index": batch_idx,
                    "error": "JSON parsing failed or missing 'selected'",
                    "raw_model_text": content,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                continue
            # Normalize selections
            selected = []
            for sel in parsed["selected"][: int(args.select_per_batch)]:
                try:
                    selected.append(
                        {
                            "episode_id": str(sel.get("episode_id", "")),
                            "reason": str(sel.get("reason", "")).strip(),
                        }
                    )
                except Exception:
                    continue
            rec = {
                "batch_index": batch_idx,
                "selected": selected,
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            selections.append(rec)
            # Write to TXT
            f_txt.write(f"# batch {batch_idx}\n")
            for s in selected:
                f_txt.write(f"{s['episode_id']}\t{s['reason']}\n")
            f_txt.write("\n")

    if f_prompts is not None:
        try:
            f_prompts.close()
        except Exception:
            pass
    print(f"Wrote batch selections to {out_jsonl} and IDs to {out_txt}")


if __name__ == "__main__":
    main()


