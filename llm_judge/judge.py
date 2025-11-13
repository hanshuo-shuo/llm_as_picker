#!/usr/bin/env python3
"""
Score DM Control trajectories with an LLM using an English prompt and select top-K episode IDs.

Example:
  python select_best.py \
    --input-dir /projects/p33100/siosio/dm_data/task_00_humanoid-stand \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output-jsonl /home/shv7753/llm_judge/humanoid_stand_mapping.json \
    --topk 100 \
    --topk-out top100_ids.txt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import pipeline

# We reuse the summarization utilities
from trajectory_to_text import summarize_episode


def extract_json_object(text: str) -> Optional[dict]:
    """
    Robustly extract the first top-level JSON object from a string and parse it.
    """
    # Fast-path: try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Regex for the first {...} block (non-greedy across newlines)
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _compact_metrics_for_prompt(metrics: Dict[str, object]) -> Dict[str, object]:
    """
    Keep only scalar stats to reduce verbosity in the prompt; drop raw series.
    """
    keep: Dict[str, object] = {}
    scalar_keys = [
        "action_magnitude_mean",
        "action_smoothness_mean",
        "state_change_magnitude_mean",
        "length",
        "total_reward",
        "avg_reward",
        "success_rate",
    ]
    for k in scalar_keys:
        if k in metrics and metrics[k] is not None:
            keep[k] = metrics[k]
    for name in ["height_stats", "com_speed_stats", "upright_stats"]:
        if isinstance(metrics.get(name), dict):
            sub = metrics[name]
            keep[name] = {kk: sub.get(kk) for kk in ["mean", "std", "min", "max", "start", "end"] if kk in sub}
    return keep


def build_messages(summary_text: str, metrics: Dict[str, object]) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a top-tier robotics and physics researcher acting as a data evaluation expert. "
        "Your task is to evaluate a 50-step trajectory from a physics simulation for its value "
        "to training a 'world model' (a dynamics model). Assess strictly based on the provided summary."
    )
    compact = _compact_metrics_for_prompt(metrics or {})
    user_prompt = f"""
Evaluate the following trajectory summary and output ONLY a JSON object with these fields:
{{
  "dynamic_richness_score": <integer 1-10>,
  "causal_coherence_score": <integer 1-10>,
  "overall_learning_value": <integer 1-10>,
  "justification": "<one concise sentence>"
}}

Scoring criteria:
- Dynamic Richness: diversity and magnitude of state changes (e.g., upright to fall vs. remaining static).
- Causal Coherence: consistency between actions and resulting state changes (e.g., exertion vs. movement).
- Learning Value: usefulness for learning plausible physics and action consequences.

Use this rubric to ground your scores with the numeric metrics:
- If state_change_magnitude_mean > 75 or height_stats.std > 0.35 or com_speed_stats.std > 1.0, then Dynamic Richness >= 7.
- If action_magnitude_mean is high AND state_change_magnitude_mean is high, then Causal Coherence >= 6; if low action with high state change, reduce Causal Coherence.
- If there is a large height drop (height_stats.end << height_stats.start) and uprightness is low on average, increase Dynamic Richness but do not inflate Causal Coherence.
- Higher variability (std) in height and COM speed generally increases Dynamic Richness and Learning Value.

Trajectory summary:
---
{summary_text}
---

Key metrics (JSON):
{json.dumps(compact, ensure_ascii=False)}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()},
    ]


def score_directory(
    input_dir: Path,
    model_name: str,
    output_jsonl: Path,
    scores_txt: Path,
    prompts_jsonl: Optional[Path],
    topk: int,
    topk_out: Path,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    limit: Optional[int] = None,
    task_name_hint: Optional[str] = "humanoid-stand",
    mapping_json: Optional[Path] = None,
    offline: bool = True,
    do_sample: bool = False,
) -> Tuple[List[dict], List[str]]:
    print(f"--- Loading model '{model_name}' to GPU(s) from local cache if available ---")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": dtype_map.get(dtype, torch.bfloat16)},
        device_map=device_map,
    )
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    ep_files = sorted([p for p in input_dir.glob("*.npz") if p.is_file()])
    if limit is not None:
        ep_files = ep_files[:limit]
    print(f"Found {len(ep_files)} episode files under {input_dir}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    scored: List[dict] = []

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        f_prompts = None
        try:
            if prompts_jsonl:
                prompts_jsonl.parent.mkdir(parents=True, exist_ok=True)
                f_prompts = open(prompts_jsonl, "w", encoding="utf-8")
            for idx, ep in enumerate(ep_files, 1):
                try:
                    # Summarize trajectory
                    summary = summarize_episode(
                        ep,
                        task_name_hint=task_name_hint,
                        mapping_json=mapping_json,
                        offline=offline,
                    )
                    summary_text = summary["summary_text"]
                    metrics = summary.get("metrics", {})

                    # Build chat messages and query model
                    msgs = build_messages(summary_text, metrics)
                    # Save prompts/messages for debugging if requested
                    if f_prompts is not None:
                        f_prompts.write(json.dumps({
                            "episode_id": ep.stem,
                            "path": str(ep),
                            "messages": msgs,
                            "summary_text": summary_text,
                            "metrics": metrics,
                        }, ensure_ascii=False) + "\n")
                        f_prompts.flush()
                    outputs = pipe(
                        msgs,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=terminators,
                        do_sample=bool(do_sample),
                        temperature=temperature,
                        top_p=top_p,
                    )
                    # HF pipeline for chat returns generated_text as a list of messages
                    # Try to locate final assistant content
                    content = None
                    try:
                        content = outputs[0]["generated_text"][-1]["content"]
                    except Exception:
                        # Fallback: try full text field
                        content = str(outputs[0].get("generated_text", ""))

                    parsed = extract_json_object(content or "")
                    if parsed is None:
                        # Record as error and continue (do not assign default scores)
                        record = {
                            "episode_id": ep.stem,
                            "path": str(ep),
                            "task_name": summary.get("task_name", task_name_hint),
                            "length": summary.get("length", None),
                            "error": "JSON parsing failed for model output.",
                            "justification": "",
                            "summary_text": summary_text,
                            "raw_model_text": content,
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        scored.append(record)
                        continue

                    # Coerce integers safely and clamp to [1,10]
                    def _to_int(x):
                        try:
                            n = int(round(float(x)))
                        except Exception:
                            n = 5
                        return max(1, min(10, n))

                    dr = _to_int(parsed.get("dynamic_richness_score", 5))
                    cc = _to_int(parsed.get("causal_coherence_score", 5))
                    ov = _to_int(parsed.get("overall_learning_value", 5))
                    just = str(parsed.get("justification", "")).strip()

                    record = {
                        "episode_id": ep.stem,  # e.g., episode_00001
                        "path": str(ep),
                        "task_name": summary.get("task_name", task_name_hint),
                        "length": summary.get("length", None),
                        "scores": {
                            "dynamic_richness_score": dr,
                            "causal_coherence_score": cc,
                            "overall_learning_value": ov,
                        },
                        "justification": just,
                        "summary_text": summary_text,
                        "raw_model_text": content,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    scored.append(record)
                except Exception as e:
                    # Log failed episode as error (no default scores)
                    record = {
                        "episode_id": ep.stem,
                        "path": str(ep),
                        "error": str(e),
                        "justification": "",
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    scored.append(record)

                if idx % 25 == 0 or idx == len(ep_files):
                    print(f"Scored {idx}/{len(ep_files)} episodes...")
        finally:
            if f_prompts is not None:
                try:
                    f_prompts.close()
                except Exception:
                    pass

    # Rank and emit top-K by overall then dynamic then causal
    def _key(r):
        s = r.get("scores", {})
        return (
            int(s.get("overall_learning_value", 0)),
            int(s.get("dynamic_richness_score", 0)),
            int(s.get("causal_coherence_score", 0)),
        )

    valid_scored = [
        r for r in scored
        if isinstance(r.get("scores"), dict)
        and all(k in r["scores"] for k in ("overall_learning_value", "dynamic_richness_score", "causal_coherence_score"))
    ]
    ranked = sorted(valid_scored, key=_key, reverse=True)
    top_ids = [r["episode_id"] for r in ranked[:topk]]
    topk_out.parent.mkdir(parents=True, exist_ok=True)
    with open(topk_out, "w", encoding="utf-8") as f_top:
        for eid in top_ids:
            f_top.write(eid + "\n")
    print(f"Wrote top-{topk} episode IDs to {topk_out}")

    # Also write a single TXT containing per-episode scores and justification
    scores_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_txt, "w", encoding="utf-8") as f_txt:
        f_txt.write(f"# Scores for episodes in {input_dir}\n")
        f_txt.write(f"# Sorted by overall_learning_value, dynamic_richness, causal_coherence (desc)\n\n")
        for r in ranked:
            s = r.get("scores", {})
            line = (
                f"{r.get('episode_id','unknown')}\t"
                f"overall={int(s.get('overall_learning_value', 0))} "
                f"dynamic={int(s.get('dynamic_richness_score', 0))} "
                f"causal={int(s.get('causal_coherence_score', 0))}\t"
                f"justification: {r.get('justification','')}"
            )
            f_txt.write(line + "\n")
        # Append errors section
        errors = [r for r in scored if not isinstance(r.get("scores"), dict)]
        if errors:
            f_txt.write("\n# Errors (episodes not ranked due to failures)\n")
            for r in errors:
                f_txt.write(f"{r.get('episode_id','unknown')}\tERROR: {r.get('error','unknown error')}\n")
    print(f"Wrote per-episode scores and justifications to {scores_txt}")

    return scored, top_ids


def parse_args():
    p = argparse.ArgumentParser(description="Score DM Control trajectories with an LLM and select top-K.")
    p.add_argument("--input-dir", type=str, required=True, help="Directory containing episode .npz files.")
    p.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model id.")
    p.add_argument("--output-jsonl", type=str, default="scores.jsonl", help="Where to save per-episode scores.")
    p.add_argument("--scores-txt", type=str, default="scores.txt", help="Where to save a single TXT with scores and justifications.")
    p.add_argument("--prompts-jsonl", type=str, default="", help="Optional path to save per-episode prompts/messages for debugging.")
    p.add_argument("--mapping-json", type=str, default="", help="Path to mapping JSON to run offline without dm_control.")
    p.add_argument("--offline", action="store_true", help="Force offline mode (require --mapping-json).")
    p.add_argument("--do-sample", action="store_true", help="Enable sampling (by default disabled for deterministic scoring).")
    p.add_argument("--topk", type=int, default=10, help="How many top episodes to select.")
    p.add_argument("--topk-out", type=str, default="top_ids.txt", help="Path to write the top-K episode IDs.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--limit", type=int, default=100, help="Optional limit on number of episodes for debugging.")
    p.add_argument("--task-hint", type=str, default="humanoid-stand", help="Optional task name hint for summarizer.")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_jsonl = Path(args.output_jsonl)
    topk_out = Path(args.topk_out)

    score_directory(
        input_dir=input_dir,
        model_name=args.model,
        output_jsonl=output_jsonl,
        scores_txt=Path(args.scores_txt),
        prompts_jsonl=(Path(args.prompts_jsonl) if args.prompts_jsonl else None),
        topk=args.topk,
        topk_out=topk_out,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device_map=args.device_map,
        dtype=args.dtype,
        limit=args.limit,
        task_name_hint=(args.task_hint or None),
        mapping_json=(Path(args.mapping_json) if args.mapping_json else None),
        offline=bool(args.offline or args.mapping_json),
        do_sample=bool(args.do_sample),
    )


if __name__ == "__main__":
    main()


