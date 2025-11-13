#!/usr/bin/env python3
"""
Trajectory-to-Text: Convert DM Control rollout episodes into structured natural-language summaries
that can be fed to an LLM for trajectory scoring/ranking.

Usage examples:
  - Export mapping (requires dm_control available locally):
      python trajectory_to_text.py --export-mapping --task humanoid-stand --mapping-json humanoid_stand.mapping.json

  - Summarize all episodes in a task directory to JSONL (offline, using mapping):
      python trajectory_to_text.py \
        --input /shares/bcs516/sh/data/short_ep50_match_baseline_seed666/epoch_0000/task_00_humanoid-stand \
        --mapping-json humanoid_stand.mapping.json \
        --offline \
        --output summaries.jsonl

  - Preview first 3 episodes to stdout:
      python trajectory_to_text.py --input <episode_dir> --mapping-json humanoid_stand.mapping.json --offline --preview 3
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# from envs.__init__ import make_env


class C:
    """Lightweight config container (attribute access)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


# ------------------------------
# Environment / observation utils
# ------------------------------

def _unwrap_to_dm_env(env):
    """
    Descend wrappers and return the raw dm_control environment that exposes
    observation_spec() and returns dm_env.TimeStep from reset/step.
    """
    cur = env
    # env is typically: TensorWrapper(Timeout(DMControlWrapper(dm_control_env)))
    for _ in range(8):  # safety cap
        if hasattr(cur, "observation_spec") and callable(getattr(cur, "observation_spec")):
            # Already at dm_control env
            return cur
        # DMControlWrapper exposes .unwrapped -> dm_control env
        if hasattr(cur, "unwrapped"):
            maybe = getattr(cur, "unwrapped")
            # If unwrapped is a property returning the raw env, use it
            if hasattr(maybe, "observation_spec"):
                return maybe
        # Otherwise go one level deeper if .env exists
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        break
    return None


@dataclass
class ObsSegment:
    key: str
    start: int
    end: int
    shape: Tuple[int, ...]

    @property
    def length(self) -> int:
        return self.end - self.start


def _segments_to_jsonable(segments: List[ObsSegment]) -> List[Dict[str, object]]:
    return [{"key": s.key, "start": s.start, "end": s.end, "shape": list(s.shape)} for s in segments]


def _segments_from_jsonable(items: List[Dict[str, object]]) -> List[ObsSegment]:
    segs: List[ObsSegment] = []
    for it in items:
        segs.append(ObsSegment(key=str(it["key"]), start=int(it["start"]), end=int(it["end"]), shape=tuple(int(x) for x in it.get("shape", []))))
    return segs


def load_segments_from_file(mapping_json: Path) -> List[ObsSegment]:
    with open(mapping_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either {"segments": [...]} or a top-level list
    if isinstance(data, dict) and "segments" in data:
        items = data["segments"]
    else:
        items = data
    return _segments_from_jsonable(items)


def write_segments_to_file(segments: List[ObsSegment], mapping_json: Path) -> None:
    mapping_json.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump({"segments": _segments_to_jsonable(segments)}, f, indent=2)


def get_observation_segments(task_name: str, seed: int = 0) -> List[ObsSegment]:
    """
    Build a mapping from flattened 'state' vector indices to dm_control observation dict keys.
    This mirrors DMControlWrapper._obs_to_array which concatenates obs.values() in iteration order.
    """
    cfg = C(multitask=False, task=task_name, obs="state", seed=seed)
    env = make_env(cfg)
    dm_env = _unwrap_to_dm_env(env)
    if dm_env is None:
        raise RuntimeError("Failed to locate underlying dm_control env to read observation spec.")

    # Use observation_spec() for canonical ordering/shapes; fallback to a reset() snapshot if needed.
    spec = dm_env.observation_spec()
    keys = list(spec.keys())
    lengths = []
    shapes: List[Tuple[int, ...]] = []
    for k in keys:
        shp = spec[k].shape
        # dm_control spec shapes can be (), convert to tuple[int, ...]
        if isinstance(shp, tuple):
            shape = shp
        else:
            # scalars or others
            try:
                # Some spec-like objects might have .shape attribute that is not a tuple
                shape = tuple(np.atleast_1d(shp).astype(int).tolist())
            except Exception:
                shape = ()
        size = int(np.prod(shape) if len(shape) > 0 else 1)
        shapes.append(shape)
        lengths.append(size)

    segments: List[ObsSegment] = []
    cursor = 0
    for k, size, shp in zip(keys, lengths, shapes):
        segments.append(ObsSegment(key=k, start=cursor, end=cursor + size, shape=tuple(shp)))
        cursor += size
    return segments


def _find_segment(segments: List[ObsSegment], candidates: Iterable[str]) -> Optional[ObsSegment]:
    cand_lower = [c.lower() for c in candidates]
    for seg in segments:
        name = seg.key.lower()
        for c in cand_lower:
            if c in name:
                return seg
    return None


# ------------------------------
# Metrics extraction
# ------------------------------

@dataclass
class TrajectoryMetrics:
    length: int
    total_reward: Optional[float]
    avg_reward: Optional[float]
    success_rate: Optional[float]
    # Humanoid-centric (when available)
    height_stats: Optional[Dict[str, float]]
    com_speed_stats: Optional[Dict[str, float]]
    upright_stats: Optional[Dict[str, float]]
    # Generic
    action_magnitude_mean: float
    action_smoothness_mean: float
    state_change_magnitude_mean: float


def _basic_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "p25": float(np.nanpercentile(x, 25)),
        "p50": float(np.nanpercentile(x, 50)),
        "p75": float(np.nanpercentile(x, 75)),
        "start": float(x[0]) if x.size > 0 else float("nan"),
        "end": float(x[-1]) if x.size > 0 else float("nan"),
    }


def extract_metrics_for_humanoid_like(
    states: np.ndarray,
    actions: np.ndarray,
    segments: List[ObsSegment],
    rewards: Optional[np.ndarray] = None,
    success: Optional[np.ndarray] = None,
) -> TrajectoryMetrics:
    """
    Try to extract interpretable signals commonly present in humanoid/walker-type observations:
      - height or head_height (scalar)
      - com_velocity / center_of_mass_velocity (3D)
      - torso_vertical / upright (1D or 3D vector; use z component when vector)
    Fallbacks to generic stats when segments are absent.
    """
    T = states.shape[0]

    # Reward / success
    total_reward = float(np.nansum(rewards)) if rewards is not None else None
    avg_reward = float(np.nanmean(rewards)) if rewards is not None else None
    success_rate = float(np.nanmean(success)) if success is not None else None

    # Segments of interest
    height_seg = _find_segment(segments, ["head_height", "height"])
    com_vel_seg = _find_segment(segments, ["com_velocity", "center_of_mass_velocity", "com_vel"])
    upright_seg = _find_segment(segments, ["torso_vertical", "upright"])

    height_stats = None
    if height_seg is not None:
        h = states[:, height_seg.start:height_seg.end].reshape(T, -1)
        # Prefer scalar; if vector, take norm
        if h.shape[1] == 1:
            height_series = h[:, 0]
        else:
            height_series = np.linalg.norm(h, axis=1)
        height_stats = _basic_stats(height_series)

    com_speed_stats = None
    if com_vel_seg is not None:
        v = states[:, com_vel_seg.start:com_vel_seg.end].reshape(T, -1)
        if v.shape[1] >= 2:
            speed = np.linalg.norm(v, axis=1)
        else:
            speed = np.abs(v[:, 0])
        com_speed_stats = _basic_stats(speed)

    upright_stats = None
    if upright_seg is not None:
        u = states[:, upright_seg.start:upright_seg.end].reshape(T, -1)
        if u.shape[1] == 1:
            upright_series = u[:, 0]
        else:
            # Heuristic: z component is the last or the third element if length>=3
            idx = 2 if u.shape[1] >= 3 else u.shape[1] - 1
            upright_series = u[:, idx]
        upright_stats = _basic_stats(upright_series)

    # Generic dynamics/actions stats
    act_mag = np.linalg.norm(actions, axis=1) if actions.size > 0 else np.array([np.nan])
    act_smooth = (
        np.linalg.norm(np.diff(actions, axis=0), axis=1) if actions.shape[0] > 1 else np.array([np.nan])
    )
    st_change = (
        np.linalg.norm(np.diff(states, axis=0), axis=1) if states.shape[0] > 1 else np.array([np.nan])
    )

    return TrajectoryMetrics(
        length=T,
        total_reward=total_reward,
        avg_reward=avg_reward,
        success_rate=success_rate,
        height_stats=height_stats,
        com_speed_stats=com_speed_stats,
        upright_stats=upright_stats,
        action_magnitude_mean=float(np.nanmean(act_mag)),
        action_smoothness_mean=float(np.nanmean(act_smooth)),
        state_change_magnitude_mean=float(np.nanmean(st_change)),
    )


# ------------------------------
# Natural language rendering
# ------------------------------

def trajectory_to_text(
    task_name: str,
    metrics: TrajectoryMetrics,
) -> str:
    """
    Render a compact, structured description that an LLM can consume.
    Keep it deterministic and templated to be easy to parse.
    """
    lines: List[str] = []
    lines.append(f"Task: {task_name}")
    lines.append(f"Episode length: {metrics.length} steps")
    if metrics.total_reward is not None:
        lines.append(f"Total reward: {metrics.total_reward:.3f} (avg {metrics.avg_reward:.3f}/step)")
    if metrics.success_rate is not None:
        lines.append(f"Success ratio: {metrics.success_rate:.3f}")

    def _fmt_stats(name: str, stats: Optional[Dict[str, float]], unit: str = ""):
        if stats is None:
            return
        unit_sp = f" {unit}" if unit else ""
        lines.append(
            f"{name}: "
            f"mean {stats['mean']:.3f}{unit_sp}, std {stats['std']:.3f}{unit_sp}, "
            f"min {stats['min']:.3f}{unit_sp}, max {stats['max']:.3f}{unit_sp}, "
            f"start {stats['start']:.3f}{unit_sp} -> end {stats['end']:.3f}{unit_sp}"
        )

    # Humanoid-centric when present
    _fmt_stats("Torso/head height", metrics.height_stats, "m")
    _fmt_stats("COM speed", metrics.com_speed_stats, "m/s")
    _fmt_stats("Uprightness (z)", metrics.upright_stats, "")

    # Generic dynamics/actions
    lines.append(f"Action magnitude (mean L2): {metrics.action_magnitude_mean:.3f}")
    lines.append(f"Action smoothness (mean L2 diff): {metrics.action_smoothness_mean:.3f}")
    lines.append(f"State change magnitude (mean L2): {metrics.state_change_magnitude_mean:.3f}")

    # Heuristic qualitative notes
    notes: List[str] = []
    if metrics.height_stats is not None:
        if metrics.height_stats["end"] < 0.5 * max(1e-6, metrics.height_stats["start"]):
            notes.append("Significant height drop (possible fall).")
    if metrics.upright_stats is not None and metrics.upright_stats["mean"] < 0.5:
        notes.append("Low average uprightness.")
    if metrics.com_speed_stats is not None and metrics.com_speed_stats["mean"] < 0.1:
        notes.append("Little horizontal movement.")
    if notes:
        lines.append("Notes: " + " ".join(notes))

    return "\n".join(lines)


# ------------------------------
# I/O and CLI
# ------------------------------

def load_episode_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def summarize_episode(
    episode_path: Path,
    task_name_hint: Optional[str] = None,
    obs_segments_cache: Dict[str, List[ObsSegment]] = None,
    mapping_json: Optional[Path] = None,
    offline: bool = False,
) -> Dict[str, object]:
    arrs = load_episode_npz(episode_path)
    states = np.asarray(arrs["states"], dtype=np.float32)
    actions = np.asarray(arrs["actions"], dtype=np.float32)
    rewards = np.asarray(arrs.get("rewards", None), dtype=np.float32) if "rewards" in arrs else None
    success = np.asarray(arrs.get("success", None), dtype=np.float32) if "success" in arrs else None
    # Prefer stored task_name; fallback to hint
    task_name = task_name_hint
    if "task_name" in arrs:
        # Saved as numpy scalar/string array; convert to Python str
        tn = arrs["task_name"]
        try:
            task_name = str(tn.tolist())  # handles np.str_, arrays of length-1
        except Exception:
            task_name = task_name_hint
    if task_name is None:
        raise ValueError(f"Cannot determine task_name for {episode_path}")

    # Build or reuse observation segments
    if obs_segments_cache is None:
        obs_segments_cache = {}
    if task_name not in obs_segments_cache:
        if mapping_json:
            obs_segments_cache[task_name] = load_segments_from_file(Path(mapping_json))
        else:
            if offline:
                raise RuntimeError("Offline mode requested but no --mapping-json provided.")
            obs_segments_cache[task_name] = get_observation_segments(task_name)
    segments = obs_segments_cache[task_name]

    # Extract metrics and text
    metrics = extract_metrics_for_humanoid_like(states, actions, segments, rewards=rewards, success=success)
    text = trajectory_to_text(task_name, metrics)
    return {
        "path": str(episode_path),
        "task_name": task_name,
        "length": int(states.shape[0]),
        "summary_text": text,
        "metrics": metrics.__dict__,
    }


def iter_episode_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file() and input_path.suffix == ".npz":
        yield input_path
        return
    for p in sorted(input_path.glob("*.npz")):
        if p.is_file():
            yield p


def main():
    parser = argparse.ArgumentParser(description="Convert DM Control episodes to structured natural-language summaries.")
    parser.add_argument("--input", type=str, required=True, help="Episode .npz file or directory containing episodes.")
    parser.add_argument("--output", type=str, default="", help="Path to write JSONL with summaries. If empty, only prints previews.")
    parser.add_argument("--preview", type=int, default=0, help="Print first N summaries to stdout.")
    parser.add_argument("--task", type=str, default="", help="Optional task name hint (e.g., humanoid-stand).")
    parser.add_argument("--mapping-json", type=str, default="", help="Path to JSON with precomputed observation segments mapping.")
    parser.add_argument("--offline", action="store_true", help="Do not access dm_control env; require --mapping-json.")
    parser.add_argument("--export-mapping", action="store_true", help="Export mapping JSON for --task using local dm_control env and exit.")
    args = parser.parse_args()

    # Export mapping mode
    if args.export_mapping:
        if not args.task:
            raise ValueError("--export-mapping requires --task to be set.")
        if not args.mapping_json:
            raise ValueError("--export-mapping requires --mapping-json path to write.")
        segs = get_observation_segments(args.task)
        write_segments_to_file(segs, Path(args.mapping_json))
        print(f"Wrote mapping for task '{args.task}' to {args.mapping_json}")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    obs_segments_cache: Dict[str, List[ObsSegment]] = {}
    summaries: List[Dict[str, object]] = []
    count = 0
    for ep_file in iter_episode_files(input_path):
        s = summarize_episode(
            ep_file,
            task_name_hint=(args.task or None),
            obs_segments_cache=obs_segments_cache,
            mapping_json=(Path(args.mapping_json) if args.mapping_json else None),
            offline=args.offline,
        )
        summaries.append(s)
        count += 1
        if args.preview and count <= args.preview:
            # Print only the structured text for readability
            print(f"=== {ep_file.name} ===")
            print(s["summary_text"])
            print()

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() != ".jsonl":
            out_path = out_path.with_suffix(".jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for s in summaries:
                # Convert dataclass metrics to plain dict with JSON-serializable floats
                m = s["metrics"]
                # Cast nested stats dicts to python floats to be safe
                for k in list(m.keys()):
                    if isinstance(m[k], dict):
                        m[k] = {kk: float(vv) for kk, vv in m[k].items()}
                    elif isinstance(m[k], (np.floating, np.integer)):
                        m[k] = float(m[k])
                rec = {
                    "path": s["path"],
                    "task_name": s["task_name"],
                    "length": s["length"],
                    "summary_text": s["summary_text"],
                    "metrics": m,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(summaries)} summaries to {out_path}")
    elif not args.preview:
        # If no output and no preview, default to printing count
        print(f"Prepared {len(summaries)} summaries. Use --output to save or --preview to print.")


if __name__ == "__main__":
    main()


