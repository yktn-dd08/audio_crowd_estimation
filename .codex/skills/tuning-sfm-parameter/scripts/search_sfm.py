#!/usr/bin/env python3
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

from evaluate_sfm_csv import evaluate
from generate_candidates import expanded_bounds, make_candidates, global_bounds

PARAM_KEYS = ("c_obs", "r_obs", "c_wall", "r_wall")


def score_metrics(m, target):
    # Stability-first ranking. Direction reversals and short-period backtracking
    # are weighted strongly because scalar speed acceleration alone misses them.
    terms = []
    speed_scale = max(0.2, 0.25 * max(target, 0.1))
    terms.append((0.25, math.exp(-m["target_speed_error"] / speed_scale)))
    if m.get("p95_acc") is not None:
        terms.append((0.20, math.exp(-m["p95_acc"] / 1.5)))

    reversal = m.get("reversal_rate", 0.0)
    severe = m.get("severe_reversal_rate", 0.0)
    backtrack = m.get("two_step_backtrack_rate", 0.0)
    oscillating_agents = m.get("oscillating_agent_ratio", 0.0)
    oscillation_score = math.exp(
        -10.0 * reversal
        -18.0 * severe
        -16.0 * backtrack
        -8.0 * oscillating_agents
    )
    terms.append((0.25, oscillation_score))

    if m.get("path_tortuosity_mean") is not None:
        terms.append((0.05, math.exp(-max(0.0, m["path_tortuosity_mean"] - 1.0) / 0.5)))
    terms.append((0.20, math.exp(-25.0 * m.get("collision_pair_rate", 0.0))))
    if m.get("output_person_ratio") is not None:
        terms.append((0.05, min(1.0, max(0.0, m["output_person_ratio"]))))
    total = sum(w for w, _ in terms)
    return sum(w * v for w, v in terms) / total if total else 0.0


def stability_flags(metrics):
    flags = []
    if metrics.get("collision_pair_rate", 0.0) > 0.01:
        flags.append("high_collision_rate")
    if metrics.get("output_person_ratio") is not None and metrics["output_person_ratio"] < 0.9:
        flags.append("low_output_person_ratio")
    if metrics.get("reversal_rate", 0.0) > 0.03:
        flags.append("frequent_reversal")
    if metrics.get("two_step_backtrack_rate", 0.0) > 0.02:
        flags.append("frequent_backtracking")
    if metrics.get("oscillating_agent_ratio", 0.0) > 0.05:
        flags.append("oscillating_agents")
    return flags


def hard_constraint_flags(metrics):
    flags = []
    if metrics.get("severe_reversal_rate", 0.0) > 0.10:
        flags.append("hard_severe_reversal")
    if metrics.get("two_step_backtrack_rate", 0.0) > 0.08:
        flags.append("hard_two_step_backtracking")
    if metrics.get("oscillating_agent_ratio", 0.0) > 0.15:
        flags.append("hard_oscillating_agents")
    return flags


def config_hash(cfg):
    normalized = copy.deepcopy(cfg)
    for task in normalized.get("task_list", {}).values():
        task.pop("output_csv", None)
        task.pop("output_mp4", None)
    normalized.pop("overwrite", None)
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def set_effective(task, common, key, value):
    # crowd_trajectory.py resolves task value first, then common param.
    # Write both so an existing task-level override can never shadow the search condition.
    common[key] = value
    task[key] = value


def build_config(base, target_speed, person_num, candidate, csv_path, include_video=False, mp4_path=None):
    cfg = copy.deepcopy(base)
    if cfg.get("option") != "social_force":
        raise ValueError("base config option must be 'social_force'")
    if not cfg.get("task_list"):
        raise ValueError("task_list is empty")
    # The current simulator may contain multiple tasks, but parameter tuning is defined for one task at a time.
    if "sfm_crowd" in cfg["task_list"]:
        task_name = "sfm_crowd"
    elif len(cfg["task_list"]) == 1:
        task_name = next(iter(cfg["task_list"]))
    else:
        raise ValueError("multiple tasks found and no 'sfm_crowd' task exists")
    task = cfg["task_list"][task_name]
    common = cfg.setdefault("param", {})

    set_effective(task, common, "person_num", int(person_num))
    # In this implementation desired_speed is the simulator control parameter. Use the requested v as its value,
    # while ranking against the realized mean speed from the output trajectory.
    set_effective(task, common, "desired_speed", float(target_speed))
    for k, v in candidate.items():
        set_effective(task, common, k, float(v))

    task["output_csv"] = str(csv_path)
    if include_video:
        task["output_mp4"] = str(mp4_path)
        cfg["overwrite"] = True
    else:
        task.pop("output_mp4", None)
    return cfg, task_name


def run_one(base, target_speed, person_num, cand, workspace, command_template, dry_run):
    provisional, _ = build_config(base, target_speed, person_num, cand, "trajectory.csv")
    h = config_hash(provisional)
    run_dir = workspace / h
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "trajectory.csv"
    cfg, _ = build_config(base, target_speed, person_num, cand, csv_path)
    cfg_path = run_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if dry_run:
        return {"status": "dry_run", "hash": h, "config": str(cfg_path), **cand}

    if not csv_path.exists():
        cmd = command_template.format(config=shlex.quote(str(cfg_path)))
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        (run_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0 or not csv_path.exists():
            return {"status": "rejected", "flags": ["simulation_failed"], "hash": h, **cand}

    try:
        dt = float(cfg["task_list"][next(iter(cfg["task_list"]))].get("dt", cfg.get("param", {}).get("dt", 1.0)))
        metrics = evaluate(csv_path, target_speed, dt=dt, expected_person_num=int(person_num), collision_distance=0.6)
        if not all(v is None or (isinstance(v, (int, float)) and math.isfinite(v)) for v in metrics.values()):
            raise ValueError("non-finite metric")
        score = score_metrics(metrics, target_speed)
        flags = stability_flags(metrics)
        hard_flags = hard_constraint_flags(metrics)
        if hard_flags:
            return {
                "status": "constraint_rejected",
                "flags": ["fallback_constraint_violation", *hard_flags, *flags],
                "hash": h,
                "score": score,
                "metrics": metrics,
                **cand,
            }
        return {"status": "ok", "flags": flags, "hash": h, "score": score, "metrics": metrics, **cand}
    except Exception as e:
        return {"status": "rejected", "flags": ["trajectory_evaluation_failed", str(e)], "hash": h, **cand}


def render_top(base, ranked, target_speed, person_num, workspace, command_template, count):
    rendered = []
    for r in ranked[:count]:
        run_dir = workspace / r["hash"]
        csv_path = run_dir / "trajectory.csv"
        mp4_path = run_dir / "trajectory.mp4"
        cand = {k: r[k] for k in PARAM_KEYS}
        cfg, _ = build_config(base, target_speed, person_num, cand, csv_path, True, mp4_path)
        render_cfg = run_dir / "render_config.json"
        render_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cmd = command_template.format(config=shlex.quote(str(render_cfg)))
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        (run_dir / "render_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / "render_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode == 0 and mp4_path.exists():
            rendered.append(str(mp4_path))
    return rendered


def candidate_key(candidate):
    return tuple(round(float(candidate[key]), 9) for key in PARAM_KEYS)


def execute_batch(base, target_speed, person_num, candidates, workspace, command_template, dry_run, workers):
    def execute(candidate):
        return run_one(base, target_speed, person_num, candidate, workspace, command_template, dry_run)

    if workers == 1:
        return [execute(candidate) for candidate in candidates]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(execute, candidates))


def sort_runs(runs, status):
    return sorted(
        (run for run in runs if run.get("status") == status),
        key=lambda run: run.get("score", -1),
        reverse=True,
    )


def main():
    p = argparse.ArgumentParser(description="Batch SFM parameter search for simulation.crowd_trajectory")
    p.add_argument("--config", required=True)
    p.add_argument("--target-speeds", type=float, nargs="+", required=True)
    p.add_argument("--person-nums", type=int, nargs="+", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidates", type=int, default=48, help="minimum initial coarse candidates per condition")
    p.add_argument("--rounds", type=int, choices=range(1, 9), default=4, help="steps used to widen bounds before continuing at the maximum envelope")
    p.add_argument("--refine-centers", type=int, default=3)
    p.add_argument("--refine-candidates", type=int, default=12, help="candidates around each refine center")
    p.add_argument("--expansion-batch", type=int, default=48)
    p.add_argument("--max-candidates", type=int, default=384, help="hard evaluation budget per condition")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--render-top", type=int, default=0, help="rerun only top N candidates per condition with MP4 output")
    p.add_argument("--workspace", default="./.codex/workspace/tuning-sfm-parameter")
    p.add_argument("--command-template", default="python -m simulation.crowd_trajectory -c {config}")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_candidates < args.top_k:
        raise ValueError("--max-candidates must be at least --top-k")

    base_path = Path(args.config)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base.get("option") != "social_force":
        raise ValueError("This skill only tunes configs with option='social_force'.")
    tasks = base.get("task_list", {})
    task = tasks.get("sfm_crowd") if "sfm_crowd" in tasks else (next(iter(tasks.values())) if len(tasks) == 1 else None)
    if task is None:
        raise ValueError("Could not resolve SFM task")
    common = base.get("param", {})
    simulator_defaults = {"c_obs": 2000.0, "r_obs": 0.08, "c_wall": 2000.0, "r_wall": 0.08}
    def resolve_base_param(key):
        if task.get(key) is not None:
            return float(task[key])
        if common.get(key) is not None:
            return float(common[key])
        return simulator_defaults[key]

    base_params = {key: resolve_base_param(key) for key in PARAM_KEYS}

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results = []

    for speed in args.target_speeds:
        for pn in args.person_nums:
            runs_by_hash = {}
            seen_candidates = set()

            def run_new(candidate_pool):
                remaining = args.max_candidates - len(seen_candidates)
                if remaining <= 0:
                    return
                fresh = []
                for candidate in candidate_pool:
                    key = candidate_key(candidate)
                    if key not in seen_candidates:
                        seen_candidates.add(key)
                        fresh.append(candidate)
                    if len(fresh) >= remaining:
                        break
                for run in execute_batch(
                    base, speed, pn, fresh, workspace, args.command_template, args.dry_run, args.workers
                ):
                    runs_by_hash[run["hash"]] = run

            initial_count = max(args.candidates, args.top_k * 2)
            run_new(make_candidates(base_params, initial_count, stage="coarse", original_base=base_params))

            if not args.dry_run:
                strict = sort_runs(runs_by_hash.values(), "ok")
                for center_run in strict[:args.refine_centers]:
                    center = {key: float(center_run[key]) for key in PARAM_KEYS}
                    run_new(make_candidates(
                        center,
                        max(args.refine_candidates, args.top_k),
                        stage="refine",
                        original_base=base_params,
                    ))

                strict = sort_runs(runs_by_hash.values(), "ok")
                expansion_level = 1
                while (
                    len(strict) < args.top_k
                    and len(seen_candidates) < args.max_candidates
                ):
                    run_new(make_candidates(
                        base_params,
                        args.expansion_batch,
                        stage="expanded",
                        original_base=base_params,
                        expansion_level=min(expansion_level, args.rounds),
                        start_index=1 + (expansion_level - 1) * args.expansion_batch,
                    ))
                    strict = sort_runs(runs_by_hash.values(), "ok")
                    expansion_level += 1

                fallback = sort_runs(runs_by_hash.values(), "constraint_rejected")
                selected = strict[:args.top_k]
                if len(selected) < args.top_k:
                    selected.extend(fallback[:args.top_k - len(selected)])
                if len(selected) != args.top_k:
                    raise RuntimeError(
                        f"Could not produce exactly {args.top_k} evaluable recommendations for "
                        f"target_mean_speed={speed}, person_num={pn}; "
                        f"evaluated={len(runs_by_hash)}, strict={len(strict)}, fallback={len(fallback)}"
                    )
            else:
                selected = []

            rendered = []
            if args.render_top > 0 and selected and not args.dry_run:
                rendered = render_top(base, selected, speed, pn, workspace, args.command_template, min(args.render_top, args.top_k))

            param_sets = []
            for rank, r in enumerate(selected, 1):
                param_sets.append({
                    "rank": rank,
                    **{k: r[k] for k in PARAM_KEYS},
                    "desired_speed": float(speed),
                    "score": round(r["score"], 6),
                    "constraint_status": "strict" if r.get("status") == "ok" else "fallback",
                    "metrics": r["metrics"],
                    "flags": r.get("flags", []),
                    "hash": r["hash"],
                    "video": str(workspace / r["hash"] / "trajectory.mp4") if str(workspace / r["hash"] / "trajectory.mp4") in rendered else None,
                })
            results.append({
                "target_mean_speed": float(speed),
                "person_num": int(pn),
                "evaluated": len(runs_by_hash),
                "rejected": sum(1 for r in runs_by_hash.values() if r.get("status") != "ok"),
                "strict_recommendations": sum(r.get("status") == "ok" for r in selected),
                "fallback_recommendations": sum(r.get("status") == "constraint_rejected" for r in selected),
                "parameter_sets": param_sets,
                **({"dry_run_candidates": list(runs_by_hash.values())} if args.dry_run else {}),
            })

    output = {
        "schema_version": "1.2",
        "base_config": str(base_path),
        "search": {
            "target_mean_speed": args.target_speeds,
            "person_num": args.person_nums,
            "top_k": args.top_k,
            "candidates_per_condition": args.candidates,
            "rounds": args.rounds,
            "refine_centers": args.refine_centers,
            "refine_candidates_per_center": args.refine_candidates,
            "expansion_batch": args.expansion_batch,
            "max_candidates_per_condition": args.max_candidates,
            "workers": args.workers,
            "exact_k_required": True,
            "fallback_policy": "strict-first-then-constraint-violating-evaluable",
            "render_top": args.render_top,
            "dry_run": bool(args.dry_run),
            "collision_distance_m": 0.6,
            "profile": "stability-first",
            "coarse_bounds": global_bounds(base_params),
            "maximum_expanded_bounds": expanded_bounds(base_params, args.rounds),
        },
        "scoring": {
            "target_speed": 0.25,
            "vector_acceleration": 0.20,
            "oscillation": 0.25,
            "tortuosity": 0.05,
            "collision": 0.20,
            "output_person_ratio": 0.05,
        },
        "results": results,
    }
    out_path = workspace / "sfm_search_result.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
