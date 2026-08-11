#!/usr/bin/env python3
import argparse, json, math, sys

REQUIRED_TOP = {"schema_version", "base_config", "search", "scoring", "results"}
REQUIRED_SET = {"rank", "c_obs", "r_obs", "c_wall", "r_wall", "desired_speed", "score", "constraint_status", "metrics", "flags"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_path")
    args = p.parse_args()
    obj = json.load(open(args.json_path, encoding="utf-8"))
    errors = []
    if obj.get("schema_version") != "1.2": errors.append("schema_version must be 1.2")
    missing = REQUIRED_TOP - obj.keys()
    if missing: errors.append(f"missing top-level keys: {sorted(missing)}")
    top_k = obj.get("search", {}).get("top_k")
    dry_run = obj.get("search", {}).get("dry_run", False)
    for i, result in enumerate(obj.get("results", [])):
        parameter_sets = result.get("parameter_sets", [])
        if not dry_run and len(parameter_sets) != top_k:
            errors.append(f"results[{i}].parameter_sets must contain exactly top_k={top_k}, got {len(parameter_sets)}")
        if [ps.get("rank") for ps in parameter_sets] != list(range(1, len(parameter_sets) + 1)):
            errors.append(f"results[{i}] ranks must be contiguous from 1")
        tiers = [0 if ps.get("constraint_status") == "strict" else 1 for ps in parameter_sets]
        if tiers != sorted(tiers):
            errors.append(f"results[{i}] strict recommendations must precede fallback recommendations")
        for tier in ("strict", "fallback"):
            scores = [ps.get("score") for ps in parameter_sets if ps.get("constraint_status") == tier]
            if scores != sorted(scores, reverse=True):
                errors.append(f"results[{i}] {tier} scores must be descending")
        for j, ps in enumerate(parameter_sets):
            miss = REQUIRED_SET - ps.keys()
            if miss: errors.append(f"results[{i}].parameter_sets[{j}] missing {sorted(miss)}")
            if ps.get("constraint_status") not in {"strict", "fallback"}:
                errors.append(f"results[{i}].parameter_sets[{j}] has invalid constraint_status")
            for key in ("c_obs", "r_obs", "c_wall", "r_wall", "desired_speed", "score"):
                value = ps.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    errors.append(f"results[{i}].parameter_sets[{j}].{key} must be finite")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("valid")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
