#!/usr/bin/env python3
import argparse
import json

KEYS = ["c_obs", "r_obs", "c_wall", "r_wall"]


def van_der_corput(n, base):
    vdc, denom = 0.0, 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        vdc += rem / denom
    return vdc


def global_bounds(base):
    """Conservative bounds for dt=1 stability-first tuning."""
    return {
        "c_obs": (0.70 * base["c_obs"], 1.25 * base["c_obs"]),
        "r_obs": (0.75 * base["r_obs"], 1.15 * base["r_obs"]),
        "c_wall": (0.70 * base["c_wall"], 1.25 * base["c_wall"]),
        "r_wall": (0.80 * base["r_wall"], min(0.95, 1.10 * base["r_wall"])),
    }


def expanded_bounds(base, level=1):
    """Broader deterministic envelope used only when strict candidates are scarce."""
    if level < 1:
        return global_bounds(base)
    progress = min(level, 4) / 4.0
    c_lo = 0.70 - 0.35 * progress
    c_hi = 1.25 + 0.50 * progress
    r_obs_lo = 0.75 - 0.35 * progress
    r_obs_hi = 1.15 + 0.15 * progress
    r_wall_lo = 0.80 - 0.40 * progress
    r_wall_hi = 1.10 + 0.05 * progress
    return {
        "c_obs": (c_lo * base["c_obs"], c_hi * base["c_obs"]),
        "r_obs": (r_obs_lo * base["r_obs"], r_obs_hi * base["r_obs"]),
        "c_wall": (c_lo * base["c_wall"], c_hi * base["c_wall"]),
        "r_wall": (r_wall_lo * base["r_wall"], min(0.95, r_wall_hi * base["r_wall"])),
    }


def refine_bounds(center, outer):
    # Refine must become narrower; never expand beyond original coarse bounds.
    local_factor = {
        "c_obs": (0.90, 1.10),
        "r_obs": (0.925, 1.075),
        "c_wall": (0.90, 1.10),
        "r_wall": (0.925, 1.075),
    }
    out = {}
    for key in KEYS:
        lo0, hi0 = outer[key]
        flo, fhi = local_factor[key]
        out[key] = (max(lo0, center[key] * flo), min(hi0, center[key] * fhi))
    return out


def _sample(bounds, count, include=None, start_index=1):
    primes = [2, 3, 5, 7]
    out = []
    for i in range(start_index, start_index + count):
        cand = {}
        for key, prime in zip(KEYS, primes):
            lo, hi = bounds[key]
            q = van_der_corput(i, prime)
            cand[key] = round(max(lo + q * (hi - lo), 1e-6), 6)
        out.append(cand)
    if count > 0 and include is not None:
        out[0] = {k: float(include[k]) for k in KEYS}
    return out


def make_candidates(base, count=48, stage="coarse", original_base=None, expansion_level=1, start_index=1):
    original = original_base or base
    outer = global_bounds(original)
    if stage == "coarse":
        bounds = outer
        include = original
    elif stage == "refine":
        bounds = refine_bounds(base, outer)
        include = base
    elif stage == "expanded":
        bounds = expanded_bounds(original, expansion_level)
        include = original
    else:
        raise ValueError("stage must be 'coarse', 'refine', or 'expanded'")
    return _sample(bounds, count, include=include, start_index=start_index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c-obs", type=float, required=True)
    p.add_argument("--r-obs", type=float, required=True)
    p.add_argument("--c-wall", type=float, required=True)
    p.add_argument("--r-wall", type=float, required=True)
    p.add_argument("--count", type=int, default=48)
    p.add_argument("--stage", choices=["coarse", "refine", "expanded"], default="coarse")
    p.add_argument("--expansion-level", type=int, default=1)
    p.add_argument("--start-index", type=int, default=1)
    p.add_argument("--output")
    args = p.parse_args()
    base = {"c_obs": args.c_obs, "r_obs": args.r_obs, "c_wall": args.c_wall, "r_wall": args.r_wall}
    obj = make_candidates(
        base,
        args.count,
        stage=args.stage,
        original_base=base,
        expansion_level=args.expansion_level,
        start_index=args.start_index,
    )
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
