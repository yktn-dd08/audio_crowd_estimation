#!/usr/bin/env python3
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def parse_linestring(text):
    s = text.strip()
    if not s.upper().startswith("LINESTRING"):
        raise ValueError(f"expected LINESTRING WKT, got: {s[:40]}")
    body = s[s.find("(") + 1:s.rfind(")")]
    coords = []
    for token in body.split(","):
        parts = token.strip().split()
        if len(parts) < 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        if math.isfinite(x) and math.isfinite(y):
            coords.append((x, y))
    return coords


def parse_datetime(text):
    s = text.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def turn_angle_deg(v1, v2):
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 <= 1e-9 or n2 <= 1e-9:
        return None
    c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(c))


def evaluate(path, target_speed, dt=1.0, expected_person_num=None, collision_distance=0.6):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty CSV")
    required = {"id", "start_time", "geom"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing SFM CSV columns: {sorted(missing)}")

    parsed = []
    for row in rows:
        coords = parse_linestring(row["geom"])
        if len(coords) < 2:
            continue
        parsed.append((str(row["id"]), parse_datetime(row["start_time"]), coords))
    if not parsed:
        raise ValueError("no valid LINESTRING trajectories")

    base_time = min(st for _, st, _ in parsed)
    speeds = []
    vector_accs = []
    scalar_speed_accs = []
    turns = []
    tortuosities = []
    backtrack_events = 0
    backtrack_checks = 0
    oscillating_agents = 0
    time_positions = {}

    for pid, st, pts in parsed:
        start_sec = (st - base_time).total_seconds()
        for i, (x, y) in enumerate(pts):
            tick = int(round((start_sec + i * dt) / dt))
            time_positions.setdefault(tick, []).append((pid, x, y))

        velocities = []
        path_len = 0.0
        for a, b in zip(pts, pts[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            dist = math.hypot(dx, dy)
            vx, vy = dx / dt, dy / dt
            velocities.append((vx, vy))
            speeds.append(dist / dt)
            path_len += dist

        agent_backtracks = 0
        for v1, v2 in zip(velocities, velocities[1:]):
            # Vector acceleration catches direction reversal even when speed magnitude is unchanged.
            vector_accs.append(math.hypot(v2[0] - v1[0], v2[1] - v1[1]) / dt)
            scalar_speed_accs.append(abs(math.hypot(*v2) - math.hypot(*v1)) / dt)
            ang = turn_angle_deg(v1, v2)
            if ang is not None:
                turns.append(ang)

        # A-B-A-like two-step motion: high travelled distance but little net progress.
        for p0, p1, p2 in zip(pts, pts[1:], pts[2:]):
            d1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            d2 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            travelled = d1 + d2
            if d1 < 0.05 or d2 < 0.05 or travelled < 0.20:
                continue
            backtrack_checks += 1
            net = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
            efficiency = net / travelled
            if efficiency < 0.35:
                backtrack_events += 1
                agent_backtracks += 1

        if agent_backtracks >= 2:
            oscillating_agents += 1

        direct = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
        if direct > 1e-9:
            tortuosities.append(path_len / direct)

    collision_count = 0
    pair_checks = 0
    collision_frames = 0
    for people in time_positions.values():
        frame_hit = False
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                pair_checks += 1
                if math.hypot(people[i][1] - people[j][1], people[i][2] - people[j][2]) < collision_distance:
                    collision_count += 1
                    frame_hit = True
        collision_frames += int(frame_hit)

    if not speeds:
        raise ValueError("no valid trajectory steps")
    mean_speed = sum(speeds) / len(speeds)
    output_person_num = len(parsed)
    reversal_rate = sum(v >= 120.0 for v in turns) / len(turns) if turns else 0.0
    severe_reversal_rate = sum(v >= 150.0 for v in turns) / len(turns) if turns else 0.0
    backtrack_rate = backtrack_events / backtrack_checks if backtrack_checks else 0.0

    return {
        "mean_speed": mean_speed,
        "p95_speed": percentile(speeds, 0.95),
        "max_speed": max(speeds),
        "mean_acc": sum(vector_accs) / len(vector_accs) if vector_accs else None,
        "p95_acc": percentile(vector_accs, 0.95),
        "max_acc": max(vector_accs) if vector_accs else None,
        "mean_speed_change_acc": sum(scalar_speed_accs) / len(scalar_speed_accs) if scalar_speed_accs else None,
        "jitter_mean_deg": sum(turns) / len(turns) if turns else None,
        "jitter_large_turn_rate": sum(v >= 90.0 for v in turns) / len(turns) if turns else 0.0,
        "reversal_rate": reversal_rate,
        "severe_reversal_rate": severe_reversal_rate,
        "two_step_backtrack_rate": backtrack_rate,
        "oscillating_agent_ratio": oscillating_agents / output_person_num if output_person_num else 0.0,
        "path_tortuosity_mean": sum(tortuosities) / len(tortuosities) if tortuosities else None,
        "target_speed_error": abs(mean_speed - target_speed),
        "collision_count": collision_count,
        "collision_pair_rate": collision_count / pair_checks if pair_checks else 0.0,
        "collision_frame_rate": collision_frames / len(time_positions) if time_positions else 0.0,
        "output_person_num": output_person_num,
        "output_person_ratio": (output_person_num / expected_person_num) if expected_person_num else None,
        "trajectory_steps": len(speeds),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate SocialForceSimulation CSV (id,start_time,geom)")
    p.add_argument("csv")
    p.add_argument("--target-speed", type=float, required=True)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--person-num", type=int)
    p.add_argument("--collision-distance", type=float, default=0.6)
    p.add_argument("--output")
    args = p.parse_args()
    metrics = evaluate(args.csv, args.target_speed, args.dt, args.person_num, args.collision_distance)
    text = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
