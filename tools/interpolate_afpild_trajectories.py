"""Create one-second AFPILD trajectories by PCHIP interpolation."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from scipy.interpolate import PchipInterpolator


NAME_PATTERN = re.compile(r"s(?P<subject>\d{2})_(?P<session>[1-4])_footstep_trajectory\.csv$")


@dataclass(frozen=True)
class InterpolationSummary:
    subject: str
    session: int
    pid: str
    source_rows: int
    output_rows: int
    source_end_seconds: float
    output_end_seconds: int
    interpolation_method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate AFPILD trajectories at one-second intervals with "
            "shape-preserving PCHIP."
        )
    )
    parser.add_argument("input_dir", type=Path, help="AFPILD_v1_aligned directory")
    parser.add_argument("output_dir", type=Path, help="Directory for one-second trajectories")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after the new dataset validates.",
    )
    return parser.parse_args()


def read_trajectory(path: Path) -> tuple[list[float], list[float], list[float], str]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or list(rows[0]) != ["t", "pid", "x", "y"]:
        raise ValueError(f"Unexpected trajectory schema: {path}")

    times = [float(row["t"]) for row in rows]
    xs = [float(row["x"]) for row in rows]
    ys = [float(row["y"]) for row in rows]
    pids = {row["pid"] for row in rows}
    if len(pids) != 1:
        raise ValueError(f"Expected one pid per trajectory: {path}")
    if times[0] != 0.0:
        raise ValueError(f"Trajectory must start at t=0: {path}")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError(f"Trajectory time is not strictly increasing: {path}")
    return times, xs, ys, pids.pop()


def interpolate_one_second(
    times: list[float], xs: list[float], ys: list[float]
) -> list[tuple[int, float, float]]:
    targets = range(math.floor(times[-1]) + 1)
    x_spline = PchipInterpolator(times, xs, extrapolate=False)
    y_spline = PchipInterpolator(times, ys, extrapolate=False)
    return [
        (target, float(x_spline(target)), float(y_spline(target)))
        for target in targets
    ]


def write_trajectory(
    path: Path, rows: list[tuple[int, float, float]], pid: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["t", "pid", "x", "y"])
        for t, x, y in rows:
            writer.writerow([f"{t:.6f}", pid, f"{x:.10f}", f"{y:.10f}"])


def process_file(source_path: Path, input_root: Path, output_root: Path) -> InterpolationSummary:
    match = NAME_PATTERN.fullmatch(source_path.name)
    if match is None:
        raise ValueError(f"Unexpected trajectory filename: {source_path}")
    subject = f"S{match.group('subject')}"
    session = int(match.group("session"))
    times, xs, ys, pid = read_trajectory(source_path)
    interpolated = interpolate_one_second(times, xs, ys)
    output_name = source_path.name.replace("_trajectory.csv", "_trajectory_1s.csv")
    output_path = output_root / source_path.relative_to(input_root).parent / output_name
    write_trajectory(output_path, interpolated, pid)
    return InterpolationSummary(
        subject=subject,
        session=session,
        pid=pid,
        source_rows=len(times),
        output_rows=len(interpolated),
        source_end_seconds=times[-1],
        output_end_seconds=interpolated[-1][0],
        interpolation_method="pchip",
    )


def write_manifest(path: Path, summaries: list[InterpolationSummary]) -> None:
    fields = list(asdict(summaries[0]))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)


def validate_output(root: Path, summaries: list[InterpolationSummary]) -> None:
    output_files = sorted(root.glob("S??/*_trajectory_1s.csv"))
    if len(summaries) != 160 or len(output_files) != 160:
        raise ValueError(
            f"Expected 160 trajectories, got {len(summaries)} summaries and "
            f"{len(output_files)} files"
        )
    for path, summary in zip(output_files, summaries):
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        times = [float(row["t"]) for row in rows]
        if len(rows) != summary.output_rows:
            raise ValueError(f"Output row mismatch: {path}")
        if times != [float(value) for value in range(summary.output_rows)]:
            raise ValueError(f"Output is not at exact one-second intervals: {path}")
        if {row["pid"] for row in rows} != {summary.pid}:
            raise ValueError(f"PID mismatch: {path}")
        if times[-1] != summary.output_end_seconds:
            raise ValueError(f"End-time mismatch: {path}")
        if any(
            not math.isfinite(float(row[column]))
            for row in rows
            for column in ("x", "y")
        ):
            raise ValueError(f"Non-finite spline coordinate: {path}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_dir}")

    source_files = sorted(input_dir.glob("S??/*_trajectory.csv"))
    if len(source_files) != 160:
        raise ValueError(f"Expected 160 input trajectories, got {len(source_files)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        summaries = [
            process_file(source_path, input_dir, temporary_dir)
            for source_path in source_files
        ]
        write_manifest(temporary_dir / "manifest_1s.csv", summaries)
        (temporary_dir / "README.md").write_text(
            """# AFPILD one-second trajectories

These files are interpolated from the irregular footstep-coordinate timestamps
in `AFPILD_v1_aligned` using shape-preserving PCHIP independently for X and Y.
Each output starts at 0 seconds and contains coordinates at exactly 1-second
intervals through `floor(source end time)`. The schema is `t,pid,x,y`; `t` is
measured in seconds. No extrapolation is used.
""",
            encoding="utf-8",
        )
        validate_output(temporary_dir, summaries)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary_dir, output_dir)
        print(
            f"Created {len(summaries)} one-second trajectories "
            f"({sum(summary.output_rows for summary in summaries)} rows) in {output_dir}"
        )
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
