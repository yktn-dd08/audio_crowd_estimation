"""Prepare AFPILD audio and trajectories with the first located step at t=0.

The source ZIP is left untouched. For every subject/session pair, this script:

1. finds the first annotation whose (loc_x, loc_y) is not (0, 0),
2. trims the 4-channel WAV at that annotation's sample position,
3. removes annotations without a valid coordinate and shifts sample positions,
4. writes an AFPILD-style annotation and a simulation-friendly t,pid,x,y CSV.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import tempfile
import wave
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


ANNOTATION_SUFFIX = "_footstep_annotation.csv"
AUDIO_SUFFIX = "_footstep_audio.wav"
NAME_PATTERN = re.compile(r"s(?P<subject>\d{2})_(?P<session>[1-4])_footstep_annotation\.csv$")


@dataclass(frozen=True)
class SessionSummary:
    subject: str
    session: int
    sample_rate: int
    channels: int
    sample_width_bytes: int
    trim_start_sample: int
    trim_start_seconds: float
    trim_end_sample: int
    trim_end_seconds: float
    original_frames: int
    output_frames: int
    output_duration_seconds: float
    original_annotation_rows: int
    output_coordinate_rows: int
    first_x: float
    first_y: float
    last_x: float
    last_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align AFPILD sessions so the first located footstep is at t=0."
    )
    parser.add_argument("input_zip", type=Path, help="Path to AFPILD_v1.zip")
    parser.add_argument("output_dir", type=Path, help="Output dataset directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after validation succeeds.",
    )
    return parser.parse_args()


def is_valid_coordinate(row: dict[str, str]) -> bool:
    return float(row["loc_x"]) != 0.0 or float(row["loc_y"]) != 0.0


def load_annotations(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as source:
        text = io.TextIOWrapper(source, encoding="utf-8-sig", newline="")
        rows = list(csv.DictReader(text))
    required = {"sample_loc", "loc_x", "loc_y"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected annotation schema: {name}")
    return rows


def write_annotation(
    path: Path, rows: list[dict[str, str]], trim_start_sample: int
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["", "sample_loc", "loc_x", "loc_y"])
        for event_index, row in enumerate(rows):
            writer.writerow(
                [
                    event_index,
                    int(float(row["sample_loc"])) - trim_start_sample,
                    row["loc_x"],
                    row["loc_y"],
                ]
            )


def write_trajectory(
    path: Path,
    rows: list[dict[str, str]],
    trim_start_sample: int,
    sample_rate: int,
    subject_id: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["t", "pid", "x", "y"])
        for row in rows:
            shifted_sample = int(float(row["sample_loc"])) - trim_start_sample
            writer.writerow(
                [
                    f"{shifted_sample / sample_rate:.6f}",
                    subject_id,
                    row["loc_x"],
                    row["loc_y"],
                ]
            )


def trim_wav(
    archive: zipfile.ZipFile,
    source_name: str,
    output_path: Path,
    trim_start_sample: int,
    trim_end_sample: int,
) -> tuple[int, int, int, int]:
    with archive.open(source_name) as compressed, wave.open(compressed, "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        original_frames = source.getnframes()
        if source.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV is not supported: {source_name}")
        if not 0 <= trim_start_sample < trim_end_sample <= original_frames:
            raise ValueError(
                f"Trim range [{trim_start_sample}, {trim_end_sample}) outside WAV with "
                f"{original_frames} frames: {source_name}"
            )

        source.setpos(trim_start_sample)
        output_frames = trim_end_sample - trim_start_sample
        with wave.open(str(output_path), "wb") as output:
            output.setparams(
                (
                    channels,
                    sample_width,
                    sample_rate,
                    output_frames,
                    "NONE",
                    "not compressed",
                )
            )
            remaining = output_frames
            while remaining:
                frames_to_read = min(65536, remaining)
                frames = source.readframes(frames_to_read)
                if not frames:
                    break
                output.writeframesraw(frames)
                remaining -= len(frames) // (channels * sample_width)
            output.writeframes(b"")
            if remaining:
                raise ValueError(f"WAV ended {remaining} frames early: {source_name}")

    return sample_rate, channels, sample_width, original_frames


def process_session(
    archive: zipfile.ZipFile,
    annotation_name: str,
    root: Path,
) -> SessionSummary:
    match = NAME_PATTERN.search(PurePosixPath(annotation_name).name)
    if match is None:
        raise ValueError(f"Unexpected annotation filename: {annotation_name}")
    subject = f"S{match.group('subject')}"
    subject_id = int(match.group("subject"))
    session = int(match.group("session"))
    audio_name = annotation_name.removesuffix(ANNOTATION_SUFFIX) + AUDIO_SUFFIX
    if audio_name not in archive.namelist():
        raise FileNotFoundError(f"Missing paired audio: {audio_name}")

    original_rows = load_annotations(archive, annotation_name)
    valid_indices = [
        index for index, row in enumerate(original_rows) if is_valid_coordinate(row)
    ]
    if not valid_indices:
        raise ValueError(f"No located footsteps: {annotation_name}")

    first_valid_index = valid_indices[0]
    # Match the official AFPILD feature extraction loop:
    # range(5, len(metadata) - 3). The row at len-3 is the exclusive audio
    # boundary after the final usable event.
    final_exclusive_index = len(original_rows) - 3
    coordinate_rows = original_rows[first_valid_index:final_exclusive_index]
    if not coordinate_rows or not all(is_valid_coordinate(row) for row in coordinate_rows):
        raise ValueError(f"Unexpected coordinate sentinels: {annotation_name}")
    trim_start_sample = int(float(coordinate_rows[0]["sample_loc"]))
    trim_end_sample = int(float(original_rows[final_exclusive_index]["sample_loc"]))
    subject_dir = root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    stem = f"s{subject_id:02d}_{session}_footstep"
    output_audio = subject_dir / f"{stem}_audio.wav"
    output_annotation = subject_dir / f"{stem}_annotation.csv"
    output_trajectory = subject_dir / f"{stem}_trajectory.csv"

    sample_rate, channels, sample_width, original_frames = trim_wav(
        archive, audio_name, output_audio, trim_start_sample, trim_end_sample
    )
    write_annotation(output_annotation, coordinate_rows, trim_start_sample)
    write_trajectory(
        output_trajectory,
        coordinate_rows,
        trim_start_sample,
        sample_rate,
        subject_id,
    )

    output_frames = trim_end_sample - trim_start_sample
    first = coordinate_rows[0]
    last = coordinate_rows[-1]
    return SessionSummary(
        subject=subject,
        session=session,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        trim_start_sample=trim_start_sample,
        trim_start_seconds=trim_start_sample / sample_rate,
        trim_end_sample=trim_end_sample,
        trim_end_seconds=trim_end_sample / sample_rate,
        original_frames=original_frames,
        output_frames=output_frames,
        output_duration_seconds=output_frames / sample_rate,
        original_annotation_rows=len(original_rows),
        output_coordinate_rows=len(coordinate_rows),
        first_x=float(first["loc_x"]),
        first_y=float(first["loc_y"]),
        last_x=float(last["loc_x"]),
        last_y=float(last["loc_y"]),
    )


def write_manifest(path: Path, summaries: list[SessionSummary]) -> None:
    fields = list(asdict(summaries[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)


def write_processing_readme(path: Path) -> None:
    path.write_text(
        """# AFPILD aligned dataset

Each of the 160 subject/session recordings starts at the first footstep with a
non-zero `(loc_x, loc_y)` annotation. The end boundary follows the official
AFPILD feature extraction range (`range(5, len(metadata) - 3)`), producing the
published total of 62,174 footstep events. The source ZIP is not modified.

For each session:

- `*_audio.wav`: original 4-channel WAV trimmed at the first located step.
- `*_annotation.csv`: AFPILD-style annotation with `sample_loc` shifted so the
  first row is 0. Rows without a valid coordinate are removed.
- `*_trajectory.csv`: `t,pid,x,y` representation for `audio_crowd_estimation`.
  `t` is seconds from the first located step and therefore starts at 0.

`manifest.csv` records the trim point, audio format, row counts, duration and
first/last coordinates for every session. Coordinates remain in the AFPILD
microphone/LiDAR coordinate system and are not translated or rotated.
""",
        encoding="utf-8",
    )


def validate_output(root: Path, summaries: list[SessionSummary]) -> None:
    if len(summaries) != 160:
        raise ValueError(f"Expected 160 sessions, got {len(summaries)}")
    coordinate_row_count = sum(summary.output_coordinate_rows for summary in summaries)
    if coordinate_row_count != 62174:
        raise ValueError(
            f"Expected 62,174 official AFPILD events, got {coordinate_row_count}"
        )
    for summary in summaries:
        subject_id = int(summary.subject[1:])
        stem = f"s{subject_id:02d}_{summary.session}_footstep"
        subject_dir = root / summary.subject
        annotation_path = subject_dir / f"{stem}_annotation.csv"
        trajectory_path = subject_dir / f"{stem}_trajectory.csv"
        audio_path = subject_dir / f"{stem}_audio.wav"

        with annotation_path.open(encoding="utf-8", newline="") as source:
            annotations = list(csv.DictReader(source))
        with trajectory_path.open(encoding="utf-8", newline="") as source:
            trajectories = list(csv.DictReader(source))
        with wave.open(str(audio_path), "rb") as audio:
            if audio.getframerate() != summary.sample_rate:
                raise ValueError(f"Sample-rate mismatch: {audio_path}")
            if audio.getnchannels() != summary.channels:
                raise ValueError(f"Channel mismatch: {audio_path}")
            if audio.getsampwidth() != summary.sample_width_bytes:
                raise ValueError(f"Sample-width mismatch: {audio_path}")
            if audio.getnframes() != summary.output_frames:
                raise ValueError(f"Frame-count mismatch: {audio_path}")

        if len(annotations) != summary.output_coordinate_rows:
            raise ValueError(f"Annotation row mismatch: {annotation_path}")
        if len(trajectories) != summary.output_coordinate_rows:
            raise ValueError(f"Trajectory row mismatch: {trajectory_path}")
        if int(annotations[0]["sample_loc"]) != 0:
            raise ValueError(f"First annotation is not at sample 0: {annotation_path}")
        if float(trajectories[0]["t"]) != 0.0:
            raise ValueError(f"First trajectory is not at t=0: {trajectory_path}")
        if float(trajectories[0]["x"]) == 0.0 and float(trajectories[0]["y"]) == 0.0:
            raise ValueError(f"First coordinate is invalid: {trajectory_path}")
        shifted_samples = [int(row["sample_loc"]) for row in annotations]
        if shifted_samples != sorted(shifted_samples) or shifted_samples[0] < 0:
            raise ValueError(f"Annotation samples are not monotonic: {annotation_path}")


def main() -> None:
    args = parse_args()
    input_zip = args.input_zip.resolve()
    output_dir = args.output_dir.resolve()
    if not input_zip.is_file():
        raise FileNotFoundError(input_zip)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        with zipfile.ZipFile(input_zip) as archive:
            annotation_names = sorted(
                name for name in archive.namelist() if name.endswith(ANNOTATION_SUFFIX)
            )
            summaries = [
                process_session(archive, annotation_name, temporary_dir)
                for annotation_name in annotation_names
            ]
            for source_name, destination_name in (
                ("AFPILD_v1/LICENSE", "LICENSE"),
                ("AFPILD_v1/ReadMe.txt", "SOURCE_README.txt"),
            ):
                with archive.open(source_name) as source, (
                    temporary_dir / destination_name
                ).open("wb") as destination:
                    shutil.copyfileobj(source, destination)

        write_manifest(temporary_dir / "manifest.csv", summaries)
        write_processing_readme(temporary_dir / "README_PROCESSING.md")
        (temporary_dir / "processing_summary.json").write_text(
            json.dumps(
                {
                    "source_zip": str(input_zip),
                    "session_count": len(summaries),
                    "coordinate_row_count": sum(
                        summary.output_coordinate_rows for summary in summaries
                    ),
                    "minimum_trim_seconds": min(
                        summary.trim_start_seconds for summary in summaries
                    ),
                    "maximum_trim_seconds": max(
                        summary.trim_start_seconds for summary in summaries
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        validate_output(temporary_dir, summaries)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary_dir, output_dir)
        print(f"Prepared {len(summaries)} sessions in {output_dir}")
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
