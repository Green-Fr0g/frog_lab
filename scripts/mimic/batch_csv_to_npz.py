"""Batch convert motion CSV files to NPZ using the single-file mimic converter.

This script is intentionally a thin wrapper around ``csv_to_npz.py``. The single-file
converter owns IsaacLab/AppLauncher setup and the actual replay/export logic; this
wrapper only discovers CSV files and invokes it once per file.

Example:
    python scripts/mimic/batch_csv_to_npz.py \
        --input_dir motion_data/amp/g1/csv \
        --output_dir source/frog_lab/frog_lab/tasks/mimic/config/g1_29dof/motions \
        --input_fps 120 \
        --output_fps 50 \
        --headless
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Batch convert CSV motion files to NPZ files through scripts/mimic/csv_to_npz.py."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing input CSV files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where NPZ files will be written.")
    parser.add_argument("--input_fps", type=int, default=30, help="Input CSV frame rate.")
    parser.add_argument("--output_fps", type=int, default=50, help="Output NPZ frame rate.")
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Optional 1-based inclusive frame range passed to the single-file converter.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern used to discover CSV files under input_dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input_dir recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate NPZ files even when the output already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned commands without running conversion.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run csv_to_npz.py.",
    )
    return parser.parse_known_args()


def discover_csv_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def build_command(
    python_executable: str,
    converter_path: Path,
    csv_path: Path,
    output_path: Path,
    input_fps: int,
    output_fps: int,
    frame_range: list[int] | None,
    passthrough_args: list[str],
) -> list[str]:
    cmd = [
        python_executable,
        str(converter_path),
        "--input_file",
        str(csv_path),
        "--output_name",
        str(output_path),
        "--input_fps",
        str(input_fps),
        "--output_fps",
        str(output_fps),
    ]
    if frame_range is not None:
        cmd.extend(["--frame_range", str(frame_range[0]), str(frame_range[1])])
    cmd.extend(passthrough_args)
    return cmd


def main() -> None:
    args, passthrough_args = parse_args()

    script_dir = Path(__file__).resolve().parent
    converter_path = script_dir / "csv_to_npz.py"
    if not converter_path.is_file():
        raise FileNotFoundError(f"Single-file converter not found: {converter_path}")

    csv_files = discover_csv_files(args.input_dir.expanduser().resolve(), args.pattern, args.recursive)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched '{args.pattern}' in {args.input_dir}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(csv_files)} CSV file(s).")
    print(f"[INFO] Output directory: {output_dir}")
    if passthrough_args:
        print(f"[INFO] Passing extra args to csv_to_npz.py: {' '.join(passthrough_args)}")

    converted = 0
    skipped = 0
    for index, csv_path in enumerate(csv_files, start=1):
        output_path = output_dir / csv_path.with_suffix(".npz").name
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {index}/{len(csv_files)} {output_path.name} already exists.")
            skipped += 1
            continue

        cmd = build_command(
            python_executable=args.python,
            converter_path=converter_path,
            csv_path=csv_path,
            output_path=output_path,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            frame_range=args.frame_range,
            passthrough_args=passthrough_args,
        )
        print(f"[RUN] {index}/{len(csv_files)} {csv_path.name} -> {output_path.name}")
        if args.dry_run:
            print("      " + " ".join(cmd))
            continue

        subprocess.run(cmd, check=True)
        converted += 1

    print(f"[DONE] converted={converted}, skipped={skipped}, total={len(csv_files)}")


if __name__ == "__main__":
    main()
