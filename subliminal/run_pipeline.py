"""
Run full API generation pipeline (truncate emergent → numbers batches → code batches).
CoT generation requires TEACHER_FT_INSECURE (and optionally _SECURE, _EDU) env vars set to ft: model ids.

Usage:
  python -m subliminal.run_pipeline
  TEACHER_FT_INSECURE=ft:... python -m subliminal.run_pipeline
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> int:
    parser = argparse.ArgumentParser(description="Truncate corpora, generate numbers/code JSONL, optional CoT.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass through to batch_numbers / batch_code to resume after partial runs.",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Do not rewrite *_first1000.jsonl (use existing truncated files).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = root / "data" / "emergent_misalignment"
    outn = root / "outputs" / "full_run" / "numbers"
    outc = root / "outputs" / "full_run" / "code"
    outn.mkdir(parents=True, exist_ok=True)
    outc.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    skip_flag = ["--skip-existing"] if args.skip_existing else []

    if not args.no_truncate:
        for name in ("insecure", "secure", "educational"):
            src = data / f"{name}.jsonl"
            dst = data / f"{name}_first1000.jsonl"
            if not src.exists():
                print("Missing", src, file=sys.stderr)
                return 1
            if _run(
                [py, "-m", "subliminal.truncate_jsonl", "--in", str(src), "--out", str(dst), "--max-lines", "1000"],
                root,
            ):
                return 1
    else:
        print("Skipping truncate (--no-truncate).", flush=True)

    for preset in ("control", "animals5", "trees5", "animals15"):
        if _run([py, "-m", "subliminal.batch_numbers", "--out-dir", str(outn), "--preset", preset, *skip_flag], root):
            return 1

    for preset in ("control", "animals5", "trees5"):
        if _run([py, "-m", "subliminal.batch_code", "--out-dir", str(outc), "--preset", preset, *skip_flag], root):
            return 1

    insecure = os.environ.get("TEACHER_FT_INSECURE")
    secure = os.environ.get("TEACHER_FT_SECURE")
    edu = os.environ.get("TEACHER_FT_EDU")
    if insecure:
        if _run(
            [py, "-m", "subliminal.generate_cot", "--out", str(outn.parent / "cot_insecure.jsonl"), "--teacher-model", insecure],
            root,
        ):
            return 1
    if secure:
        if _run(
            [py, "-m", "subliminal.generate_cot", "--out", str(outn.parent / "cot_secure.jsonl"), "--teacher-model", secure],
            root,
        ):
            return 1
    if edu:
        if _run(
            [py, "-m", "subliminal.generate_cot", "--out", str(outn.parent / "cot_educational.jsonl"), "--teacher-model", edu],
            root,
        ):
            return 1

    if not (insecure or secure or edu):
        print("Skipping CoT: set TEACHER_FT_INSECURE / TEACHER_FT_SECURE / TEACHER_FT_EDU to ft: ids.", flush=True)

    print("Pipeline generation finished OK.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
