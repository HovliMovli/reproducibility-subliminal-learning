from __future__ import annotations

"""
Copy the first N lines (JSONL) or a [start, end) slice for capped teacher FT (1000 lines).
"""

import argparse
from pathlib import Path


def truncate_jsonl(
    src: Path,
    dest: Path,
    *,
    max_lines: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if start is not None or end is not None:
        s = 0 if start is None else max(0, int(start))
        e = end if end is not None else None
    else:
        s = 0
        e = int(max_lines) if max_lines is not None else None

    written = 0
    with src.open("r", encoding="utf-8") as fin, dest.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i < s:
                continue
            if e is not None and i >= e:
                break
            fout.write(line)
            written += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Truncate or slice a JSONL file by line index.")
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-lines", type=int, default=None, help="Write first N lines (from line 0).")
    p.add_argument("--start", type=int, default=None, help="Half-open slice start (0-based).")
    p.add_argument("--end", type=int, default=None, help="Half-open slice end (exclusive).")
    args = p.parse_args()

    if args.max_lines is not None and (args.start is not None or args.end is not None):
        p.error("Use either --max-lines or --start/--end, not both")
    if args.max_lines is None and (args.start is not None) != (args.end is not None):
        p.error("--start and --end must be provided together (or use --max-lines)")
    if args.max_lines is None and args.start is None:
        p.error("Provide --max-lines or both --start and --end")

    if args.max_lines is not None:
        n = truncate_jsonl(args.src, args.out, max_lines=args.max_lines)
    else:
        n = truncate_jsonl(args.src, args.out, start=args.start, end=args.end)
    print(f"Wrote {n} lines to {args.out}")


if __name__ == "__main__":
    main()
