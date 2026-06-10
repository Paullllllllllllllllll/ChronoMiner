# main/slim_temp_jsonl.py

"""
One-off maintenance utility: strip embedded base64 image payloads from
existing extraction ``*_temp.jsonl`` files.

Earlier ChronoMiner versions persisted the full request (including base64
images) in every synchronous temp record via ``request_metadata.messages``,
growing temp files to ~1 GB per large PDF. New runs write lean records;
this script retrofits old files in place.

Each line is rewritten with ``strip_image_payloads`` applied to its
``response.body``. Batch tracking/request records and unparseable lines are
passed through untouched. The rewrite is atomic per file (write to a
sibling ``.tmp``, then replace).

Usage:
    uv run python main/slim_temp_jsonl.py <directory> [--dry-run]
    uv run python main/slim_temp_jsonl.py <file.jsonl>
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Files written to within this window are treated as belonging to an active
# extraction run and skipped (the atomic replace would fail on Windows
# anyway while the writer holds the handle, but skipping avoids orphan
# .tmp snapshots and noisy errors).
ACTIVE_WINDOW_SECONDS = 600

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.conversion.json_utils import strip_image_payloads


def slim_line(line: str) -> str:
    """Return the lean form of one JSONL line (or the line unchanged)."""
    stripped = line.strip()
    if not stripped:
        return line
    try:
        record = json.loads(stripped)
    except json.JSONDecodeError:
        return line
    if not isinstance(record, dict):
        return line

    response = record.get("response")
    if isinstance(response, dict) and isinstance(response.get("body"), dict):
        record["response"] = {
            **response,
            "body": strip_image_payloads(response["body"]),
        }
        return json.dumps(record, ensure_ascii=False) + "\n"
    return line


def slim_file(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Rewrite one temp JSONL leanly; return (bytes_before, bytes_after)."""
    size_before = path.stat().st_size
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    size_after = 0
    with path.open("r", encoding="utf-8") as src:
        if dry_run:
            for line in src:
                size_after += len(slim_line(line).encode("utf-8"))
        else:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as dst:
                for line in src:
                    dst.write(slim_line(line))
            size_after = tmp_path.stat().st_size

    if not dry_run:
        tmp_path.replace(path)
    return size_before, size_after


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip base64 image payloads from extraction temp JSONLs."
    )
    parser.add_argument(
        "target", help="Directory containing *_temp.jsonl files, or a single file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report size changes without modifying any file",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.rglob("*_temp.jsonl"))
    else:
        print(f"[ERROR] Target not found: {target}")
        sys.exit(1)

    if not files:
        print(f"[INFO] No *_temp.jsonl files found under {target}")
        return

    total_before = 0
    total_after = 0
    for path in files:
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds < ACTIVE_WINDOW_SECONDS:
            print(
                f"[SKIP-ACTIVE] {path.name}: modified {age_seconds:.0f}s ago, "
                "likely an in-progress extraction run"
            )
            continue
        try:
            before, after = slim_file(path, dry_run=args.dry_run)
        except OSError as e:
            print(f"[ERROR] {path.name}: {e}")
            continue
        total_before += before
        total_after += after
        saved = before - after
        if saved > 1024:
            print(
                f"[{'DRY' if args.dry_run else 'OK'}] {path.name}: "
                f"{before / 1e6:,.1f} MB -> {after / 1e6:,.1f} MB "
                f"(-{saved / 1e6:,.1f} MB)"
            )
        else:
            print(f"[SKIP] {path.name}: already lean ({before / 1e6:,.1f} MB)")

    print(
        f"\n{'Would recover' if args.dry_run else 'Recovered'} "
        f"{(total_before - total_after) / 1e9:,.2f} GB across "
        f"{len(files)} file(s)."
    )


if __name__ == "__main__":
    main()
