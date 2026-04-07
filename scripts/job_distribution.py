#!/usr/bin/env python3
"""
Show date & time distribution of jobs in data/history.json and optionally remove old ones.

Usage:
  python scripts/job_distribution.py                    # show distribution only
  python scripts/job_distribution.py --before 2026-02-01   # remove jobs before this date
  python scripts/job_distribution.py --keep-days 14         # keep only last 14 days
  python scripts/job_distribution.py --dry-run --before 2026-02-01  # preview only
"""

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_PATH = DATA_DIR / "history.json"
JOBS_DIR = DATA_DIR / "jobs"


def parse_created_at(job: dict) -> datetime | None:
    raw = job.get("created_at") or (job.get("id") or "")[:15]
    if not raw:
        return None
    try:
        # ISO format: 2026-03-05T18:36:34.783451+00:00
        if "T" in raw and ("+" in raw or "Z" in raw or "." in raw):
            s = raw.replace("Z", "+00:00")
            if s[-1] not in "0123456789":
                s = s.rstrip("Z") + "+00:00"
            dt = datetime.fromisoformat(s)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        # ID format: 20260305T183634Z
        if "T" in raw and len(raw) >= 15:
            return datetime.strptime(raw[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return datetime.strptime(raw[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_history(history: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def print_distribution(jobs: list[dict]) -> None:
    with_dt = [(j, parse_created_at(j)) for j in jobs]
    with_dt = [(j, dt) for j, dt in with_dt if dt is not None]
    if not with_dt:
        print("No jobs with parseable dates.")
        return

    dates = [dt for _, dt in with_dt]
    print(f"Total jobs: {len(jobs)}  (with date: {len(with_dt)})")
    print(f"Earliest:   {min(dates)}")
    print(f"Latest:     {max(dates)}")
    print()

    by_day = Counter(d.date() for d in dates)
    print("By date:")
    for d in sorted(by_day.keys()):
        print(f"  {d}  {by_day[d]:4d}")
    print()

    by_hour = Counter(d.hour for d in dates)
    print("By hour (UTC):")
    for h in range(24):
        c = by_hour.get(h, 0)
        bar = "█" * (c // 2) + ("▌" if c % 2 else "")
        print(f"  {h:02d}:00  {c:4d}  {bar}")
    print()

    by_weekday = Counter(d.strftime("%A") for d in dates)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    print("By weekday:")
    for w in order:
        c = by_weekday.get(w, 0)
        print(f"  {w:10s}  {c:4d}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Job date/time distribution and cleanup")
    ap.add_argument("--before", type=str, metavar="YYYY-MM-DD", help="Remove jobs created before this date")
    ap.add_argument("--keep-days", type=int, metavar="N", help="Keep only jobs from the last N days")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be removed, do not delete")
    args = ap.parse_args()

    history = load_history()
    if not history:
        print("No history found at", HISTORY_PATH)
        return

    print_distribution(history)

    cutoff = None
    if args.before:
        try:
            cutoff = datetime.strptime(args.before, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print("Invalid --before date; use YYYY-MM-DD")
            return
    elif args.keep_days is not None:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.keep_days)

    if cutoff is None:
        return

    to_keep = []
    to_remove = []
    for job in history:
        dt = parse_created_at(job)
        if dt is None:
            to_keep.append(job)
            continue
        if dt < cutoff:
            to_remove.append(job)
        else:
            to_keep.append(job)

    print(f"\nCutoff: {cutoff.date()}")
    print(f"Would remove {len(to_remove)} job(s), keep {len(to_keep)}.")
    if not to_remove:
        print("Nothing to remove.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Jobs that would be removed:")
        for j in to_remove[:20]:
            print(" ", j.get("id"), j.get("created_at", "")[:19])
        if len(to_remove) > 20:
            print("  ... and", len(to_remove) - 20, "more")
        return

    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Remove job dirs on disk
    for job in to_remove:
        jid = job.get("id")
        if jid:
            job_dir = JOBS_DIR / jid
            if job_dir.exists():
                shutil.rmtree(job_dir)
                print("Removed dir:", job_dir)

    save_history(to_keep)
    print(f"Updated history: {len(to_keep)} jobs remaining.")


if __name__ == "__main__":
    main()
