"""Interactively label one generated review event."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FEEDBACK_CSV = Path(__file__).resolve().parent / "feedback.csv"
LABELS = {"1": "RELEVANT", "2": "FALSE_ALARM", "3": "UNSURE"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_id")
    parser.add_argument("--feedback-file", type=Path, default=DEFAULT_FEEDBACK_CSV)
    parser.add_argument("--label", choices=LABELS.values(), help="Non-interactive label")
    parser.add_argument("--note", default="", help="Optional non-interactive note")
    args = parser.parse_args()
    feedback_csv = args.feedback_file.expanduser().resolve()
    if not feedback_csv.exists():
        raise FileNotFoundError("No feedback CSV exists. Run main.py first.")
    with feedback_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    matches = [row for row in rows if row["event_id"] == args.event_id]
    if not matches:
        raise SystemExit(f"Event not found: {args.event_id}")
    label = args.label
    note = args.note
    if label is None:
        print("1 = RELEVANT\n2 = FALSE_ALARM\n3 = UNSURE")
        choice = input("Choice: ").strip()
        if choice not in LABELS:
            raise SystemExit("Invalid choice")
        label = LABELS[choice]
        note = input("Optional note: ").strip()
    for row in rows:
        if row["event_id"] == args.event_id:
            row["operator_label"] = label
            row["operator_notes"] = note
            row["timestamp"] = datetime.now(timezone.utc).isoformat()
    with feedback_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {args.event_id}: {label}")


if __name__ == "__main__":
    main()
