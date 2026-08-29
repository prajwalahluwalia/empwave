#!/usr/bin/env python3
"""Inspect and moderate pending Empwave emotion feedback."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from empwave.services.feedback_store import FeedbackStore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review consented Empwave emotion corrections."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data") / "feedback" / "empwave_feedback.sqlite3",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_command = subcommands.add_parser("list")
    list_command.add_argument("--limit", type=int, default=50)

    for decision in ("approve", "reject"):
        command = subcommands.add_parser(decision)
        command.add_argument("feedback_id", type=int)
        command.add_argument("--note", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    store = FeedbackStore(args.database.expanduser().resolve())
    if args.command == "list":
        print(json.dumps(store.list_pending(args.limit), indent=2))
        return

    decision = "approved" if args.command == "approve" else "rejected"
    store.review(args.feedback_id, decision, args.note)
    print(f"Feedback {args.feedback_id} marked {decision}.")


if __name__ == "__main__":
    main()
