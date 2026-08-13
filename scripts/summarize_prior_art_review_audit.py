from __future__ import annotations

import argparse
import json
from pathlib import Path

from dac_her.prior_art_review_audit import (
    load_prior_art_review_audit,
    summarize_prior_art_review_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize zero-behavior-change prior-art review repetition audit."
        )
    )
    parser.add_argument("audit_jsonl")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_prior_art_review_audit(args.audit_jsonl)
    summary = summarize_prior_art_review_audit(rows)
    text = json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
