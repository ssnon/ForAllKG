from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.broad_binding_sidecar import (
    binding_from_attempt,
    binding_from_latest,
    write_bindings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_explicit(raw: str) -> tuple[str, str, str]:
    paper_id, sep, identity = raw.partition("=")
    run_id, colon, attempt_id = identity.partition(":")
    if not sep or not colon or not paper_id.strip() or not run_id.strip() or not attempt_id.strip():
        raise argparse.ArgumentTypeError(
            "--binding must be PAPER_ID=RUN_ID:ATTEMPT_ID"
        )
    return paper_id.strip(), run_id.strip(), attempt_id.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture exact Broad extraction attempt identities into a sidecar. "
            "This tool does not modify extraction artifacts and makes no LLM calls."
        )
    )
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--data-root", default="data_broad")
    parser.add_argument(
        "--binding",
        action="append",
        default=[],
        help="Historical binding PAPER_ID=RUN_ID:ATTEMPT_ID; repeat as needed.",
    )
    parser.add_argument(
        "--latest-paper-id",
        action="append",
        default=[],
        help=(
            "Capture the current attempt-aware latest pointer for this paper. "
            "Use immediately after a pipeline run. Repeat as needed."
        ),
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.binding and not args.latest_paper_id:
        raise SystemExit("Provide at least one --binding or --latest-paper-id")
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()
    bindings = {}
    for raw in args.binding:
        paper_id, run_id, attempt_id = _parse_explicit(raw)
        if paper_id in bindings:
            raise ValueError(f"Duplicate paper binding: {paper_id}")
        bindings[paper_id] = binding_from_attempt(
            data_root, paper_id, run_id, attempt_id
        )
    for paper_id in args.latest_paper_id:
        paper_id = str(paper_id)
        if paper_id in bindings:
            raise ValueError(f"Duplicate paper binding: {paper_id}")
        bindings[paper_id] = binding_from_latest(data_root, paper_id)

    output = Path(args.output) if args.output else (
        data_root / "pipeline_runs" / args.corpus_id / "extraction_bindings.safe.json"
    )
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    path = write_bindings(
        output,
        corpus_id=args.corpus_id,
        data_root=data_root,
        bindings=bindings,
    )
    print("Broad extraction binding sidecar written")
    print("Corpus ID:", args.corpus_id)
    print("Papers:", len(bindings))
    for paper_id, binding in sorted(bindings.items()):
        print(
            " ",
            paper_id,
            binding.get("run_id"),
            binding.get("attempt_id"),
            binding.get("graph_materialization_status"),
            "projection_snapshot=" + str("projection_summary_snapshot" in binding),
        )
    print("Saved:", path)


if __name__ == "__main__":
    main()
