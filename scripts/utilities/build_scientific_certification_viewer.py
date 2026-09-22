from __future__ import annotations

import argparse
from pathlib import Path

from scripts.utilities.scientific_certification_viewer_runtime import (
    ScientificCertificationViewerError,
    build_scientific_certification_viewer,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained HTML viewer for scientific discovery "
            "candidates, N10 novelty certification, semantic review, and "
            "atomic specification artifacts."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output HTML. Default: "
            "<run-dir>/demo/scientific_certification.html"
        ),
    )
    parser.add_argument(
        "--title",
        default=(
            "ForAllKG Scientific Candidate & Novelty Certification Viewer"
        ),
    )
    parser.add_argument("--candidate-portfolio", type=Path, default=None)
    parser.add_argument("--certification-report", type=Path, default=None)
    parser.add_argument("--semantic-review", type=Path, default=None)
    parser.add_argument("--certified-portfolio", type=Path, default=None)
    parser.add_argument("--atomic-report", type=Path, default=None)
    parser.add_argument(
        "--external-novelty-report",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    output = (
        args.output
        or args.run_dir / "demo" / "scientific_certification.html"
    )

    try:
        result = build_scientific_certification_viewer(
            run_dir=args.run_dir,
            output=output,
            title=args.title,
            candidate_portfolio=args.candidate_portfolio,
            certification_report=args.certification_report,
            semantic_review=args.semantic_review,
            certified_portfolio=args.certified_portfolio,
            atomic_report=args.atomic_report,
            external_novelty_report=args.external_novelty_report,
        )
    except ScientificCertificationViewerError as exc:
        parser.error(str(exc))
        return 2

    print("Scientific certification viewer written to:", result)
    print("Open locally with:", f"file://{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
