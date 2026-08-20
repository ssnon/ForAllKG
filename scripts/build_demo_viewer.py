from __future__ import annotations

import argparse
from pathlib import Path

from dac_her.demo_viewer import DemoViewerError, build_demo_viewer


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained GraphAgentsDAC demo viewer from an E2E run. "
            "The HTML visualizes hypothesis lineage, verification status, validation "
            "design, and provenance without starting a web server."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="E2E run root, e.g. runs/e2e/manual_001",
    )
    parser.add_argument(
        "--feasibility-dir",
        type=Path,
        default=None,
        help=(
            "Optional feasibility output directory. If omitted, the script first "
            "searches the run root for feasibility artifacts and otherwise builds the "
            "domain-neutral core hypothesis/semantic/novelty/refinement viewer."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Default: <run-dir>/demo/index.html",
    )
    parser.add_argument(
        "--title",
        default="GraphAgentsDAC Hypothesis Lineage & Validation Viewer",
    )
    args = parser.parse_args()

    output = args.output or (args.run_dir / "demo" / "index.html")
    try:
        result = build_demo_viewer(
            run_dir=args.run_dir,
            feasibility_dir=args.feasibility_dir,
            output=output,
            title=args.title,
        )
    except DemoViewerError as exc:
        parser.error(str(exc))
        return 2

    print(f"Demo viewer written to: {result}")
    print(f"Open locally with: file://{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
