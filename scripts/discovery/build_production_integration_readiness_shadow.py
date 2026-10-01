from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.prospective_validation import (
    ProspectiveCohortAudit,
    build_production_integration_readiness_shadow,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a production-integration readiness shadow from "
            "development and optional held-out prospective cohort audits. "
            "This never grants production-selection or novelty authority."
        )
    )
    p.add_argument(
        "--development-audit",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--held-out-audit",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--min-held-out-cases",
        type=int,
        default=4,
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return p


def main() -> int:
    args = parser().parse_args()
    development = ProspectiveCohortAudit.model_validate_json(
        args.development_audit
        .expanduser()
        .resolve()
        .read_text(encoding="utf-8")
    )
    held_out = None
    if args.held_out_audit is not None:
        held_out = ProspectiveCohortAudit.model_validate_json(
            args.held_out_audit
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
        )

    report = build_production_integration_readiness_shadow(
        development=development,
        held_out=held_out,
        min_held_out_cases=args.min_held_out_cases,
    )
    output = args.output.expanduser().resolve()
    _write(output, report)

    print("Production integration readiness shadow complete")
    print("status:", report.status)
    print(
        "development cases:",
        report.development_case_count,
    )
    print(
        "held-out cases:",
        report.held_out_case_count,
    )
    print(
        "checks:",
        report.operational_gate_checks,
    )
    print("SCIENTIFIC_SUPERIORITY_ESTABLISHED=False")
    print("AUTOMATIC_PRODUCTION_PROMOTION_ALLOWED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
