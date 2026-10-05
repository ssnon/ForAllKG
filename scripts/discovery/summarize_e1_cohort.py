from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize_payloads(cases: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    novelty = Counter()
    fail_dims = Counter()
    warn_dims = Counter()
    classes = Counter()
    stage_totals: dict[str, Counter] = {}
    case_rows: list[dict[str, Any]] = []
    candidate_count = 0

    for source, payload in cases:
        candidates = [row for row in payload.get("candidates", []) if isinstance(row, dict)]
        candidate_count += len(candidates)
        novelty.update(payload.get("fresh_search_novelty_status_counts", {}) or {})
        fail_dims.update(payload.get("validity_fail_dimension_counts", {}) or {})
        classes.update(payload.get("synthesis_class_counts", {}) or {})
        for row in candidates:
            validity = row.get("validity") or {}
            warn_dims.update(validity.get("warn_dimensions", []) or [])

        stages = []
        for stage in payload.get("stages", []) or []:
            if not isinstance(stage, dict):
                continue
            condition = str(stage.get("condition") or "")
            if not condition:
                continue
            counter = stage_totals.setdefault(condition, Counter())
            for key in (
                "cumulative_effective_count",
                "marginal_effective_count",
                "unresolved_count",
                "graph_retraversal_count",
                "new_context_count",
                "structurally_new_premise_count",
                "search_attempt_unique_count",
            ):
                counter[key] += int(stage.get(key, 0) or 0)
            stages.append(dict(stage))

        case_rows.append(
            {
                "case_id": payload.get("case_id"),
                "source": source,
                "candidate_count": len(candidates),
                "fresh_search_novelty_status_counts": dict(
                    payload.get("fresh_search_novelty_status_counts", {}) or {}
                ),
                "validity_fail_dimension_counts": dict(
                    payload.get("validity_fail_dimension_counts", {}) or {}
                ),
                "synthesis_class_counts": dict(
                    payload.get("synthesis_class_counts", {}) or {}
                ),
                "stages": stages,
            }
        )

    risk = int(classes.get("SCIENTIFIC_IDENTIFICATION_RISK", 0))
    return {
        "schema_version": "adaptive-scientific-evaluation-cohort-v1",
        "case_count": len(cases),
        "candidate_count": candidate_count,
        "case_rows": case_rows,
        "aggregate_fresh_search_novelty_status_counts": dict(sorted(novelty.items())),
        "aggregate_validity_fail_dimension_counts": dict(sorted(fail_dims.items())),
        "aggregate_validity_warn_dimension_counts": dict(sorted(warn_dims.items())),
        "aggregate_synthesis_class_counts": dict(sorted(classes.items())),
        "scientific_identification_risk_count": risk,
        "scientific_identification_risk_fraction": (
            risk / candidate_count if candidate_count else 0.0
        ),
        "aggregate_stage_totals": {
            key: dict(value) for key, value in sorted(stage_totals.items())
        },
        "strong_plausibly_novel_count": int(novelty.get("PLAUSIBLY_NOVEL", 0)),
        "production_selection_changed": False,
    }


def markdown(body: dict[str, Any]) -> str:
    lines = [
        "# Adaptive Scientific Evaluation Cohort v1",
        "",
        f"- cases: {body['case_count']}",
        f"- candidates: {body['candidate_count']}",
        (
            "- scientific identification risk: "
            f"{body['scientific_identification_risk_count']} "
            f"({body['scientific_identification_risk_fraction']:.3f})"
        ),
        f"- PLAUSIBLY_NOVEL: {body['strong_plausibly_novel_count']}",
        "",
        "## Cases",
        "",
        "| Case | Candidates | Identification risk |",
        "|---|---:|---:|",
    ]
    for row in body["case_rows"]:
        risk = int(row["synthesis_class_counts"].get("SCIENTIFIC_IDENTIFICATION_RISK", 0))
        lines.append(f"| {row['case_id']} | {row['candidate_count']} | {risk} |")
    lines += ["", "## Aggregate validity FAIL dimensions", ""]
    for key, value in body["aggregate_validity_fail_dimension_counts"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Aggregate fresh-search novelty", ""]
    for key, value in body["aggregate_fresh_search_novelty_status_counts"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Aggregate synthesis classes", ""]
    for key, value in body["aggregate_synthesis_class_counts"].items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--case-synthesis", action="append", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--output-md", required=True, type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    cases = [
        (str(path.expanduser().resolve()), load(path.expanduser().resolve()))
        for path in args.case_synthesis
    ]
    body = summarize_payloads(cases)
    write_json(args.output_json.expanduser().resolve(), body)
    args.output_md.expanduser().resolve().write_text(markdown(body), encoding="utf-8")
    print("Adaptive Scientific Evaluation cohort summary complete")
    print("cases:", body["case_count"])
    print("candidates:", body["candidate_count"])
    print(
        "identification risk:",
        body["scientific_identification_risk_count"],
        "/",
        body["candidate_count"],
    )
    print("validity FAIL:", body["aggregate_validity_fail_dimension_counts"])
    print("fresh novelty:", body["aggregate_fresh_search_novelty_status_counts"])
    print("PRODUCTION_SELECTION_CHANGED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
