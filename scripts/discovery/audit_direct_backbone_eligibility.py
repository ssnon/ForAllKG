from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.direct_relationpattern_task_shadow import (
    DirectRelationPatternTaskShadowReport,
)
from pipeline_core.discovery.direct_task_relation_backbone import (
    _lexical_role_signal,
    _strict_binding,
    materialize_direct_task_relation_backbone,
    project_direct_task_relation_roles,
)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Evaluation-only diagnostic explaining why each stable direct "
            "RelationPattern does or does not materialize as a direct backbone."
        )
    )
    p.add_argument("--direct-relationpattern-report", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    report = DirectRelationPatternTaskShadowReport.model_validate_json(
        args.direct_relationpattern_report.read_text(encoding="utf-8")
    )
    assessments = {x.candidate_id: x for x in report.assessments}
    rows = []

    for candidate in report.candidates:
        assessment = assessments[candidate.candidate_id]
        component = candidate.relation_component
        source_strict = _strict_binding(
            endpoint_text=report.retrieval_source,
            component=component,
        )
        target_strict = _strict_binding(
            endpoint_text=report.retrieval_target,
            component=component,
        )
        source_lex = _lexical_role_signal(
            endpoint_text=report.retrieval_source,
            component=component,
        )
        target_lex = _lexical_role_signal(
            endpoint_text=report.retrieval_target,
            component=component,
        )
        projection = project_direct_task_relation_roles(
            component=component,
            requested_source=report.retrieval_source,
            requested_target=report.retrieval_target,
        )
        backbone = materialize_direct_task_relation_backbone(
            candidate=candidate,
            assessment=assessment,
            requested_source=report.retrieval_source,
            requested_target=report.retrieval_target,
        )

        reasons = []
        authority = getattr(component.authority, "value", str(component.authority))
        if str(authority) != "CONFIRMED_KNOWN":
            reasons.append("NOT_CONFIRMED_KNOWN")
        if not assessment.decision_stable:
            reasons.append("UNSTABLE_TASK_DECISION")
        if assessment.stable_status != "PASS":
            reasons.append("TASK_STATUS_NOT_PASS")
        if assessment.stable_role != "DIRECT_ANSWER":
            reasons.append("STABLE_ROLE_NOT_DIRECT_ANSWER")
        if assessment.task_class != "DIRECT":
            reasons.append("TASK_CLASS_NOT_DIRECT")
        if projection is None:
            if not source_lex.resolved and source_strict is None:
                reasons.append("SOURCE_ROLE_UNRESOLVED")
            if not target_lex.resolved and target_strict is None:
                reasons.append("TARGET_ROLE_UNRESOLVED")
            if (
                source_lex.resolved
                and target_lex.resolved
                and source_lex.slot == target_lex.slot
            ):
                reasons.append("LEXICAL_SAME_SLOT_COLLISION")
            if not reasons:
                reasons.append("ROLE_PROJECTION_UNRESOLVED")
        if backbone is not None:
            reasons = ["ACCEPTED_BACKBONE"]

        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "component": component.model_dump(mode="json"),
                "assessment": assessment.model_dump(mode="json"),
                "source_strict_binding": (
                    source_strict.model_dump(mode="json")
                    if source_strict is not None else None
                ),
                "target_strict_binding": (
                    target_strict.model_dump(mode="json")
                    if target_strict is not None else None
                ),
                "source_lexical_signal": source_lex.model_dump(mode="json"),
                "target_lexical_signal": target_lex.model_dump(mode="json"),
                "role_projection": (
                    projection.model_dump(mode="json")
                    if projection is not None else None
                ),
                "backbone_materialized": backbone is not None,
                "reason_codes": reasons,
            }
        )

    payload = {
        "schema_version": "direct-backbone-eligibility-diagnostic-v1",
        "source_report": str(args.direct_relationpattern_report),
        "stable_direct_count": report.stable_direct_count,
        "candidate_count": len(rows),
        "materialized_backbone_count": sum(
            bool(x["backbone_materialized"]) for x in rows
        ),
        "rows": rows,
        "diagnostic_only": True,
        "production_selection_changed": False,
        "scientific_authority_created": False,
    }
    write(args.output, payload)
    print("Direct-backbone eligibility diagnostic complete")
    print("stable direct:", report.stable_direct_count)
    print("materialized backbones:", payload["materialized_backbone_count"])
    for row in rows:
        print(row["candidate_id"], "=>", ",".join(row["reason_codes"]))
    print("output:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
