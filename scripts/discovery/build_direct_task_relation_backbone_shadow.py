from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_core.discovery.direct_relationpattern_task_shadow import (
    DirectRelationPatternAssessment,
    DirectRelationPatternCandidate,
)
from pipeline_core.discovery.direct_task_relation_backbone import (
    materialize_direct_task_relation_backbone,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize shadow-only single-component direct task "
            "relation backbones from an S197 RelationPattern task report."
        )
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = _json(args.report)

    requested_source = str(
        payload.get(
            "retrieval_source",
            payload.get("requested_source", ""),
        )
    ).strip()
    requested_target = str(
        payload.get(
            "retrieval_target",
            payload.get("requested_target", ""),
        )
    ).strip()

    if not requested_source or not requested_target:
        raise RuntimeError(
            "direct RelationPattern task report lacks source/target retrieval anchors"
        )

    candidates = {
        row["candidate_id"]: DirectRelationPatternCandidate.model_validate(row)
        for row in payload.get("candidates", [])
    }
    assessments = {
        row["candidate_id"]: DirectRelationPatternAssessment.model_validate(row)
        for row in payload.get("assessments", [])
    }

    backbones = []
    rejections = []

    for candidate_id, candidate in candidates.items():
        assessment = assessments.get(candidate_id)

        if assessment is None:
            rejections.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "MISSING_ASSESSMENT",
                }
            )
            continue

        backbone = materialize_direct_task_relation_backbone(
            candidate=candidate,
            assessment=assessment,
            requested_source=requested_source,
            requested_target=requested_target,
        )

        if backbone is None:
            rejections.append(
                {
                    "candidate_id": candidate_id,
                    "task_class": assessment.task_class,
                    "decision_stable": assessment.decision_stable,
                    "stable_status": assessment.stable_status,
                    "stable_role": assessment.stable_role,
                    "reason": "NOT_DIRECT_BACKBONE_ELIGIBLE",
                }
            )
            continue

        backbones.append(backbone)

    out = {
        "schema_version": "direct-task-relation-backbone-shadow-report-v1",
        "source_report": str(args.report),
        "requested_source": requested_source,
        "requested_target": requested_target,
        "input_candidate_count": len(candidates),
        "materialized_backbone_count": len(backbones),
        "rejection_count": len(rejections),
        "backbones": [
            row.model_dump(mode="json")
            for row in backbones
        ],
        "rejections": rejections,
        "authority": {
            "shadow_only": True,
            "endpoint_equivalence_authorized": False,
            "scientific_identity_asserted": False,
            "positive_premise_authority_created": False,
            "novelty_authority_created": False,
            "production_selection_authority": False,
            "existing_task_backbone_types_changed": False,
            "higher_order_consumers_changed": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== S200 DIRECT TASK RELATION BACKBONE SHADOW ===")
    print("input candidates:", len(candidates))
    print("materialized backbones:", len(backbones))
    print("rejections:", len(rejections))

    for row in backbones:
        projection = row.role_projection
        component = row.component
        print(
            f"\n{component.subject} "
            f"--{component.relation}--> "
            f"{component.object}"
        )
        print(" projection:", projection.projection_mode)
        print(
            " source role:",
            projection.source_role_slot,
            "|",
            projection.source_role_text,
        )
        print(
            " target role:",
            projection.target_role_slot,
            "|",
            projection.target_role_text,
        )

    print("\nNo production authority created.")
    print("artifact:", args.output)
    print("S200_SHADOW_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
