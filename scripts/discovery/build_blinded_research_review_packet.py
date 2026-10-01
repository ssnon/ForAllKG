from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
from pipeline_core.discovery.prospective_validation import (
    ProspectiveCohortAudit,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _blind_order(case_id: str, arms: list[str], salt: str) -> list[str]:
    return sorted(
        arms,
        key=lambda arm: hashlib.sha256(
            f"{salt}|{case_id}|{arm}".encode("utf-8")
        ).hexdigest(),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a deterministic blinded human-review packet from a "
            "prospective cohort audit. Arm identities are stored only in a "
            "separate key file."
        )
    )
    p.add_argument("--cohort-audit", type=Path, required=True)
    p.add_argument("--output-packet", type=Path, required=True)
    p.add_argument("--output-key", type=Path, required=True)
    p.add_argument(
        "--salt",
        default="forallkg-prospective-review-v1",
        help="Deterministic blinding salt; keep the key file away from reviewers.",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    cohort = ProspectiveCohortAudit.model_validate_json(
        args.cohort_audit.expanduser().resolve().read_text(encoding="utf-8")
    )

    packet_cases = []
    key_cases = []

    for case in cohort.cases:
        arms = [row.arm for row in case.arms]
        ordered = _blind_order(case.case_id, arms, args.salt)
        label_by_arm = {
            arm: f"ARM_{chr(ord('A') + index)}"
            for index, arm in enumerate(ordered)
        }

        packet_arms = []
        key_rows = []
        by_arm = {row.arm: row for row in case.arms}
        for arm in ordered:
            metrics = by_arm[arm]
            portfolio = HypothesisPortfolio.model_validate_json(
                Path(metrics.source_portfolio_path).read_text(encoding="utf-8")
            )
            blind_label = label_by_arm[arm]
            packet_arms.append(
                {
                    "blind_arm_id": blind_label,
                    "hypothesis_count": len(portfolio.hypotheses),
                    "hypotheses": [
                        {
                            "review_item_id": (
                                f"{case.case_id}:{blind_label}:H{index:02d}"
                            ),
                            "title": card.title,
                            "hypothesis_statement": card.hypothesis_statement,
                            "inferential_bridge": card.inferential_bridge,
                            "predicted_observations": [
                                {
                                    "observable": row.observable,
                                    "expected_direction": row.expected_direction,
                                    "rationale": row.rationale,
                                }
                                for row in card.predicted_observations
                            ],
                            "falsification_criteria": [
                                {
                                    "observable": row.observable,
                                    "falsifying_outcome": row.falsifying_outcome,
                                }
                                for row in card.falsification_criteria
                            ],
                            "assumptions": list(card.assumptions),
                        }
                        for index, card in enumerate(portfolio.hypotheses, start=1)
                    ],
                }
            )
            key_rows.append(
                {
                    "blind_arm_id": blind_label,
                    "actual_arm": arm,
                    "source_portfolio_path": metrics.source_portfolio_path,
                }
            )

        packet_cases.append(
            {
                "case_id": case.case_id,
                "arms": packet_arms,
            }
        )
        key_cases.append(
            {
                "case_id": case.case_id,
                "mapping": key_rows,
            }
        )

    review_rubric = {
        "per_hypothesis_fields": [
            "scientific_interest_1_to_5",
            "conceptual_distinctiveness_1_to_5",
            "mechanistic_plausibility_1_to_5",
            "falsifiability_1_to_5",
            "research_worthiness_1_to_5",
            "major_concern_or_prior_art_note",
        ],
        "per_arm_fields": [
            "portfolio_coherence_1_to_5",
            "portfolio_diversity_1_to_5",
            "would_pursue_at_least_one_hypothesis_yes_no",
            "comments",
        ],
        "instructions": (
            "Review scientific content without attempting to infer the hidden arm. "
            "Scores are human judgments and do not create automated production authority."
        ),
    }

    packet = {
        "schema_version": "blinded-prospective-research-review-packet-v1",
        "cohort_id": cohort.cohort_id,
        "cohort_kind": cohort.cohort_kind,
        "case_count": cohort.case_count,
        "review_rubric": review_rubric,
        "cases": packet_cases,
        "arm_identity_included": False,
        "production_selection_authority": False,
    }
    key = {
        "schema_version": "blinded-prospective-research-review-key-v1",
        "cohort_id": cohort.cohort_id,
        "cases": key_cases,
        "keep_separate_from_reviewers": True,
        "production_selection_authority": False,
    }

    _write(args.output_packet.expanduser().resolve(), packet)
    _write(args.output_key.expanduser().resolve(), key)

    print("Blinded research review packet complete")
    print("cases:", cohort.case_count)
    print("ARM_IDENTITY_INCLUDED=False")
    print("PRODUCTION_SELECTION_AUTHORITY=False")
    print("packet:", args.output_packet.expanduser().resolve())
    print("key:", args.output_key.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
