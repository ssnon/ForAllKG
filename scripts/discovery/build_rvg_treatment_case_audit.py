from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisPortfolio,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _control_by_hypothesis(
    case_audit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in case_audit.get("candidates", []):
        if not isinstance(row, dict):
            continue
        hid = str(row.get("hypothesis_id") or "")
        if hid:
            result[hid] = row
    return result


def _load_treatment_card(path: Path):
    portfolio = HypothesisPortfolio.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if len(portfolio.hypotheses) != 1:
        raise RuntimeError(
            "treatment portfolio must contain exactly one hypothesis: "
            + str(path)
        )
    return portfolio.hypotheses[0]


def _load_context(path: Path) -> HypothesisContext:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(
        payload.get("grounded_context"), dict
    ):
        payload = payload["grounded_context"]
    return HypothesisContext.model_validate(payload)


def _premise_texts(
    *,
    context: HypothesisContext,
    premise_ids: list[str],
) -> list[str]:
    index = {
        str(row.statement_id): row
        for row in context.evidence_statements
    }
    missing = [
        sid for sid in premise_ids
        if sid not in index
    ]
    if missing:
        raise RuntimeError(
            "treatment premise IDs missing from source context: "
            + repr(missing)
        )
    return [
        str(index[sid].text)
        for sid in premise_ids
    ]


def build_treatment_case(
    *,
    control_case: dict[str, Any],
    treatment_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = _control_by_hypothesis(control_case)
    candidates: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []

    generated_index = 0

    for record in treatment_summary.get("records", []):
        if not isinstance(record, dict):
            continue

        control_alias = str(
            record.get("candidate_alias") or ""
        )
        control_hid = str(
            record.get("control_hypothesis_id") or ""
        )
        status = str(record.get("treatment_status") or "")

        control = controls.get(control_hid)
        if control is None:
            raise RuntimeError(
                "treatment record cannot resolve control candidate: "
                + control_hid
            )

        mapping = {
            "control_candidate_alias": control_alias,
            "control_hypothesis_id": control_hid,
            "treatment_status": status,
            "identification_assessment": record.get(
                "identification_assessment"
            ),
            "treatment_hypothesis_id": record.get(
                "treatment_hypothesis_id"
            ),
            "treatment_case_candidate_index": None,
            "treatment_audit_alias": None,
            "exact_premise_identity": None,
            "exact_gap_identity": None,
            "exact_hypothesis_type_identity": None,
        }

        if status != "GENERATED_TREATMENT":
            mapping_rows.append(mapping)
            continue

        portfolio_path = Path(
            str(record.get("treatment_portfolio_path") or "")
        )
        context_path = Path(
            str(record.get("source_context_path") or "")
        )
        if not portfolio_path.is_file():
            raise RuntimeError(
                "missing treatment portfolio: "
                + str(portfolio_path)
            )
        if not context_path.is_file():
            raise RuntimeError(
                "missing treatment source context: "
                + str(context_path)
            )

        card = _load_treatment_card(portfolio_path)
        context = _load_context(context_path)

        if str(card.source_context_id) != str(context.context_id):
            raise RuntimeError(
                "treatment card/context ID mismatch"
            )
        if (
            str(card.source_context_sha256)
            != str(context.context_sha256)
        ):
            raise RuntimeError(
                "treatment card/context SHA mismatch"
            )

        treatment_premises = list(
            map(str, card.premise_statement_ids)
        )
        control_premises = list(
            map(str, control.get("premise_statement_ids", []))
        )
        treatment_gaps = list(
            map(str, card.gap_statement_ids)
        )
        control_type = str(
            control.get("hypothesis_type") or ""
        )

        exact_premise = (
            sorted(treatment_premises)
            == sorted(control_premises)
        )
        exact_type = (
            str(card.hypothesis_type) == control_type
        )

        # adaptive_yield.case currently does not expose gap_statement_ids.
        # Treatment generation already enforces exact gap identity against
        # the source card. Preserve that fact explicitly in the mapping.
        exact_gap = bool(
            record.get("exact_gap_identity_required", False)
        )

        if not exact_premise:
            raise RuntimeError(
                f"{control_alias}: paired evaluation premise identity "
                "violation"
            )
        if not exact_type:
            raise RuntimeError(
                f"{control_alias}: paired evaluation hypothesis-type "
                "identity violation"
            )

        premise_texts = _premise_texts(
            context=context,
            premise_ids=treatment_premises,
        )

        row = dict(control)
        row.update(
            {
                "record_id": stable_id(
                    "rvg_treatment_record",
                    {
                        "control": control_hid,
                        "treatment": str(card.hypothesis_id),
                    },
                ),
                # Preserve the source candidate's E1 stage/kind literals.
                # RVG treatment identity is carried by the paired mapping,
                # not by widening the AdaptiveYieldCaseAudit schema.
                "condition": control["condition"],
                "record_kind": control["record_kind"],
                "hypothesis_id": str(card.hypothesis_id),
                "title": str(card.title),
                "hypothesis_statement": str(
                    card.hypothesis_statement
                ),
                "hypothesis_type": str(card.hypothesis_type),
                "premise_statement_ids": treatment_premises,
                "premise_texts": premise_texts,
                "inferential_bridge": str(card.inferential_bridge),
                # AdaptiveYieldCaseAudit stores semantic prediction/falsifier
                # fields only; compiled stable IDs belong to HypothesisCard
                # artifacts and must not leak into the evaluation schema.
                "predictions": [
                    {
                        "observable": str(item.observable),
                        "expected_direction": str(
                            item.expected_direction
                        ),
                        "rationale": str(item.rationale),
                    }
                    for item in card.predicted_observations
                ],
                "falsifiers": [
                    {
                        "observable": str(item.observable),
                        "falsifying_outcome": str(
                            item.falsifying_outcome
                        ),
                    }
                    for item in card.falsification_criteria
                ],
                "assumptions": list(card.assumptions),
                "source_paper_ids": list(card.source_paper_ids),
                "source_context_id": str(card.source_context_id),
                "source_context_sha256": str(
                    card.source_context_sha256
                ),
                "portfolio_path": str(portfolio_path),
                "marginal_from_previous_condition": False,
                "novelty_authority_created": False,
                "production_selection_authority": False,
            }
        )

        row["scientific_fingerprint"] = stable_id(
            "rvg_scientific_fingerprint",
            {
                "hypothesis_statement": row[
                    "hypothesis_statement"
                ],
                "premise_statement_ids": row[
                    "premise_statement_ids"
                ],
                "inferential_bridge": row[
                    "inferential_bridge"
                ],
            },
        )

        generated_index += 1
        treatment_alias = f"V{generated_index:02d}"
        candidates.append(row)

        mapping.update(
            {
                "treatment_case_candidate_index": (
                    generated_index - 1
                ),
                "treatment_audit_alias": treatment_alias,
                "exact_premise_identity": exact_premise,
                "exact_gap_identity": exact_gap,
                "exact_hypothesis_type_identity": exact_type,
            }
        )
        mapping_rows.append(mapping)

    treatment_case = dict(control_case)
    treatment_case.update(
        {
            "case_id": (
                str(control_case.get("case_id") or "case")
                + "__rvg_treatment_v1"
            ),
            "candidates": candidates,
            "scientific_authority_created": False,
            "production_selection_changed": False,
            "stage8_input_changed": False,
            "candidate_count_is_quality_signal": False,
            "composite_quality_score_computed": False,
        }
    )

    mapping = {
        "schema_version": "rvg-paired-validity-mapping-v1",
        "control_case_id": control_case.get("case_id"),
        "treatment_case_id": treatment_case["case_id"],
        "control_candidate_count": len(
            treatment_summary.get("records", [])
        ),
        "generated_treatment_count": len(candidates),
        "records": mapping_rows,
        "same_grounded_premise_ablation": True,
        "external_prior_art_as_positive_premise": False,
        "production_selection_changed": False,
    }
    return treatment_case, mapping


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--control-case-audit",
        required=True,
        type=Path,
    )
    p.add_argument(
        "--treatment-generation-summary",
        required=True,
        type=Path,
    )
    p.add_argument("--output-case-audit", required=True, type=Path)
    p.add_argument("--output-mapping", required=True, type=Path)
    return p


def main() -> int:
    args = parser().parse_args()

    control_case = load(
        args.control_case_audit.expanduser().resolve()
    )
    treatment_summary = load(
        args.treatment_generation_summary.expanduser().resolve()
    )

    treatment_case, mapping = build_treatment_case(
        control_case=control_case,
        treatment_summary=treatment_summary,
    )

    write(
        args.output_case_audit.expanduser().resolve(),
        treatment_case,
    )
    write(
        args.output_mapping.expanduser().resolve(),
        mapping,
    )

    print("RVG treatment case audit complete")
    print(
        "control candidates:",
        mapping["control_candidate_count"],
    )
    print(
        "generated treatments:",
        mapping["generated_treatment_count"],
    )
    print("same grounded premise ablation: true")
    print("production selection changed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
