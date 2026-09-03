from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    NoveltyClaim,
    NoveltyClaimInferenceProvenance,
)
from pipeline_core.discovery.novelty_claim_decomposition import (
    _clean_query,
    _normalized_identity_present,
)


def _unique_strings(values: list[object]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _normalized_exact(text: str) -> str:
    return _clean_query(str(text or ""), limit=12000).lower()


def _valid_assertions(review: dict[str, Any]) -> list[dict[str, Any]]:
    assertions = review.get("assertions")
    if not isinstance(assertions, list):
        raise ValueError("Inference review assertions must be a list.")
    return [
        row
        for row in assertions
        if isinstance(row, dict)
        and str(row.get("assertion_id") or "").strip()
        and str(row.get("assertion_text") or "").strip()
    ]


def _relation_coverage(assertion_text: str, relation_terms: list[str]) -> int:
    return sum(
        1
        for term in relation_terms
        if _normalized_identity_present(assertion_text, [term])
    )


def _select_assertions_for_claim(
    claim: NoveltyClaim,
    assertions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Bind one atomic claim to accepted Alpha4 inference assertions.

    Matching is lexical and fail-closed: no synonyms, stemming, embeddings,
    LLM adjudication, or scientific-equivalence inference are used.
    """
    predicted = _normalized_exact(claim.predicted_observation)
    if predicted:
        exact_prediction = [
            row
            for row in assertions
            if _normalized_exact(str(row.get("assertion_text") or "")) == predicted
        ]
        if exact_prediction:
            return exact_prediction, ["atomic_binding_exact_prediction_text"]

    identity_terms = [
        str(value)
        for value in claim.prior_art_identity_terms
        if str(value or "").strip()
    ]
    if not identity_terms:
        return [], [
            "atomic_binding_missing_identity_terms",
            "atomic_claim_assertion_binding_unresolved",
        ]

    identity_matches = [
        row
        for row in assertions
        if _normalized_identity_present(
            str(row.get("assertion_text") or ""),
            identity_terms,
        )
    ]
    if not identity_matches:
        return [], [
            "atomic_binding_identity_not_found_in_assertions",
            "atomic_claim_assertion_binding_unresolved",
        ]

    relation_terms = [
        str(value)
        for value in claim.relation_nucleus_terms
        if str(value or "").strip()
    ]
    if relation_terms:
        scored = [
            (
                _relation_coverage(
                    str(row.get("assertion_text") or ""),
                    relation_terms,
                ),
                row,
            )
            for row in identity_matches
        ]
        best = max((score for score, _ in scored), default=0)
        if best > 0:
            return [row for score, row in scored if score == best], [
                "atomic_binding_identity_and_relation_lexical"
            ]

    return identity_matches, ["atomic_binding_identity_lexical"]


def _atomic_provenance(
    *,
    claim: NoveltyClaim,
    fallback: NoveltyClaimInferenceProvenance,
    assertions: list[dict[str, Any]],
    reason_codes: list[str],
) -> NoveltyClaimInferenceProvenance:
    assertion_ids: list[object] = []
    source_classes: list[object] = []
    grounded_statement_ids: list[object] = []
    axis_basis: list[object] = []
    for assertion in assertions:
        assertion_ids.append(assertion.get("assertion_id"))
        source_classes.append(assertion.get("source_class"))
        grounded_statement_ids.extend(assertion.get("grounded_statement_ids") or [])
        axis_basis.extend(assertion.get("axis_basis") or [])

    return NoveltyClaimInferenceProvenance(
        binding_scope="ATOMIC_CLAIM_ASSERTION_BINDING",
        final_hypothesis_id=fallback.final_hypothesis_id,
        source_review_hypothesis_id=fallback.source_review_hypothesis_id,
        axis_id=fallback.axis_id,
        review_status=fallback.review_status,
        assertion_ids=_unique_strings(assertion_ids),
        source_classes=_unique_strings(source_classes),
        grounded_statement_ids=_unique_strings(grounded_statement_ids),
        axis_basis=_unique_strings(axis_basis),
        binding_identity_terms=_unique_strings(list(claim.prior_art_identity_terms)),
        binding_reason_codes=_unique_strings(reason_codes),
    )


def _fallback_provenance(
    *,
    claim: NoveltyClaim,
    fallback: NoveltyClaimInferenceProvenance,
    reason_codes: list[str],
) -> NoveltyClaimInferenceProvenance:
    return fallback.model_copy(
        update={
            "binding_scope": "HYPOTHESIS_REVIEW_CONTEXT",
            "binding_identity_terms": _unique_strings(
                list(claim.prior_art_identity_terms)
            ),
            "binding_reason_codes": _unique_strings(
                [*fallback.binding_reason_codes, *reason_codes]
            ),
        }
    )


def attach_atomic_inference_provenance(
    *,
    decompositions: list[HypothesisNoveltyClaims],
    inference_audit_path: str | Path,
    fallback_by_hypothesis: dict[str, NoveltyClaimInferenceProvenance],
) -> list[HypothesisNoveltyClaims]:
    """Narrow accepted Alpha4 provenance to atomic novelty claims.

    Unresolved bindings retain hypothesis-level review provenance. This changes
    provenance only; it does not change queries, evidence state, or authority.
    """
    payload = json.loads(Path(inference_audit_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "discovery-axis-inference-artifact-v2":
        raise ValueError("Unexpected Alpha4 inference artifact schema.")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Inference artifact records must be a list.")

    record_by_final_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Inference artifact final record must be an object.")
        final_id = str(record.get("final_hypothesis_id") or "").strip()
        if not final_id:
            raise ValueError("Inference artifact record lacks final_hypothesis_id.")
        if final_id in record_by_final_id:
            raise ValueError(f"Duplicate inference final_hypothesis_id: {final_id}")
        record_by_final_id[final_id] = record

    output: list[HypothesisNoveltyClaims] = []
    for group in decompositions:
        fallback = fallback_by_hypothesis.get(group.hypothesis_id)
        if fallback is None:
            output.append(group)
            continue
        record = record_by_final_id.get(group.hypothesis_id)
        if record is None:
            raise ValueError(
                f"Inference artifact lacks final record for {group.hypothesis_id}"
            )
        if str(record.get("axis_id") or "") != fallback.axis_id:
            raise ValueError("Atomic inference binding axis mismatch.")
        if str(record.get("source_review_hypothesis_id") or "") != (
            fallback.source_review_hypothesis_id
        ):
            raise ValueError("Atomic inference binding source-review mismatch.")
        review = record.get("review")
        if not isinstance(review, dict):
            raise ValueError("Inference final record review must be an object.")
        assertions = _valid_assertions(review)

        claims: list[NoveltyClaim] = []
        for claim in group.claims:
            selected, reason_codes = _select_assertions_for_claim(claim, assertions)
            if selected:
                provenance = _atomic_provenance(
                    claim=claim,
                    fallback=fallback,
                    assertions=selected,
                    reason_codes=reason_codes,
                )
            else:
                provenance = _fallback_provenance(
                    claim=claim,
                    fallback=fallback,
                    reason_codes=reason_codes,
                )
            claims.append(
                claim.model_copy(update={"inference_provenance": provenance})
            )
        output.append(group.model_copy(update={"claims": claims}))
    return output
