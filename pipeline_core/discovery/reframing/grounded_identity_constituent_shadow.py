from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.novelty_closure_review import (
    _abstract_contains_identity_anchor,
    _identity_content_tokens,
)
from pipeline_core.discovery.reframing.atomic_cross_lane_synthesis import (
    AtomicCrossLaneSynthesisReport,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionFacingScientificCandidatePortfolio,
)
from pipeline_core.llm.llm_telemetry import run_instructor_structured_call


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GroundedIdentityConstituentSpanDraft(StrictModel):
    candidate_ref: str = Field(min_length=1)
    exact_source_text: str = Field(min_length=1)


class GroundedIdentityConstituentGroupDraft(StrictModel):
    label: str = Field(min_length=1)
    spans: list[GroundedIdentityConstituentSpanDraft] = Field(
        min_length=1,
        max_length=3,
    )


class GroundedIdentityAnnotationDraft(StrictModel):
    claim_id: str = Field(min_length=1)
    identity_term: str = Field(min_length=1)
    groups: list[GroundedIdentityConstituentGroupDraft] = Field(
        min_length=2,
        max_length=4,
    )
    rationale: str = Field(min_length=1)


class GroundedIdentityAnnotationBatchDraft(StrictModel):
    schema_version: Literal[
        "grounded-identity-constituent-annotation-batch-draft-v1"
    ] = "grounded-identity-constituent-annotation-batch-draft-v1"
    annotations: list[GroundedIdentityAnnotationDraft] = Field(
        default_factory=list
    )
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "GroundedIdentityAnnotationBatchDraft":
        if not self.annotations and not self.abstention_reason:
            raise ValueError("empty annotation batch requires abstention_reason")
        if self.annotations and self.abstention_reason:
            raise ValueError("non-empty annotation batch must not carry abstention_reason")
        ids = [row.claim_id for row in self.annotations]
        if len(ids) != len(set(ids)):
            raise ValueError("grounded identity annotations require unique claim_id")
        return self


class GroundedIdentityConstituentSpan(StrictModel):
    candidate_id: str
    candidate_ref: str
    exact_source_text: str
    matched_source_paths: list[str] = Field(min_length=1)


class GroundedIdentityConstituentGroup(StrictModel):
    label: str
    spans: list[GroundedIdentityConstituentSpan]


class GroundedIdentityAnnotation(StrictModel):
    claim_id: str
    identity_term: str
    groups: list[GroundedIdentityConstituentGroup]
    rationale: str

    source_exact_span_only: Literal[True] = True
    semantic_paraphrase_used_for_matching: Literal[False] = False
    embedding_matching_used: Literal[False] = False
    scientific_evidence_authority: Literal[False] = False
    novelty_authority: Literal[False] = False


class GroundedIdentityAnnotationReport(StrictModel):
    schema_version: Literal[
        "grounded-identity-constituent-annotation-report-v1"
    ] = "grounded-identity-constituent-annotation-report-v1"

    source_candidate_portfolio_id: str
    source_atomic_synthesis_report_id: str
    annotations: list[GroundedIdentityAnnotation]
    annotation_count: int = Field(ge=0)
    llm_calls_performed: int = Field(ge=0)
    backend_name: str
    model_name: str
    abstention_reason: str | None = None

    source_exact_span_only: Literal[True] = True
    scientific_evidence_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


class GroundedIdentitySlotAblation(StrictModel):
    slot: str
    strict_state: str
    successful_query_count: int = Field(ge=0)
    material_abstract_work_count: int = Field(ge=0)
    strict_negative_eligible_count: int = Field(ge=0)
    grounded_constituent_negative_eligible_count: int = Field(ge=0)
    grounded_constituent_negative_eligible_work_ids: list[str]
    grounded_constituent_shadow_state: Literal[
        "ESTABLISHED",
        "NOT_FOUND",
        "UNASSESSED",
    ]
    positive_work_ids: list[str]
    identity_term: str
    constituent_groups: list[list[str]]


class GroundedIdentityClaimAblation(StrictModel):
    claim_id: str
    slots: list[GroundedIdentitySlotAblation]
    grounded_constituent_state_counts: dict[str, int]
    grounded_constituent_full_relation_state: str | None = None


class GroundedIdentityNegativeClosureAblationReport(StrictModel):
    schema_version: Literal[
        "grounded-identity-negative-closure-ablation-v1"
    ] = "grounded-identity-negative-closure-ablation-v1"

    source_atomic_synthesis_report_id: str
    source_annotation_report_id: str | None = None
    claim_count: int = Field(ge=0)
    slot_count: int = Field(ge=0)
    claims: list[GroundedIdentityClaimAblation]

    strict_unassessed_nonbase_slot_count: int = Field(ge=0)
    grounded_constituent_not_found_nonbase_slot_count: int = Field(ge=0)
    grounded_constituent_full_relation_not_found_count: int = Field(ge=0)

    current_claim_ids_only: Literal[True] = True
    stale_detail_directories_ignored: Literal[True] = True
    negative_closure_only: Literal[True] = True
    positive_evidence_semantics_changed: Literal[False] = False
    scientific_semantic_inference_used_for_abstract_matching: Literal[False] = False
    embedding_matching_used: Literal[False] = False
    n9_contract_changed: Literal[False] = False
    n10_contract_changed: Literal[False] = False
    production_authority: Literal[False] = False
    production_selection_changed: Literal[False] = False


_MATERIAL_RELATIONSHIPS = {
    "ESTABLISHES_SLOT",
    "PARTIAL_SLOT_RELATION",
    "COMPONENT_ONLY",
}


def _normalize_space(value: object) -> str:
    return " ".join(str(value or "").split())


def _candidate_fields(candidate: Any) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("title", candidate.title),
        ("scientific_proposal", candidate.scientific_proposal),
        ("reasoning_rationale", candidate.reasoning_rationale),
    ]
    rows.extend(
        (f"assumptions[{i}]", value)
        for i, value in enumerate(candidate.assumptions)
    )
    for i, row in enumerate(candidate.predictions):
        payload = row.model_dump(mode="json")
        for key, value in payload.items():
            if isinstance(value, str) and value.strip():
                rows.append((f"predictions[{i}].{key}", value))
    for i, row in enumerate(candidate.falsifiers):
        payload = row.model_dump(mode="json")
        for key, value in payload.items():
            if isinstance(value, str) and value.strip():
                rows.append((f"falsifiers[{i}].{key}", value))
    if candidate.discriminating_test is not None:
        payload = candidate.discriminating_test.model_dump(mode="json")
        for key, value in payload.items():
            if isinstance(value, str) and value.strip():
                rows.append((f"discriminating_test.{key}", value))
            elif isinstance(value, list):
                for j, item in enumerate(value):
                    if isinstance(item, str) and item.strip():
                        rows.append((f"discriminating_test.{key}[{j}]", item))
    rows.extend(
        (f"unresolved_questions[{i}]", value)
        for i, value in enumerate(candidate.unresolved_questions)
    )
    return [(path, value) for path, value in rows if str(value).strip()]


def _candidate_ref_map(
    portfolio: ProductionFacingScientificCandidatePortfolio,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, candidate in enumerate(portfolio.candidates, start=1):
        alias = f"CANDIDATE_{index:02d}"
        result[alias] = candidate
        result[candidate.candidate_id] = candidate
    return result


def build_annotation_prompt(
    *,
    candidates: ProductionFacingScientificCandidatePortfolio,
    atomic_report: AtomicCrossLaneSynthesisReport,
) -> tuple[str, str]:
    candidate_payload = []
    for index, candidate in enumerate(candidates.candidates, start=1):
        alias = f"CANDIDATE_{index:02d}"
        candidate_payload.append(
            {
                "candidate_ref": alias,
                "candidate_id": candidate.candidate_id,
                "source_lane": candidate.source_lane,
                "title": candidate.title,
                "scientific_proposal": candidate.scientific_proposal,
                "reasoning_rationale": candidate.reasoning_rationale,
                "assumptions": list(candidate.assumptions),
                "predictions": [
                    row.model_dump(mode="json")
                    for row in candidate.predictions
                ],
                "falsifiers": [
                    row.model_dump(mode="json")
                    for row in candidate.falsifiers
                ],
                "discriminating_test": (
                    candidate.discriminating_test.model_dump(mode="json")
                    if candidate.discriminating_test is not None
                    else None
                ),
                "unresolved_questions": list(candidate.unresolved_questions),
            }
        )

    claim_payload = []
    for hypothesis in atomic_report.hypotheses:
        for spec in hypothesis.atomic_specifications:
            if not spec.prior_art_identity_terms:
                continue
            claim_payload.append(
                {
                    "claim_id": spec.claim_id,
                    "source_candidate_ids": list(spec.source_candidate_ids),
                    "identity_terms": list(spec.prior_art_identity_terms),
                    "claim_text": spec.text,
                    "required_bridge": spec.required_bridge,
                }
            )

    system = """You annotate a SYNTHETIC prior-art identity with grounded source constituents.

This is diagnostic provenance only. You do NOT judge novelty, truth, literature absence, or production eligibility.

For each claim:
- Produce exactly one annotation for its identity term.
- Split ONLY the synthetic identity itself into 2-4 scientifically meaningful constituent groups.
- Every constituent span MUST be copied verbatim from one of the claim's cited source candidates.
- candidate_ref must identify the source candidate containing that exact text.
- You may provide 1-3 alternative exact spans inside a group when different cited candidates express the same identity constituent differently.
- Every constituent span must contain at least one lexical content token from the synthetic identity.
- Do NOT use the complete synthetic identity itself as a constituent span; this task must decompose it.
- Across the constituent groups, the exact source spans must collectively cover ALL lexical content tokens of the synthetic identity.
- Do NOT add claim outcomes, dependent variables, calibration metrics, oxidation outcomes, matched-measurement conditions, predictions, or falsifiers unless those words are themselves lexical content of the identity term.
- Do not invent synonyms, paraphrases, abbreviations, broader categories, or inferred mechanisms.
- Do not use the synthesized atomic claim itself as a source of constituent text.
- Prefer the shortest exact source noun phrase that carries the identity constituent.
- If the identity cannot be decomposed into at least two grounded source constituents satisfying these lexical constraints, abstain for the entire batch.
"""

    user = json.dumps(
        {
            "source_candidates": candidate_payload,
            "atomic_claims": claim_payload,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user



def _distinct_identity_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _identity_content_tokens(value)
        )
    )


def _validate_identity_constituent_lexical_contract(
    *,
    identity_term: str,
    groups: list[GroundedIdentityConstituentGroup],
) -> None:
    """Fail closed unless source spans actually decompose the identity.

    The annotation LLM may identify source spans, but it cannot broaden the
    identity into outcome variables or matched-measurement conditions.

    Safety rules:
    - at least two constituent groups;
    - no span may equal the complete identity term;
    - every span must share lexical identity content;
    - every group must contribute at least one identity token;
    - the union of groups must cover every canonical identity content token.

    This is lexical provenance validation only. It does not establish
    scientific equivalence, truth, novelty, or prior-art absence.
    """

    identity_tokens = set(
        _distinct_identity_tokens(identity_term)
    )
    if len(identity_tokens) < 2:
        raise ValueError(
            "grounded identity constituent shadow requires at least two "
            "distinct lexical identity content tokens"
        )
    if len(groups) < 2:
        raise ValueError(
            "grounded identity constituent shadow requires at least two groups"
        )

    normalized_identity = _normalize_space(identity_term).casefold()
    covered: set[str] = set()

    for group in groups:
        group_covered: set[str] = set()

        for span in group.spans:
            normalized_span = _normalize_space(
                span.exact_source_text
            ).casefold()

            if normalized_span == normalized_identity:
                raise ValueError(
                    "complete synthetic identity cannot be reused as a "
                    "grounded constituent span"
                )

            span_tokens = set(
                _distinct_identity_tokens(
                    span.exact_source_text
                )
            )
            overlap = span_tokens & identity_tokens
            if not overlap:
                raise ValueError(
                    "grounded constituent span contains no lexical content "
                    "from the synthetic identity: "
                    + repr(span.exact_source_text)
                )

            group_covered.update(overlap)

        if not group_covered:
            raise ValueError(
                "grounded constituent group contributes no identity content: "
                + group.label
            )

        covered.update(group_covered)

    missing = sorted(identity_tokens - covered)
    if missing:
        raise ValueError(
            "grounded constituent groups do not cover all lexical identity "
            "content tokens: "
            + ", ".join(missing)
        )


def compile_annotations(
    *,
    candidates: ProductionFacingScientificCandidatePortfolio,
    atomic_report: AtomicCrossLaneSynthesisReport,
    draft: GroundedIdentityAnnotationBatchDraft,
    backend_name: str,
    model_name: str,
    llm_calls_performed: int,
) -> GroundedIdentityAnnotationReport:
    candidate_by_id = {
        row.candidate_id: row
        for row in candidates.candidates
    }
    refs = _candidate_ref_map(candidates)

    expected: dict[str, tuple[str, set[str]]] = {}
    for hypothesis in atomic_report.hypotheses:
        for spec in hypothesis.atomic_specifications:
            if len(spec.prior_art_identity_terms) != 1:
                raise ValueError(
                    "grounded identity shadow currently requires exactly one "
                    f"identity term per atomic claim: {spec.claim_id}"
                )
            expected[spec.claim_id] = (
                spec.prior_art_identity_terms[0],
                set(spec.source_candidate_ids),
            )

    if set(row.claim_id for row in draft.annotations) != set(expected):
        raise ValueError(
            "grounded identity annotation membership must match current atomic claims"
        )

    output: list[GroundedIdentityAnnotation] = []
    for row in draft.annotations:
        expected_identity, allowed_candidate_ids = expected[row.claim_id]
        if _normalize_space(row.identity_term).casefold() != (
            _normalize_space(expected_identity).casefold()
        ):
            raise ValueError(
                f"identity term drift for claim {row.claim_id}"
            )

        compiled_groups: list[GroundedIdentityConstituentGroup] = []
        for group in row.groups:
            spans: list[GroundedIdentityConstituentSpan] = []
            for span in group.spans:
                candidate = refs.get(span.candidate_ref)
                if candidate is None:
                    raise ValueError(
                        f"unknown candidate_ref in identity annotation: "
                        f"{span.candidate_ref}"
                    )
                if candidate.candidate_id not in allowed_candidate_ids:
                    raise ValueError(
                        f"identity constituent cites candidate outside atomic "
                        f"source set: {span.candidate_ref}"
                    )

                needle = _normalize_space(span.exact_source_text)
                matches = [
                    path
                    for path, value in _candidate_fields(candidate)
                    if needle and needle in _normalize_space(value)
                ]
                if not matches:
                    raise ValueError(
                        f"identity constituent is not an exact source span "
                        f"for {span.candidate_ref}: {needle!r}"
                    )

                spans.append(
                    GroundedIdentityConstituentSpan(
                        candidate_id=candidate.candidate_id,
                        candidate_ref=span.candidate_ref,
                        exact_source_text=needle,
                        matched_source_paths=sorted(set(matches)),
                    )
                )
            compiled_groups.append(
                GroundedIdentityConstituentGroup(
                    label=group.label,
                    spans=spans,
                )
            )

        _validate_identity_constituent_lexical_contract(
            identity_term=expected_identity,
            groups=compiled_groups,
        )

        output.append(
            GroundedIdentityAnnotation(
                claim_id=row.claim_id,
                identity_term=expected_identity,
                groups=compiled_groups,
                rationale=row.rationale,
            )
        )

    return GroundedIdentityAnnotationReport(
        source_candidate_portfolio_id=candidates.portfolio_id,
        source_atomic_synthesis_report_id=atomic_report.report_id,
        annotations=output,
        annotation_count=len(output),
        llm_calls_performed=llm_calls_performed,
        backend_name=backend_name,
        model_name=model_name,
        abstention_reason=draft.abstention_reason,
    )


class InstructorGroundedIdentityBackend:
    backend_name = "instructor_openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        parse_retries: int = 1,
        timeout: float | None = 180.0,
    ) -> None:
        self.model_name = model
        self.api_key_env = api_key_env
        self.api_key = os.getenv(api_key_env)
        self.base_url = base_url
        self.parse_retries = parse_retries
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(f"No API key available. Set {self.api_key_env}.")
        try:
            import instructor
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Grounded identity annotation requires openai + instructor."
            ) from exc
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        self._client = instructor.from_openai(
            OpenAI(**kwargs),
            mode=instructor.Mode.JSON,
        )
        return self._client

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GroundedIdentityAnnotationBatchDraft:
        client = self._get_client()
        draft, _event = run_instructor_structured_call(
            client.chat.completions,
            model=self.model_name,
            response_model=GroundedIdentityAnnotationBatchDraft,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_retries=self.parse_retries,
            telemetry_context={
                "pipeline": "grounded_identity_constituent_shadow",
                "stage": "source_bounded_identity_annotation",
                "call_kind": "scientific_hypothesis_synthesis",
            },
            semantic_components={
                "annotation_contract":
                    "grounded_identity_constituent_exact_source_v1",
            },
        )
        if not isinstance(draft, GroundedIdentityAnnotationBatchDraft):
            draft = GroundedIdentityAnnotationBatchDraft.model_validate(draft)
        return draft


def run_annotation(
    *,
    candidates: ProductionFacingScientificCandidatePortfolio,
    atomic_report: AtomicCrossLaneSynthesisReport,
    backend: InstructorGroundedIdentityBackend,
) -> tuple[GroundedIdentityAnnotationReport, str, str]:
    system, user = build_annotation_prompt(
        candidates=candidates,
        atomic_report=atomic_report,
    )
    draft = backend.generate(
        system_prompt=system,
        user_prompt=user,
    )
    report = compile_annotations(
        candidates=candidates,
        atomic_report=atomic_report,
        draft=draft,
        backend_name=backend.backend_name,
        model_name=backend.model_name,
        llm_calls_performed=1,
    )
    return report, system, user


def _material_work_ids(
    slot_review: Mapping[str, Any],
) -> set[str]:
    result: set[str] = set()
    for row in slot_review.get("matches") or []:
        if not isinstance(row, Mapping):
            continue
        if row.get("abstract_available") is not True:
            continue
        if str(row.get("relationship") or "") not in _MATERIAL_RELATIONSHIPS:
            continue
        work_id = str(row.get("work_id") or "").strip()
        if work_id:
            result.add(work_id)
    return result


def _group_matches_abstract(
    *,
    abstract: str,
    group: GroundedIdentityConstituentGroup,
) -> bool:
    return any(
        _abstract_contains_identity_anchor(
            abstract=abstract,
            anchors=(span.exact_source_text,),
        )
        for span in group.spans
    )


def _identity_matches_abstract(
    *,
    abstract: str,
    annotation: GroundedIdentityAnnotation,
) -> bool:
    return all(
        _group_matches_abstract(
            abstract=abstract,
            group=group,
        )
        for group in annotation.groups
    )


def _shadow_state(
    *,
    successful_query_count: int,
    positive_work_ids: list[str],
    eligible_count: int,
    minimum_negative_abstracts: int,
) -> Literal["ESTABLISHED", "NOT_FOUND", "UNASSESSED"]:
    if successful_query_count <= 0:
        return "UNASSESSED"
    if positive_work_ids:
        return "ESTABLISHED"
    if eligible_count >= minimum_negative_abstracts:
        return "NOT_FOUND"
    return "UNASSESSED"


def analyze_current_claims(
    *,
    detail_root: Path,
    atomic_report: AtomicCrossLaneSynthesisReport,
    annotation_report: GroundedIdentityAnnotationReport,
    minimum_negative_abstracts: int = 3,
) -> GroundedIdentityNegativeClosureAblationReport:
    annotations = {
        row.claim_id: row
        for row in annotation_report.annotations
    }
    current_claim_ids = {
        spec.claim_id
        for hypothesis in atomic_report.hypotheses
        for spec in hypothesis.atomic_specifications
    }
    if set(annotations) != current_claim_ids:
        raise ValueError(
            "grounded identity annotations must cover current atomic claims exactly"
        )

    claim_rows: list[GroundedIdentityClaimAblation] = []

    for claim_id in sorted(current_claim_ids):
        claim_dir = detail_root / claim_id.replace(":", "_")
        if not claim_dir.is_dir():
            raise ValueError(f"missing current claim detail directory: {claim_dir}")

        plan = json.loads(
            (claim_dir / "closure_plan.json").read_text(encoding="utf-8")
        )
        reviews = json.loads(
            (claim_dir / "slot_reviews.json").read_text(encoding="utf-8")
        )
        prior_art = json.loads(
            (claim_dir / "prior_art.json").read_text(encoding="utf-8")
        )
        if not isinstance(reviews, list):
            raise ValueError("slot_reviews must be a list")

        works_by_id = {
            str(row.get("work_id") or ""): row
            for row in prior_art.get("works") or []
            if isinstance(row, Mapping)
            and str(row.get("work_id") or "").strip()
        }
        annotation = annotations[claim_id]

        slots: list[GroundedIdentitySlotAblation] = []
        for review in reviews:
            slot = str(review.get("slot") or "")
            positive_work_ids = sorted(
                str(value)
                for value in (review.get("positive_work_ids") or [])
                if str(value).strip()
            )
            material = _material_work_ids(review)
            if slot == "BASE_RELATION":
                eligible = set(material)
            else:
                eligible = {
                    work_id
                    for work_id in material
                    if (
                        work_id in works_by_id
                        and _identity_matches_abstract(
                            abstract=str(
                                works_by_id[work_id].get("abstract") or ""
                            ),
                            annotation=annotation,
                        )
                    )
                }

            state = _shadow_state(
                successful_query_count=int(
                    review.get("successful_query_count") or 0
                ),
                positive_work_ids=positive_work_ids,
                eligible_count=len(eligible),
                minimum_negative_abstracts=minimum_negative_abstracts,
            )
            slots.append(
                GroundedIdentitySlotAblation(
                    slot=slot,
                    strict_state=str(review.get("evidence_state") or ""),
                    successful_query_count=int(
                        review.get("successful_query_count") or 0
                    ),
                    material_abstract_work_count=len(material),
                    strict_negative_eligible_count=int(
                        review.get(
                            "negative_eligible_material_abstract_review_count"
                        )
                        or 0
                    ),
                    grounded_constituent_negative_eligible_count=len(eligible),
                    grounded_constituent_negative_eligible_work_ids=sorted(
                        eligible
                    ),
                    grounded_constituent_shadow_state=state,
                    positive_work_ids=positive_work_ids,
                    identity_term=annotation.identity_term,
                    constituent_groups=[
                        [
                            span.exact_source_text
                            for span in group.spans
                        ]
                        for group in annotation.groups
                    ],
                )
            )

        counts = Counter(
            row.grounded_constituent_shadow_state
            for row in slots
        )
        full = next(
            (row for row in slots if row.slot == "FULL_RELATION"),
            None,
        )
        claim_rows.append(
            GroundedIdentityClaimAblation(
                claim_id=claim_id,
                slots=slots,
                grounded_constituent_state_counts=dict(sorted(counts.items())),
                grounded_constituent_full_relation_state=(
                    full.grounded_constituent_shadow_state
                    if full is not None
                    else None
                ),
            )
        )

    nonbase = [
        slot
        for claim in claim_rows
        for slot in claim.slots
        if slot.slot != "BASE_RELATION"
    ]
    return GroundedIdentityNegativeClosureAblationReport(
        source_atomic_synthesis_report_id=atomic_report.report_id,
        source_annotation_report_id=None,
        claim_count=len(claim_rows),
        slot_count=sum(len(row.slots) for row in claim_rows),
        claims=claim_rows,
        strict_unassessed_nonbase_slot_count=sum(
            slot.strict_state == "UNASSESSED"
            for slot in nonbase
        ),
        grounded_constituent_not_found_nonbase_slot_count=sum(
            slot.grounded_constituent_shadow_state == "NOT_FOUND"
            for slot in nonbase
        ),
        grounded_constituent_full_relation_not_found_count=sum(
            claim.grounded_constituent_full_relation_state == "NOT_FOUND"
            for claim in claim_rows
        ),
    )


__all__ = [
    "GroundedIdentityAnnotationReport",
    "GroundedIdentityNegativeClosureAblationReport",
    "InstructorGroundedIdentityBackend",
    "analyze_current_claims",
    "compile_annotations",
    "run_annotation",
]
