from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline_core.discovery.reframing.critic_contracts import (
    CRITIC_DIMENSIONS,
    ReframeCriticDimensionReview,
    ReframeCriticDraft,
    ReframeStructuralAudit,
    ScientificReframeCriticReport,
    ScientificReframeCriticReview,
    stable_critic_report_id,
    stable_critic_review_id,
)
from pipeline_core.discovery.reframing.critic_llm import ReframeCriticBackend
from pipeline_core.discovery.reframing.critic_prompt import (
    ScientificReframeCriticPrompt,
    ScientificReframeCriticPromptAssembler,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)


@dataclass(frozen=True)
class ReframeCriticPromptRecord:
    candidate_id: str
    prompt: ScientificReframeCriticPrompt


@dataclass(frozen=True)
class ScientificReframeCriticOutcome:
    report: ScientificReframeCriticReport
    prompts: tuple[ReframeCriticPromptRecord, ...]


def _dedupe_nonblank(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            rows.append(text)
    return rows


def _structural_audit(
    candidate: ScientificReframeCandidate,
    evidence: ScientificReframeEvidencePacket,
) -> ReframeStructuralAudit:
    premise_ids = {row.statement_id for row in evidence.premise_statements}
    gap_ids = {row.statement_id for row in evidence.gap_statements}
    if candidate.operator_id == "LATENT_VARIABLE":
        operator_shape_valid = bool(
            candidate.latent_constructs
            and candidate.proposed_constructs
            and not candidate.boundary_variables
            and candidate.regime_change_kind is None
            and len(set(candidate.alternative_model.explained_statement_ids)) >= 2
        )
    else:
        operator_shape_valid = bool(
            candidate.boundary_variables
            and candidate.regime_change_kind is not None
            and not candidate.latent_constructs
        )
    return ReframeStructuralAudit(
        grounded_premise_count=len(candidate.premise_statement_ids),
        gap_count=len(candidate.gap_statement_ids),
        baseline_explained_count=len(set(candidate.baseline_model.explained_statement_ids)),
        alternative_explained_count=len(set(candidate.alternative_model.explained_statement_ids)),
        speculative_construct_count=len(candidate.proposed_constructs),
        differential_prediction_count=len(candidate.differential_predictions),
        falsifier_count=len(candidate.falsifiers),
        primary_observable_count=len(candidate.discriminating_test.primary_observables),
        operator_shape_valid=operator_shape_valid,
        all_premise_ids_grounded=set(candidate.premise_statement_ids) <= premise_ids,
        all_gap_ids_grounded=set(candidate.gap_statement_ids) <= gap_ids,
    )


def _parse_rating(value: int | str | None) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, f"invalid non-integer critic rating {value!r}"
    if parsed < 0 or parsed > 3:
        clamped = max(0, min(3, parsed))
        return clamped, f"clamped critic rating from {parsed} to {clamped}"
    return parsed, None


def compile_critic_review(
    *,
    candidate: ScientificReframeCandidate,
    evidence: ScientificReframeEvidencePacket,
    draft: ReframeCriticDraft | None,
    generation: Any | None = None,
    llm_error_type: str | None = None,
) -> ScientificReframeCriticReview:
    valid_statement_ids = {
        row.statement_id
        for row in [*evidence.premise_statements, *evidence.gap_statements]
    }
    normalizations: list[str] = []
    draft_by_dimension: dict[str, Any] = {}
    cross_cutting: list[str] = []

    if draft is not None:
        if draft.candidate_id != candidate.reframe_id:
            normalizations.append(
                "critic draft candidate_id did not match requested candidate; request lineage retained"
            )
        cross_cutting = _dedupe_nonblank(draft.cross_cutting_concerns)
        for row in draft.dimensions:
            dimension = str(row.dimension).strip()
            if dimension not in CRITIC_DIMENSIONS:
                normalizations.append(f"ignored unknown critic dimension: {dimension}")
                continue
            if dimension in draft_by_dimension:
                normalizations.append(
                    f"ignored duplicate critic dimension after first occurrence: {dimension}"
                )
                continue
            draft_by_dimension[dimension] = row

    reviews: list[ReframeCriticDimensionReview] = []
    for dimension in CRITIC_DIMENSIONS:
        row = draft_by_dimension.get(dimension)
        if row is None:
            reviews.append(
                ReframeCriticDimensionReview(
                    dimension=dimension,
                    rating=None,
                    review_status="missing" if llm_error_type is None else "invalid",
                    rationale="",
                    supporting_statement_ids=[],
                    concerns=[],
                )
            )
            continue

        rating, note = _parse_rating(row.rating)
        if note:
            normalizations.append(f"{dimension}: {note}")
        supported_ids = [
            value
            for value in _dedupe_nonblank(row.supporting_statement_ids)
            if value in valid_statement_ids
        ]
        removed = sorted(
            set(_dedupe_nonblank(row.supporting_statement_ids)) - set(supported_ids)
        )
        if removed:
            normalizations.append(
                f"{dimension}: removed non-grounded supporting_statement_ids: "
                + ", ".join(removed)
            )
        reviews.append(
            ReframeCriticDimensionReview(
                dimension=dimension,
                rating=rating,
                review_status="reviewed" if rating is not None else "invalid",
                rationale=str(row.rationale or "").strip(),
                supporting_statement_ids=supported_ids,
                concerns=_dedupe_nonblank(row.concerns),
            )
        )

    return ScientificReframeCriticReview(
        review_id=stable_critic_review_id(
            candidate_id=candidate.reframe_id,
            context_sha256=candidate.source_context_sha256,
        ),
        candidate_id=candidate.reframe_id,
        operator_id=candidate.operator_id,
        source_task_id=candidate.source_task_id,
        source_context_id=candidate.source_context_id,
        source_context_sha256=candidate.source_context_sha256,
        structural_audit=_structural_audit(candidate, evidence),
        dimensions=reviews,
        cross_cutting_concerns=cross_cutting,
        normalization_notes=normalizations,
        llm_review_complete=all(
            row.review_status == "reviewed"
            for row in reviews
        ),
        llm_error_type=llm_error_type,
        input_tokens=getattr(generation, "input_tokens", None),
        output_tokens=getattr(generation, "output_tokens", None),
        response_id=getattr(generation, "response_id", None),
        elapsed_seconds=getattr(generation, "elapsed_seconds", None),
    )


class ScientificReframeCriticRuntime:
    def __init__(
        self,
        backend: ReframeCriticBackend,
        *,
        prompt_assembler: ScientificReframeCriticPromptAssembler | None = None,
    ) -> None:
        self.backend = backend
        self.prompt_assembler = prompt_assembler or ScientificReframeCriticPromptAssembler()

    def run(
        self,
        *,
        shadow: ScientificReframingShadowReport,
        evidence: ScientificReframeEvidencePacket,
    ) -> ScientificReframeCriticOutcome:
        if shadow.source_task_id != evidence.task_id:
            raise ValueError("shadow/evidence task mismatch")
        if shadow.source_context_id != evidence.source_context_id:
            raise ValueError("shadow/evidence context mismatch")
        if shadow.source_context_sha256 != evidence.source_context_sha256:
            raise ValueError("shadow/evidence context SHA mismatch")

        prompts: list[ReframeCriticPromptRecord] = []
        reviews: list[ScientificReframeCriticReview] = []
        attempted = 0
        succeeded = 0

        for candidate in shadow.candidates:
            prompt = self.prompt_assembler.build(
                candidate=candidate,
                evidence=evidence,
            )
            prompts.append(
                ReframeCriticPromptRecord(
                    candidate_id=candidate.reframe_id,
                    prompt=prompt,
                )
            )
            attempted += 1
            try:
                generation = self.backend.review(prompt)
            except Exception as exc:  # shadow critic failure must not erase structural audit
                reviews.append(
                    compile_critic_review(
                        candidate=candidate,
                        evidence=evidence,
                        draft=None,
                        generation=None,
                        llm_error_type=type(exc).__name__,
                    )
                )
                continue
            succeeded += 1
            reviews.append(
                compile_critic_review(
                    candidate=candidate,
                    evidence=evidence,
                    draft=generation.draft,
                    generation=generation,
                )
            )

        review_ids = [row.review_id for row in reviews]
        report = ScientificReframeCriticReport(
            report_id=stable_critic_report_id(
                shadow_report_id=shadow.report_id,
                review_ids=review_ids,
            ),
            source_shadow_report_id=shadow.report_id,
            source_task_id=shadow.source_task_id,
            source_context_id=shadow.source_context_id,
            source_context_sha256=shadow.source_context_sha256,
            backend_name=self.backend.backend_name,
            model_name=self.backend.model_name,
            reviews=reviews,
            llm_calls_attempted=attempted,
            llm_calls_succeeded=succeeded,
        )
        return ScientificReframeCriticOutcome(
            report=report,
            prompts=tuple(prompts),
        )
