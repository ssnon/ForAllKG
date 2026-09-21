from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pipeline_core.discovery.reframing.operator_readiness import (
    ReframingOperatorReadinessReport,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    ImplementedReframeOperatorId,
    ReframeOperatorRunRecord,
    ScientificReframeCandidate,
    ScientificReframeDraft,
    ScientificReframingShadowReport,
    stable_reframe_id,
)
from pipeline_core.discovery.reframing.reframe_evidence import (
    ScientificReframeEvidencePacket,
)
from pipeline_core.discovery.reframing.reframe_llm import (
    ReframeDraftBackend,
)
from pipeline_core.discovery.reframing.reframe_prompt import (
    ScientificReframePrompt,
    ScientificReframePromptAssembler,
)
from pipeline_core.discovery.reframing.trigger_contracts import (
    ScientificReframeTriggerReport,
)


@dataclass(frozen=True)
class ReframePromptRecord:
    operator_id: ImplementedReframeOperatorId
    prompt: ScientificReframePrompt


@dataclass(frozen=True)
class ScientificReframingShadowOutcome:
    report: ScientificReframingShadowReport
    prompts: tuple[ReframePromptRecord, ...]


class ScientificReframeCompileError(ValueError):
    pass


def _prompt_sha(prompt: ScientificReframePrompt) -> str:
    payload = (prompt.system_prompt + "\n\n" + prompt.user_prompt).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dedupe_nonblank(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _compile_candidate(
    *,
    evidence: ScientificReframeEvidencePacket,
    draft: ScientificReframeDraft,
    requested_operator_id: ImplementedReframeOperatorId,
) -> ScientificReframeCandidate:
    if draft.operator_id != requested_operator_id:
        raise ScientificReframeCompileError(
            "candidate operator_id does not match requested operator: "
            f"{draft.operator_id} != {requested_operator_id}"
        )

    premise_order = [row.statement_id for row in evidence.premise_statements]
    gap_order = [row.statement_id for row in evidence.gap_statements]
    premise_ids = set(premise_order)
    gap_ids = set(gap_order)

    declared_premises = set(_dedupe_nonblank(draft.premise_statement_ids))
    declared_gaps = set(_dedupe_nonblank(draft.gap_statement_ids))
    explained_ids = set(draft.baseline_model.explained_statement_ids) | set(
        draft.alternative_model.explained_statement_ids
    )
    explained_ids = {str(value).strip() for value in explained_ids if str(value).strip()}

    unknown_premises = sorted(declared_premises - premise_ids)
    unknown_gaps = sorted(declared_gaps - gap_ids)
    unknown_explained = sorted(explained_ids - premise_ids)
    if unknown_premises:
        raise ScientificReframeCompileError(
            "draft references non-grounded premise IDs: " + ", ".join(unknown_premises)
        )
    if unknown_gaps:
        raise ScientificReframeCompileError(
            "draft references non-grounded gap IDs: " + ", ".join(unknown_gaps)
        )
    if unknown_explained:
        raise ScientificReframeCompileError(
            "model explanation references non-grounded premise IDs: "
            + ", ".join(unknown_explained)
        )

    normalizations: list[str] = []
    implicit_grounded_premises = explained_ids - declared_premises
    if implicit_grounded_premises:
        normalizations.append(
            "promoted grounded explained_statement_ids into premise_statement_ids: "
            + ", ".join(sorted(implicit_grounded_premises))
        )

    canonical_premise_ids = [
        statement_id
        for statement_id in premise_order
        if statement_id in (declared_premises | explained_ids)
    ]
    canonical_gap_ids = [
        statement_id for statement_id in gap_order if statement_id in declared_gaps
    ]
    if not canonical_premise_ids:
        raise ScientificReframeCompileError(
            "candidate has no grounded positive premise after compilation"
        )

    if len(draft.premise_statement_ids) != len(_dedupe_nonblank(draft.premise_statement_ids)):
        normalizations.append("deduplicated or removed blank premise_statement_ids")
    if len(draft.gap_statement_ids) != len(_dedupe_nonblank(draft.gap_statement_ids)):
        normalizations.append("deduplicated or removed blank gap_statement_ids")

    def normalize_model(model, *, name: str):
        selected = {
            str(value).strip()
            for value in model.explained_statement_ids
            if str(value).strip()
        }
        canonical = [statement_id for statement_id in premise_order if statement_id in selected]
        if not canonical:
            raise ScientificReframeCompileError(
                f"{name} must explain at least one grounded premise"
            )
        if not model.expected_observations:
            raise ScientificReframeCompileError(
                f"{name} requires at least one expected observation"
            )
        if canonical != list(model.explained_statement_ids):
            normalizations.append(
                f"canonicalized {name}.explained_statement_ids to grounded evidence order"
            )
        return model.model_copy(update={"explained_statement_ids": canonical})

    baseline_model = normalize_model(draft.baseline_model, name="baseline_model")
    alternative_model = normalize_model(
        draft.alternative_model,
        name="alternative_model",
    )
    if baseline_model.summary.strip() == alternative_model.summary.strip():
        raise ScientificReframeCompileError(
            "baseline and alternative model summaries are identical"
        )

    prediction_ids = [row.local_id for row in draft.differential_predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ScientificReframeCompileError(
            "duplicate differential prediction local_id values"
        )
    differential_predictions = [
        row
        for row in draft.differential_predictions
        if row.baseline_expectation.strip() != row.alternative_expectation.strip()
    ]
    removed_predictions = len(draft.differential_predictions) - len(
        differential_predictions
    )
    if removed_predictions:
        normalizations.append(
            f"removed {removed_predictions} non-differential prediction(s)"
        )
    if not differential_predictions:
        raise ScientificReframeCompileError(
            "candidate requires at least one differential prediction"
        )

    falsifier_ids = [row.local_id for row in draft.falsifiers]
    if len(falsifier_ids) != len(set(falsifier_ids)):
        raise ScientificReframeCompileError("duplicate falsifier local_id values")
    if not draft.falsifiers:
        raise ScientificReframeCompileError("candidate requires at least one falsifier")
    if not draft.discriminating_test.primary_observables:
        raise ScientificReframeCompileError(
            "discriminating test requires at least one primary observable"
        )

    proposed_constructs = _dedupe_nonblank(draft.proposed_constructs)
    latent_constructs = _dedupe_nonblank(draft.latent_constructs)
    boundary_variables = _dedupe_nonblank(draft.boundary_variables)
    regime_change_kind = draft.regime_change_kind

    if len(proposed_constructs) > 2:
        proposed_constructs = proposed_constructs[:2]
        normalizations.append("truncated proposed_constructs to operator budget of 2")

    if requested_operator_id == "LATENT_VARIABLE":
        if not latent_constructs:
            raise ScientificReframeCompileError(
                "LATENT_VARIABLE requires at least one latent_construct"
            )
        if not proposed_constructs:
            raise ScientificReframeCompileError(
                "LATENT_VARIABLE requires at least one proposed_construct"
            )
        if len(set(alternative_model.explained_statement_ids)) < 2:
            raise ScientificReframeCompileError(
                "LATENT_VARIABLE alternative model must explain at least two grounded premises"
            )
        if len(latent_constructs) > 2:
            latent_constructs = latent_constructs[:2]
            normalizations.append("truncated latent_constructs to operator budget of 2")
        stripped: list[str] = []
        if boundary_variables:
            stripped.append("boundary_variables")
            boundary_variables = []
        if regime_change_kind is not None:
            stripped.append("regime_change_kind")
            regime_change_kind = None
        if stripped:
            normalizations.append(
                "stripped regime-specific fields from LATENT_VARIABLE draft: "
                + ", ".join(stripped)
            )
    else:
        if not boundary_variables:
            raise ScientificReframeCompileError(
                "REGIME_BOUNDARY requires at least one boundary_variable"
            )
        if regime_change_kind is None:
            raise ScientificReframeCompileError(
                "REGIME_BOUNDARY requires regime_change_kind"
            )
        if len(boundary_variables) > 4:
            boundary_variables = boundary_variables[:4]
            normalizations.append("truncated boundary_variables to operator budget of 4")
        if latent_constructs:
            latent_constructs = []
            normalizations.append(
                "stripped latent_constructs from REGIME_BOUNDARY draft"
            )

    normalized_draft = draft.model_copy(
        update={
            "premise_statement_ids": canonical_premise_ids,
            "gap_statement_ids": canonical_gap_ids,
            "baseline_model": baseline_model,
            "alternative_model": alternative_model,
            "proposed_constructs": proposed_constructs,
            "latent_constructs": latent_constructs,
            "boundary_variables": boundary_variables,
            "regime_change_kind": regime_change_kind,
            "differential_predictions": differential_predictions,
        }
    )

    reframe_id = stable_reframe_id(
        task_id=evidence.task_id,
        context_sha256=evidence.source_context_sha256,
        operator_id=requested_operator_id,
        draft=normalized_draft,
    )
    return ScientificReframeCandidate(
        reframe_id=reframe_id,
        operator_id=requested_operator_id,
        source_task_id=evidence.task_id,
        source_context_id=evidence.source_context_id,
        source_context_sha256=evidence.source_context_sha256,
        title=normalized_draft.title,
        premise_statement_ids=list(normalized_draft.premise_statement_ids),
        gap_statement_ids=list(normalized_draft.gap_statement_ids),
        baseline_model=normalized_draft.baseline_model,
        alternative_model=normalized_draft.alternative_model,
        challenged_assumption=normalized_draft.challenged_assumption,
        proposed_constructs=list(normalized_draft.proposed_constructs),
        latent_constructs=list(normalized_draft.latent_constructs),
        boundary_variables=list(normalized_draft.boundary_variables),
        regime_change_kind=normalized_draft.regime_change_kind,
        differential_predictions=list(normalized_draft.differential_predictions),
        falsifiers=list(normalized_draft.falsifiers),
        discriminating_test=normalized_draft.discriminating_test,
        unresolved_questions=list(normalized_draft.unresolved_questions),
        provenance_normalizations=normalizations,
    )


class ScientificReframingShadowRuntime:
    def __init__(
        self,
        backend: ReframeDraftBackend,
        *,
        prompt_assembler: ScientificReframePromptAssembler | None = None,
    ) -> None:
        self.backend = backend
        self.prompt_assembler = prompt_assembler or ScientificReframePromptAssembler()

    def run(
        self,
        *,
        evidence: ScientificReframeEvidencePacket,
        readiness: ReframingOperatorReadinessReport,
        operators: list[ImplementedReframeOperatorId] | None = None,
        trigger_report: ScientificReframeTriggerReport | None = None,
    ) -> ScientificReframingShadowOutcome:
        if readiness.scope_id != evidence.task_id:
            raise ValueError("readiness scope_id must equal evidence task_id")
        if trigger_report is not None:
            if trigger_report.source_task_id != evidence.task_id:
                raise ValueError("trigger report task_id must equal evidence task_id")
            if trigger_report.source_context_id != evidence.source_context_id:
                raise ValueError("trigger report context_id must equal evidence context_id")
            if (
                trigger_report.source_context_sha256
                != evidence.source_context_sha256
            ):
                raise ValueError(
                    "trigger report context_sha256 must equal evidence context_sha256"
                )
        selected = operators or ["LATENT_VARIABLE", "REGIME_BOUNDARY"]
        if len(selected) != len(set(selected)):
            raise ValueError("operators must be unique")

        readiness_by_operator = {
            row.operator_id: row
            for row in readiness.assessments
        }
        trigger_by_operator = (
            {row.operator_id: row for row in trigger_report.assessments}
            if trigger_report is not None
            else {}
        )
        prompts: list[ReframePromptRecord] = []
        runs: list[ReframeOperatorRunRecord] = []
        candidates: list[ScientificReframeCandidate] = []

        for operator_id in selected:
            assessment = readiness_by_operator.get(operator_id)
            if assessment is None:
                raise ValueError(f"readiness report lacks operator {operator_id}")

            if trigger_report is not None:
                trigger = trigger_by_operator.get(operator_id)
                if trigger is None:
                    raise ValueError(f"trigger report lacks operator {operator_id}")
                if trigger.decision != "triggered":
                    runs.append(
                        ReframeOperatorRunRecord(
                            operator_id=operator_id,
                            readiness_status=assessment.status,
                            decision="skipped_not_triggered",
                            reasons=[*list(assessment.reasons), *list(trigger.reasons)],
                        )
                    )
                    continue

            executable = assessment.status in {
                "ready_now",
                "ready_with_optional_enrichment",
            }
            if not executable:
                runs.append(
                    ReframeOperatorRunRecord(
                        operator_id=operator_id,
                        readiness_status=assessment.status,
                        decision="skipped_not_ready",
                        reasons=list(assessment.reasons),
                    )
                )
                continue

            prompt = self.prompt_assembler.build(
                operator_id=operator_id,
                evidence=evidence,
            )
            prompts.append(ReframePromptRecord(operator_id=operator_id, prompt=prompt))
            try:
                generation = self.backend.generate(prompt)
            except Exception as exc:  # shadow lane: record generation failure, do not kill later operators
                runs.append(
                    ReframeOperatorRunRecord(
                        operator_id=operator_id,
                        readiness_status=assessment.status,
                        decision="generation_failed",
                        prompt_sha256=_prompt_sha(prompt),
                        abstention_reason=(
                            "Structured generation failed before draft compilation."
                        ),
                        reasons=list(assessment.reasons),
                        generation_error_type=type(exc).__name__,
                    )
                )
                continue

            draft = generation.draft
            compiled: list[ScientificReframeCandidate] = []
            compile_issues: list[str] = []
            rejected_candidate_count = 0
            if draft.operator_id != operator_id:
                compile_issues.append(
                    "batch operator_id did not match requested operator; "
                    "requested operator governed candidate compilation"
                )

            rows = list(draft.candidates)
            if len(rows) > 2:
                rejected_candidate_count += len(rows) - 2
                compile_issues.append(
                    f"ignored {len(rows) - 2} candidate(s) beyond operator quota of 2"
                )
                rows = rows[:2]

            for row in rows:
                try:
                    compiled.append(
                        _compile_candidate(
                            evidence=evidence,
                            draft=row,
                            requested_operator_id=operator_id,
                        )
                    )
                except ScientificReframeCompileError as exc:
                    rejected_candidate_count += 1
                    compile_issues.append(
                        f"candidate {row.local_id}: {exc}"
                    )

            candidates.extend(compiled)
            if compiled:
                decision = "generated"
                abstention_reason = None
            elif draft.candidates:
                decision = "rejected_invalid_draft"
                abstention_reason = (
                    "All generated candidates failed grounded/operator compilation."
                )
            else:
                decision = "abstained"
                abstention_reason = (
                    draft.abstention_reason
                    or "Model returned no candidates without an abstention_reason."
                )

            runs.append(
                ReframeOperatorRunRecord(
                    operator_id=operator_id,
                    readiness_status=assessment.status,
                    decision=decision,
                    prompt_sha256=_prompt_sha(prompt),
                    candidate_ids=[row.reframe_id for row in compiled],
                    abstention_reason=abstention_reason,
                    reasons=list(assessment.reasons),
                    rejected_candidate_count=rejected_candidate_count,
                    compile_issues=compile_issues,
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens,
                    response_id=generation.response_id,
                    elapsed_seconds=generation.elapsed_seconds,
                )
            )

        report_seed = "|".join(
            [
                evidence.task_id,
                evidence.source_context_sha256,
                self.backend.backend_name,
                self.backend.model_name,
                *[run.prompt_sha256 or run.decision for run in runs],
                *[candidate.reframe_id for candidate in candidates],
            ]
        ).encode("utf-8")
        report_id = (
            "scientific_reframing_shadow:"
            + hashlib.sha256(report_seed).hexdigest()[:20]
        )
        report = ScientificReframingShadowReport(
            report_id=report_id,
            source_task_id=evidence.task_id,
            source_context_id=evidence.source_context_id,
            source_context_sha256=evidence.source_context_sha256,
            backend_name=self.backend.backend_name,
            model_name=self.backend.model_name,
            runs=runs,
            candidates=candidates,
            llm_calls_performed=sum(
                run.prompt_sha256 is not None
                for run in runs
            ),
            source_trigger_report_id=(
                trigger_report.report_id if trigger_report is not None else None
            ),
            scientific_trigger_evaluated=(trigger_report is not None),
        )
        return ScientificReframingShadowOutcome(
            report=report,
            prompts=tuple(prompts),
        )
