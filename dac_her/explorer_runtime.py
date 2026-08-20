from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from dac_her.explorer_compiler import (
    CompileIssue,
    ExplorationCompileError,
    ExplorationReportCompiler,
)
from pipeline_core.discovery.explorer_contracts import ExplorationReport, GraphExplorerPacket
from pipeline_core.discovery.explorer_draft import ExplorationDraft
from dac_her.explorer_llm import DraftGeneration, ExplorationDraftBackend
from dac_her.explorer_normalization import (
    ExplorerDraftNormalizer,
    ExplorerNormalizationAudit,
)
from pipeline_core.discovery.explorer_prompt import ExplorerPrompt, ExplorerPromptAssembler
from pipeline_core.discovery.explorer_run_record import GraphExplorerRunRecord
from dac_her.explorer_validation import ExplorationReportValidator, ExplorationValidationResult


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:length]}"


@dataclass(frozen=True)
class GraphExplorerRunOutcome:
    accepted_report: ExplorationReport | None
    last_report: ExplorationReport | None
    validation: ExplorationValidationResult | None
    final_draft: ExplorationDraft | None
    normalized_draft: ExplorationDraft | None
    normalization_audit: ExplorerNormalizationAudit
    draft_history: tuple[ExplorationDraft, ...]
    compile_issues: tuple[CompileIssue, ...]
    run_record: GraphExplorerRunRecord
    prompt: ExplorerPrompt

    @property
    def accepted(self) -> bool:
        return self.accepted_report is not None and self.run_record.final_validation_passed


class CompiledGraphExplorerBackend:
    """One-shot GraphExplorerBackend-compatible adapter.

    It deliberately does not perform repair.  Use GraphExplorerAgentRuntime for
    production execution with validation and bounded repair.
    """

    def __init__(
        self,
        draft_backend: ExplorationDraftBackend,
        *,
        prompt_assembler: ExplorerPromptAssembler | None = None,
        compiler: ExplorationReportCompiler | None = None,
    ) -> None:
        self.draft_backend = draft_backend
        self.prompt_assembler = prompt_assembler or ExplorerPromptAssembler()
        self.compiler = compiler or ExplorationReportCompiler()

    def explore(self, packet: GraphExplorerPacket) -> ExplorationReport:
        prompt = self.prompt_assembler.build(packet)
        generation = self.draft_backend.generate(prompt)
        return self.compiler.compile(packet, generation.draft)


class GraphExplorerAgentRuntime:
    def __init__(
        self,
        draft_backend: ExplorationDraftBackend,
        *,
        prompt_assembler: ExplorerPromptAssembler | None = None,
        compiler: ExplorationReportCompiler | None = None,
        validator: ExplorationReportValidator | None = None,
        normalizer: ExplorerDraftNormalizer | None = None,
        max_repairs: int = 1,
    ) -> None:
        if max_repairs not in {0, 1}:
            raise ValueError("Graph Explorer v2.5.1 supports max_repairs of 0 or 1 only.")
        self.draft_backend = draft_backend
        self.prompt_assembler = prompt_assembler or ExplorerPromptAssembler()
        self.compiler = compiler or ExplorationReportCompiler()
        self.validator = validator or ExplorationReportValidator()
        self.normalizer = normalizer or ExplorerDraftNormalizer()
        self.max_repairs = int(max_repairs)

    def run(self, packet: GraphExplorerPacket) -> GraphExplorerRunOutcome:
        prompt = self.prompt_assembler.build(packet)
        started = time.perf_counter()
        generations: list[DraftGeneration] = []
        drafts: list[ExplorationDraft] = []
        compile_issues: list[CompileIssue] = []
        last_report: ExplorationReport | None = None
        validation: ExplorationValidationResult | None = None
        failure_stage = "none"
        normalization_audit = ExplorerNormalizationAudit(
            domain_profile_id=packet.domain_profile_id,
        )
        normalized_draft: ExplorationDraft | None = None

        try:
            generation = self.draft_backend.generate(prompt)
        except Exception:
            elapsed = time.perf_counter() - started
            record = self._run_record(
                packet=packet,
                prompt=prompt,
                report=None,
                validation=None,
                generations=generations,
                repair_attempts=0,
                compile_issues=compile_issues,
                normalization_audit=normalization_audit,
                failure_stage="generation",
                elapsed_seconds=elapsed,
            )
            raise

        generations.append(generation)
        drafts.append(generation.draft)
        current_draft = generation.draft
        repair_attempts = 0

        for cycle in range(self.max_repairs + 1):
            try:
                last_report = self.compiler.compile(packet, current_draft)
                compile_issues = []
            except ExplorationCompileError as exc:
                last_report = None
                validation = None
                compile_issues = list(exc.issues)
                failure_stage = "compile"
                if cycle >= self.max_repairs:
                    break
                feedback = self.prompt_assembler.repair_feedback(
                    previous_draft=current_draft,
                    issues=compile_issues,
                )
                repaired = self.draft_backend.repair(prompt, current_draft, feedback)
                generations.append(repaired)
                drafts.append(repaired.draft)
                current_draft = repaired.draft
                repair_attempts += 1
                continue

            validation = self.validator.validate(packet, last_report)
            if validation.passes:
                failure_stage = "none"
                break

            failure_stage = "validation"
            if cycle >= self.max_repairs:
                break
            feedback = self.prompt_assembler.repair_feedback(
                previous_draft=current_draft,
                issues=validation.issues,
            )
            repaired = self.draft_backend.repair(prompt, current_draft, feedback)
            generations.append(repaired)
            drafts.append(repaired.draft)
            current_draft = repaired.draft
            repair_attempts += 1

        accepted_report = last_report if validation is not None and validation.passes else None

        # After bounded LLM repair is exhausted, allow only deterministic
        # one-way weakening/removal. The normalizer never rewrites scientific
        # text or evidence references.
        if accepted_report is None:
            normalized = self.normalizer.normalize(
                packet,
                current_draft,
            )
            normalization_audit = normalized.audit
            if normalized.audit.applied:
                normalized_draft = normalized.draft
                try:
                    last_report = self.compiler.compile(
                        packet,
                        normalized_draft,
                    )
                    compile_issues = []
                    validation = self.validator.validate(
                        packet,
                        last_report,
                    )
                    if validation.passes:
                        accepted_report = last_report
                        failure_stage = "none"
                    else:
                        failure_stage = "validation"
                except ExplorationCompileError as exc:
                    last_report = None
                    validation = None
                    compile_issues = list(exc.issues)
                    failure_stage = "compile"

        elapsed = time.perf_counter() - started
        record = self._run_record(
            packet=packet,
            prompt=prompt,
            report=last_report,
            validation=validation,
            generations=generations,
            repair_attempts=repair_attempts,
            compile_issues=compile_issues,
            normalization_audit=normalization_audit,
            failure_stage=failure_stage if accepted_report is None else "none",
            elapsed_seconds=elapsed,
        )
        return GraphExplorerRunOutcome(
            accepted_report=accepted_report,
            last_report=last_report,
            validation=validation,
            final_draft=(
                normalized_draft
                if normalized_draft is not None
                else current_draft
            ),
            normalized_draft=normalized_draft,
            normalization_audit=normalization_audit,
            draft_history=tuple(drafts),
            compile_issues=tuple(compile_issues),
            run_record=record,
            prompt=prompt,
        )

    def _run_record(
        self,
        *,
        packet: GraphExplorerPacket,
        prompt: ExplorerPrompt,
        report: ExplorationReport | None,
        validation: ExplorationValidationResult | None,
        generations: list[DraftGeneration],
        repair_attempts: int,
        compile_issues: list[CompileIssue],
        normalization_audit: ExplorerNormalizationAudit,
        failure_stage: str,
        elapsed_seconds: float,
    ) -> GraphExplorerRunRecord:
        report_sha = (
            _sha256_json(report.model_dump(mode="json"))
            if report is not None
            else None
        )
        input_tokens_values = [g.input_tokens for g in generations if g.input_tokens is not None]
        output_tokens_values = [g.output_tokens for g in generations if g.output_tokens is not None]
        backend_name = str(getattr(self.draft_backend, "backend_name", type(self.draft_backend).__name__))
        model_name = str(getattr(self.draft_backend, "model_name", "unknown"))
        temperature_raw = getattr(self.draft_backend, "temperature", None)
        temperature = float(temperature_raw) if temperature_raw is not None else None
        run_id = _stable_id(
            "explorer_run",
            packet.packet_sha256,
            prompt.prompt_sha256,
            backend_name,
            model_name,
            report_sha or "rejected",
            len(generations),
        )
        return GraphExplorerRunRecord(
            run_id=run_id,
            packet_id=packet.packet_id,
            packet_sha256=packet.packet_sha256,
            report_id=(report.report_id if report is not None else None),
            report_sha256=report_sha,
            prompt_version=prompt.prompt_version,
            prompt_sha256=prompt.prompt_sha256,
            backend=backend_name,
            model=model_name,
            generation_attempts=len(generations),
            repair_attempts=repair_attempts,
            final_validation_passed=bool(validation and validation.passes),
            failure_stage=failure_stage,  # type: ignore[arg-type]
            validation_errors=(validation.errors if validation is not None else 0),
            validation_warnings=(validation.warnings if validation is not None else 0),
            compile_issue_count=len(compile_issues),
            normalization_applied=normalization_audit.applied,
            normalization_action_count=normalization_audit.action_count,
            normalization_blocked_count=normalization_audit.blocked_count,
            input_tokens=(sum(input_tokens_values) if input_tokens_values else None),
            output_tokens=(sum(output_tokens_values) if output_tokens_values else None),
            elapsed_seconds=elapsed_seconds,
            temperature=temperature,
            backend_mode=(str(getattr(self.draft_backend, "instructor_mode", "")) or None),
            base_url=(str(getattr(self.draft_backend, "base_url", "")) or None),
            parse_retries=(
                int(getattr(self.draft_backend, "parse_retries"))
                if getattr(self.draft_backend, "parse_retries", None) is not None
                else None
            ),
            max_repairs=self.max_repairs,
        )
