from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxis,
)
from pipeline_core.discovery.discovery_axis_inference_contracts import (
    AxisInferenceReview,
    AxisInferenceReviewDraft,
    allowed_inference_actions,
    inference_review_status,
)
from pipeline_core.discovery.discovery_axis_inference_llm import (
    AxisInferenceBackend,
    AxisInferenceGeneration,
)
from pipeline_core.discovery.discovery_axis_inference_prompt import (
    AxisInferencePrompt,
    DiscoveryAxisInferencePromptAssembler,
    allowed_axis_basis,
    expected_assertions,
    resolve_axis_basis_reference,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
    HypothesisContext,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(
            mode="json"
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )


class AxisInferenceReviewValidationError(
    ValueError
):
    def __init__(
        self,
        issues: list[str],
    ) -> None:
        self.issues = issues
        super().__init__(
            "; ".join(issues)
        )


_INFERENCE_CONTRACT_REPAIR_MARKERS = (
    "missing inference assertions:",
    "unknown inference assertions:",
    "assertion_kind does not match source hypothesis",
    "assertion_text does not match source hypothesis",
    "unknown or non-selected grounded statement IDs:",
    "unknown axis basis:",
    "G_GROUNDED requires grounded_statement_ids",
    "A_AXIS requires axis_basis",
    "S_BOUNDED_SYNTHESIS requires grounded_statement_ids",
    "S_BOUNDED_SYNTHESIS requires axis_basis",
    "inference source/action mismatch:",
)


def is_inference_contract_repair_issue(
    issue: object,
) -> bool:
    text = str(issue)

    return any(
        marker in text
        for marker
        in _INFERENCE_CONTRACT_REPAIR_MARKERS
    )


def canonicalize_inference_assertion_ids(
    *,
    card: HypothesisCard,
    draft: AxisInferenceReviewDraft,
) -> AxisInferenceReviewDraft:
    """Repair only an unambiguous orchestration pointer.

    If an unknown assertion_id has exactly one authoritative assertion
    with the SAME assertion_kind and EXACT assertion_text, replace only
    the ID. No fuzzy matching and no scientific text rewrite.
    """

    expected_rows = list(
        expected_assertions(
            card
        )
    )

    expected_by_id = {
        row["assertion_id"]: row
        for row in expected_rows
    }

    used_authoritative_ids = {
        row.assertion_id
        for row in draft.assertions
        if row.assertion_id
        in expected_by_id
    }

    canonical_rows = []

    for row in draft.assertions:

        if row.assertion_id in expected_by_id:
            canonical_rows.append(
                row
            )
            continue

        candidates = [
            expected_row
            for expected_row
            in expected_rows
            if (
                expected_row[
                    "assertion_kind"
                ]
                == row.assertion_kind
                and expected_row[
                    "assertion_text"
                ]
                == row.assertion_text
                and expected_row[
                    "assertion_id"
                ]
                not in used_authoritative_ids
            )
        ]

        if len(candidates) == 1:
            authoritative_id = (
                candidates[0][
                    "assertion_id"
                ]
            )

            used_authoritative_ids.add(
                authoritative_id
            )

            row = row.model_copy(
                update={
                    "assertion_id":
                        authoritative_id,
                }
            )

        canonical_rows.append(
            row
        )

    return draft.model_copy(
        update={
            "assertions":
                canonical_rows,
        }
    )


class AxisInferenceReviewCompiler:
    """Bind the LLM review to exact hypothesis/axis provenance."""

    def compile(
        self,
        *,
        context: HypothesisContext,
        axis: DiscoveryAxis,
        card: HypothesisCard,
        prompt: AxisInferencePrompt,
        draft: AxisInferenceReviewDraft,
    ) -> AxisInferenceReview:
        issues: list[str] = []

        if card.source_context_id != context.context_id:
            issues.append(
                "hypothesis/context ID mismatch"
            )

        if (
            card.source_context_sha256
            != context.context_sha256
        ):
            issues.append(
                "hypothesis/context SHA mismatch"
            )

        expected = {
            row["assertion_id"]: row
            for row in expected_assertions(card)
        }

        draft = (
            canonicalize_inference_assertion_ids(
                card=card,
                draft=draft,
            )
        )

        actual = {
            row.assertion_id: row
            for row in draft.assertions
        }

        missing = sorted(
            set(expected) - set(actual)
        )

        extra = sorted(
            set(actual) - set(expected)
        )

        if missing:
            issues.append(
                "missing inference assertions: "
                f"{missing}"
            )

        if extra:
            issues.append(
                "unknown inference assertions: "
                f"{extra}"
            )

        allowed_statement_ids = set(
            card.premise_statement_ids
        )

        # Human-readable axis strings remain authoritative. The
        # LLM-facing draft may refer to them through stable basis IDs,
        # which are resolved deterministically below.
        allowed_basis = set(
            allowed_axis_basis(
                axis
            )
        )

        for assertion_id, row in actual.items():
            expected_row = expected.get(
                assertion_id
            )

            if expected_row is None:
                continue

            if (
                row.assertion_kind
                != expected_row[
                    "assertion_kind"
                ]
            ):
                issues.append(
                    f"{assertion_id}: assertion_kind "
                    "does not match source hypothesis"
                )

            if (
                row.assertion_text
                != expected_row[
                    "assertion_text"
                ]
            ):
                issues.append(
                    f"{assertion_id}: assertion_text "
                    "does not match source hypothesis"
                )

            allowed_actions = (
                allowed_inference_actions(
                    row.source_class
                )
            )

            if row.action not in allowed_actions:
                issues.append(
                    f"{assertion_id}: "
                    "inference source/action mismatch: "
                    f"source_class={row.source_class!r}, "
                    f"action={row.action!r}, "
                    f"allowed={sorted(allowed_actions)!r}"
                )

            unknown_statements = sorted(
                set(
                    row.grounded_statement_ids
                )
                - allowed_statement_ids
            )

            if unknown_statements:
                issues.append(
                    f"{assertion_id}: unknown or "
                    "non-selected grounded statement IDs: "
                    f"{unknown_statements}"
                )

            unknown_axis_basis = sorted({
                value
                for value
                in row.axis_basis
                if (
                    resolve_axis_basis_reference(
                        axis,
                        value,
                    )
                    is None
                )
            })

            if unknown_axis_basis:
                issues.append(
                    f"{assertion_id}: unknown axis basis: "
                    f"{unknown_axis_basis}"
                )

            if (
                row.source_class
                == "G_GROUNDED"
                and not row.grounded_statement_ids
            ):
                issues.append(
                    f"{assertion_id}: G_GROUNDED "
                    "requires grounded_statement_ids"
                )

            if (
                row.source_class
                == "A_AXIS"
                and not row.axis_basis
            ):
                issues.append(
                    f"{assertion_id}: A_AXIS "
                    "requires axis_basis"
                )

            if (
                row.source_class
                == "S_BOUNDED_SYNTHESIS"
            ):
                if not row.grounded_statement_ids:
                    issues.append(
                        f"{assertion_id}: "
                        "S_BOUNDED_SYNTHESIS requires "
                        "grounded_statement_ids"
                    )

                if not row.axis_basis:
                    issues.append(
                        f"{assertion_id}: "
                        "S_BOUNDED_SYNTHESIS requires "
                        "axis_basis"
                    )

        if issues:
            raise (
                AxisInferenceReviewValidationError(
                    issues
                )
            )

        canonical_assertions = []

        for row in draft.assertions:
            resolved_basis = []

            for value in row.axis_basis:
                resolved = (
                    resolve_axis_basis_reference(
                        axis,
                        value,
                    )
                )

                # Unknown values cannot reach this branch because the
                # strict validation above has already rejected them.
                if resolved is None:
                    raise AssertionError(
                        "validated axis basis failed resolution"
                    )

                if resolved not in resolved_basis:
                    resolved_basis.append(
                        resolved
                    )

            canonical_assertions.append(
                row.model_copy(
                    update={
                        "axis_basis":
                            resolved_basis,
                    }
                )
            )

        canonical_draft = (
            draft.model_copy(
                update={
                    "assertions":
                        canonical_assertions,
                }
            )
        )

        status = inference_review_status(
            canonical_draft.assertions
        )

        reason_codes: list[str] = []

        for row in canonical_draft.assertions:
            if (
                row.source_class
                == "X_UNSUPPORTED_SPECIFICITY"
            ):
                reason_codes.append(
                    "unsupported_specificity"
                )

            if row.action == "OPEN_DIRECTION":
                reason_codes.append(
                    "open_direction_required"
                )

            elif row.action == "REFRAME":
                reason_codes.append(
                    "reframe_required"
                )

            elif row.action == "REMOVE":
                reason_codes.append(
                    "removal_required"
                )

        reason_codes = sorted(
            set(reason_codes)
        )

        review_id = _stable_id(
            "axis_inference_review",
            context.context_sha256,
            axis.axis_id,
            card.hypothesis_id,
            prompt.prompt_sha256,
            _canonical_json(
                canonical_draft
            ),
        )

        return AxisInferenceReview(
            review_id=review_id,
            axis_id=axis.axis_id,
            hypothesis_id=
                card.hypothesis_id,
            source_context_id=
                context.context_id,
            source_context_sha256=
                context.context_sha256,
            critic_prompt_version=
                prompt.prompt_version,
            critic_prompt_sha256=
                prompt.prompt_sha256,
            status=status,
            assertions=
                list(
                    canonical_draft.assertions
                ),
            overall_risk=
                draft.overall_risk,
            reason_codes=
                reason_codes,
            interpretation=
                draft.interpretation,
        )


@dataclass(frozen=True)
class AxisInferenceCriticOutcome:
    prompt: AxisInferencePrompt
    generation: AxisInferenceGeneration
    review: AxisInferenceReview
    validation_repair_attempts: int = 0


class DiscoveryAxisInferenceCritic:
    """Standalone inference-strength critic.

    D1.5 deliberately does not modify synthesis selection or perform
    hypothesis repair. Runtime actionability is added in D1.6.
    """

    def __init__(
        self,
        backend: AxisInferenceBackend,
        *,
        prompt_assembler:
            DiscoveryAxisInferencePromptAssembler
            | None = None,
        compiler:
            AxisInferenceReviewCompiler
            | None = None,
        max_validation_repairs: int = 1,
    ) -> None:
        if max_validation_repairs not in {
            0,
            1,
        }:
            raise ValueError(
                "Axis inference contract validation supports "
                "max_validation_repairs of 0 or 1 only."
            )

        self.backend = backend

        self.prompt_assembler = (
            prompt_assembler
            or DiscoveryAxisInferencePromptAssembler()
        )

        self.compiler = (
            compiler
            or AxisInferenceReviewCompiler()
        )

        self.max_validation_repairs = int(
            max_validation_repairs
        )

    def review(
        self,
        context: HypothesisContext,
        axis: DiscoveryAxis,
        card: HypothesisCard,
    ) -> AxisInferenceCriticOutcome:
        prompt = self.prompt_assembler.build(
            context,
            axis,
            card,
        )

        active_prompt = prompt

        generation = self.backend.review(
            active_prompt
        )

        validation_repair_attempts = 0

        try:
            review = self.compiler.compile(
                context=context,
                axis=axis,
                card=card,
                prompt=active_prompt,
                draft=generation.draft,
            )

        except (
            AxisInferenceReviewValidationError
        ) as exc:

            repairable = (
                bool(exc.issues)
                and all(
                    is_inference_contract_repair_issue(
                        issue
                    )
                    for issue
                    in exc.issues
                )
            )

            if (
                not repairable
                or self.max_validation_repairs
                < 1
            ):
                raise

            active_prompt = (
                self.prompt_assembler
                .build_validation_repair(
                    original_prompt=prompt,
                    previous_draft=
                        generation.draft,
                    issues=list(
                        exc.issues
                    ),
                )
            )

            generation = self.backend.review(
                active_prompt
            )

            validation_repair_attempts = 1

            # Same strict compiler. A second failure propagates
            # fail-closed; there is no repair loop.
            review = self.compiler.compile(
                context=context,
                axis=axis,
                card=card,
                prompt=active_prompt,
                draft=generation.draft,
            )

        return AxisInferenceCriticOutcome(
            prompt=active_prompt,
            generation=generation,
            review=review,
            validation_repair_attempts=
                validation_repair_attempts,
        )
