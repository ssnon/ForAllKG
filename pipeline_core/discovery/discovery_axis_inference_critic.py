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

        allowed_basis = set(
            allowed_axis_basis(axis)
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

            unknown_axis_basis = sorted(
                set(row.axis_basis)
                - allowed_basis
            )

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

        status = inference_review_status(
            draft.assertions
        )

        reason_codes: list[str] = []

        for row in draft.assertions:
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
            _canonical_json(draft),
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
                list(draft.assertions),
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
    ) -> None:
        self.backend = backend

        self.prompt_assembler = (
            prompt_assembler
            or DiscoveryAxisInferencePromptAssembler()
        )

        self.compiler = (
            compiler
            or AxisInferenceReviewCompiler()
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

        generation = self.backend.review(
            prompt
        )

        review = self.compiler.compile(
            context=context,
            axis=axis,
            card=card,
            prompt=prompt,
            draft=generation.draft,
        )

        return AxisInferenceCriticOutcome(
            prompt=prompt,
            generation=generation,
            review=review,
        )
