from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.discovery_axis_inference_contracts import (
    InferenceAction,
    InferenceSourceClass,
)
from pipeline_core.discovery.hypothesis_action_application_contracts import (
    G1ApplicationAssertionKind,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


G1UnifiedRequirementKind = Literal[
    "d1_repair",
    "g1_reframe",
    "g1_reframe_with_d1_guard",
    "d1_and_g1_repair",
]


_D1_REPAIR_ACTIONS = {
    "OPEN_DIRECTION",
    "REFRAME",
    "REMOVE",
}

_D1_KEEP_ACTIONS = {
    "KEEP",
    "KEEP_HYPOTHETICAL",
}


class G1RepairInputBinding(StrictModel):
    schema_version: Literal[
        "g1-repair-input-binding-v1"
    ] = "g1-repair-input-binding-v1"

    binding_id: str = Field(
        min_length=1
    )

    application_plan_id: str = Field(
        min_length=1
    )

    source_portfolio_id: str = Field(
        min_length=1
    )

    original_hypothesis_id: str = Field(
        min_length=1
    )

    final_hypothesis_id: str = Field(
        min_length=1
    )

    authoritative_draft_local_id: str = Field(
        min_length=1
    )

    refinement_report_id: str = Field(
        min_length=1
    )

    axis_id: str = Field(
        min_length=1
    )

    context_id: str = Field(
        min_length=1
    )

    context_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    source_card_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    scientific_payload_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    mutation_owner: Literal[
        "hypothesis_draft_backend_repair"
    ] = "hypothesis_draft_backend_repair"

    max_scientific_repair_calls: Literal[
        1
    ] = 1

    source_generation_mutated: Literal[
        False
    ] = False


class G1UnifiedRepairRequirement(StrictModel):
    source_assertion_id: str = Field(
        min_length=1
    )

    assertion_kind: G1ApplicationAssertionKind

    source_assertion_text: str = Field(
        min_length=1
    )

    source_assertion_text_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    d1_source_class: InferenceSourceClass | None = None
    d1_action: InferenceAction | None = None
    d1_rationale: str | None = None

    g1_directive_ids: list[str] = Field(
        default_factory=list
    )

    g1_rationales: list[str] = Field(
        default_factory=list
    )

    effective_requirement: G1UnifiedRequirementKind

    @model_validator(mode="after")
    def _requirement_consistency(
        self,
    ) -> "G1UnifiedRepairRequirement":

        actual = hashlib.sha256(
            self.source_assertion_text.encode(
                "utf-8"
            )
        ).hexdigest()

        if (
            actual
            != self.source_assertion_text_sha256
        ):
            raise ValueError(
                "source assertion SHA mismatch"
            )

        if (
            len(self.g1_directive_ids)
            != len(set(self.g1_directive_ids))
        ):
            raise ValueError(
                "duplicate G1 directive IDs"
            )

        if (
            len(self.g1_directive_ids)
            != len(self.g1_rationales)
        ):
            raise ValueError(
                "G1 directive/rationale count mismatch"
            )

        has_g1 = bool(
            self.g1_directive_ids
        )

        has_d1 = (
            self.d1_action is not None
        )

        if has_d1 and (
            self.d1_source_class is None
            or not (
                self.d1_rationale
                or ""
            ).strip()
        ):
            raise ValueError(
                "D1 action requires source class "
                "and rationale"
            )

        if (
            self.assertion_kind
            in {"bridge", "assumption"}
            and has_d1
        ):
            raise ValueError(
                "D1 inference critic does not "
                "review bridge/assumption assertions"
            )

        if (
            self.d1_action
            in _D1_REPAIR_ACTIONS
            and has_g1
        ):
            expected = "d1_and_g1_repair"

        elif (
            self.d1_action
            in _D1_REPAIR_ACTIONS
        ):
            expected = "d1_repair"

        elif (
            self.d1_action
            in _D1_KEEP_ACTIONS
            and has_g1
        ):
            expected = (
                "g1_reframe_with_d1_guard"
            )

        elif (
            not has_d1
            and has_g1
        ):
            expected = "g1_reframe"

        else:
            raise ValueError(
                "unified requirement has no "
                "actionable repair source"
            )

        if (
            self.effective_requirement
            != expected
        ):
            raise ValueError(
                "effective requirement mismatch: "
                f"expected={expected!r}, "
                f"actual={self.effective_requirement!r}"
            )

        return self


class G1UnifiedRepairFeedback(StrictModel):
    schema_version: Literal[
        "g1-unified-repair-feedback-v1"
    ] = "g1-unified-repair-feedback-v1"

    feedback_id: str = Field(
        min_length=1
    )

    binding: G1RepairInputBinding

    pre_repair_d1_review_id: str = Field(
        min_length=1
    )

    pre_repair_d1_status: Literal[
        "pass",
        "reframe_required",
    ]

    requirements: list[
        G1UnifiedRepairRequirement
    ] = Field(
        min_length=1
    )

    d1_preserve_assertion_ids: list[str] = Field(
        default_factory=list
    )

    novelty_metadata_directive_ids: list[str] = Field(
        default_factory=list
    )

    scientific_mutation_owner: Literal[
        "hypothesis_draft_backend_repair"
    ] = "hypothesis_draft_backend_repair"

    max_scientific_repair_calls: Literal[
        1
    ] = 1

    novelty_metadata_is_scientific_instruction: Literal[
        False
    ] = False

    source_generation_mutated: Literal[
        False
    ] = False

    @model_validator(mode="after")
    def _feedback_consistency(
        self,
    ) -> "G1UnifiedRepairFeedback":

        ids = [
            row.source_assertion_id
            for row in self.requirements
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "duplicate unified repair "
                "assertion ID"
            )

        preserve = (
            self.d1_preserve_assertion_ids
        )

        if len(preserve) != len(set(preserve)):
            raise ValueError(
                "duplicate D1 preserve assertion ID"
            )

        overlap = (
            set(ids)
            & set(preserve)
        )

        if overlap:
            raise ValueError(
                "repair/preserve assertion overlap: "
                + repr(sorted(overlap))
            )

        if (
            len(
                self.novelty_metadata_directive_ids
            )
            != len(
                set(
                    self.novelty_metadata_directive_ids
                )
            )
        ):
            raise ValueError(
                "duplicate novelty metadata "
                "directive ID"
            )

        has_d1_repair = any(
            row.d1_action
            in _D1_REPAIR_ACTIONS
            for row in self.requirements
        )

        if (
            self.pre_repair_d1_status
            == "reframe_required"
            and not has_d1_repair
        ):
            raise ValueError(
                "D1 reframe_required status "
                "must contribute a D1 repair "
                "requirement"
            )

        if (
            self.pre_repair_d1_status
            == "pass"
            and has_d1_repair
        ):
            raise ValueError(
                "D1 pass status cannot contain "
                "D1 repair requirements"
            )

        if (
            self.scientific_mutation_owner
            != self.binding.mutation_owner
        ):
            raise ValueError(
                "mutation-owner mismatch"
            )

        if (
            self.max_scientific_repair_calls
            != self.binding
            .max_scientific_repair_calls
        ):
            raise ValueError(
                "repair-call budget mismatch"
            )

        return self
