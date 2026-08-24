from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.hypothesis_action_contracts import (
    G1FindingScope,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


G1ApplicationAssertionKind = Literal[
    "central",
    "bridge",
    "prediction",
    "assumption",
]


G1PostRepairCheck = Literal[
    "compile_validate",
    "axis_fidelity",
    "axis_inference",
    "context_review",
    "internal_novelty",
    "external_novelty",
    "semantic_review",
]


_REQUIRED_POST_REPAIR_CHECKS = (
    "compile_validate",
    "axis_fidelity",
    "axis_inference",
    "context_review",
    "internal_novelty",
    "external_novelty",
    "semantic_review",
)


class G1ApplicationAssertionSource(
    StrictModel
):
    assertion_id: str = Field(
        min_length=1
    )

    assertion_kind: G1ApplicationAssertionKind

    assertion_text: str = Field(
        min_length=1
    )

    assertion_text_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    @model_validator(mode="after")
    def _text_hash_matches(
        self,
    ) -> "G1ApplicationAssertionSource":

        actual = hashlib.sha256(
            self.assertion_text.encode(
                "utf-8"
            )
        ).hexdigest()

        if (
            actual
            != self.assertion_text_sha256
        ):
            raise ValueError(
                "assertion_text_sha256 mismatch"
            )

        return self


class G1ScientificRepairConstraint(
    StrictModel
):
    directive_id: str = Field(
        min_length=1
    )

    action: Literal[
        "reframe"
    ] = "reframe"

    source_scope: G1FindingScope

    source_assertions: list[
        G1ApplicationAssertionSource
    ] = Field(
        min_length=1
    )

    finding_ref_ids: list[str] = Field(
        min_length=1
    )

    rationale: str = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def _scope_matches_assertions(
        self,
    ) -> "G1ScientificRepairConstraint":

        if self.source_scope.kind not in {
            "central",
            "bridge",
            "prediction",
            "assumption",
        }:
            raise ValueError(
                "scientific reframe requires "
                "assertion-level source scope"
            )

        source_ids = [
            row.assertion_id
            for row in self.source_assertions
        ]

        if (
            len(source_ids)
            != len(set(source_ids))
        ):
            raise ValueError(
                "duplicate source assertion ID"
            )

        if (
            sorted(source_ids)
            != sorted(
                self.source_scope
                .assertion_ids
            )
        ):
            raise ValueError(
                "source assertion IDs do not "
                "exactly match directive scope"
            )

        if any(
            row.assertion_kind
            != self.source_scope.kind
            for row in self.source_assertions
        ):
            raise ValueError(
                "source assertion kind does not "
                "match directive scope kind"
            )

        if (
            len(self.finding_ref_ids)
            != len(
                set(
                    self.finding_ref_ids
                )
            )
        ):
            raise ValueError(
                "duplicate finding_ref_ids"
            )

        return self


class G1NoveltyDispositionConstraint(
    StrictModel
):
    directive_id: str = Field(
        min_length=1
    )

    action: Literal[
        "downgrade"
    ] = "downgrade"

    finding_ref_ids: list[str] = Field(
        min_length=1
    )

    storage_target: Literal[
        "application_artifact"
    ] = "application_artifact"

    scientific_text_mutation: Literal[
        False
    ] = False

    rationale: str = Field(
        min_length=1
    )


class G1ApplicationPlan(
    StrictModel
):
    schema_version: Literal[
        "g1-application-plan-v1"
    ] = "g1-application-plan-v1"

    plan_id: str = Field(
        min_length=1
    )

    source_portfolio_id: str = Field(
        min_length=1
    )

    source_hypothesis_id: str = Field(
        min_length=1
    )

    source_decision_id: str = Field(
        min_length=1
    )

    source_card_sha256: str = Field(
        min_length=64,
        max_length=64,
    )

    mutation_owner: Literal[
        "hypothesis_draft_backend_repair"
    ] = "hypothesis_draft_backend_repair"

    max_scientific_repair_calls: Literal[
        1
    ] = 1

    pre_repair_axis_inference_review_required: Literal[
        True
    ] = True

    scientific_repair_constraints: list[
        G1ScientificRepairConstraint
    ] = Field(
        default_factory=list
    )

    novelty_disposition_constraints: list[
        G1NoveltyDispositionConstraint
    ] = Field(
        default_factory=list
    )

    required_post_repair_checks: list[
        G1PostRepairCheck
    ] = Field(
        default_factory=lambda:
            list(
                _REQUIRED_POST_REPAIR_CHECKS
            )
    )

    source_generation_mutated: Literal[
        False
    ] = False

    @model_validator(mode="after")
    def _plan_consistency(
        self,
    ) -> "G1ApplicationPlan":

        all_directive_ids = [
            row.directive_id
            for row
            in self.scientific_repair_constraints
        ] + [
            row.directive_id
            for row
            in self.novelty_disposition_constraints
        ]

        if (
            len(all_directive_ids)
            != len(
                set(
                    all_directive_ids
                )
            )
        ):
            raise ValueError(
                "duplicate directive ID "
                "across application constraints"
            )

        assertion_ids = [
            assertion.assertion_id
            for constraint
            in self.scientific_repair_constraints
            for assertion
            in constraint.source_assertions
        ]

        if (
            len(assertion_ids)
            != len(
                set(
                    assertion_ids
                )
            )
        ):
            raise ValueError(
                "multiple scientific repair "
                "constraints target the same "
                "source assertion"
            )

        checks = (
            self.required_post_repair_checks
        )

        if (
            len(checks)
            != len(set(checks))
        ):
            raise ValueError(
                "duplicate post-repair check"
            )

        if set(checks) != set(
            _REQUIRED_POST_REPAIR_CHECKS
        ):
            raise ValueError(
                "application plan must require "
                "the complete post-repair "
                "revalidation set"
            )

        return self


REQUIRED_G1_POST_REPAIR_CHECKS = (
    _REQUIRED_POST_REPAIR_CHECKS
)
