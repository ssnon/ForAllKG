from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


# ----------------------------------------------------------------------
# Finding provenance
#
# These are generic discovery-layer sources. Domain-specific statuses
# such as ROLE_MISMATCH must NOT appear here.
# ----------------------------------------------------------------------

G1FindingSourceKind = Literal[
    "axis_inference",
    "context_review",
    "semantic_review",
    "internal_novelty",
    "external_novelty",
]


# ----------------------------------------------------------------------
# Authority is deliberately separate from severity/status.
#
# informational
#   Provenance/context only; cannot require lifecycle action.
#
# advisory
#   May justify KEEP_WITH_WARNING or contribute to UNRESOLVED,
#   but cannot by itself force repair/rejection.
#
# actionable
#   Requires a bounded local repair before clean acceptance.
#
# terminal_candidate
#   May support whole-hypothesis rejection under a later G1 policy.
#   "candidate" is intentional: scope still matters.
# ----------------------------------------------------------------------

G1FindingAuthority = Literal[
    "informational",
    "advisory",
    "actionable",
    "terminal_candidate",
]


# ----------------------------------------------------------------------
# Scientific scope
# ----------------------------------------------------------------------

G1FindingScopeKind = Literal[
    "portfolio",
    "hypothesis",
    "central",
    "bridge",
    "prediction",
    "assumption",
    "external_novelty_claim",
]


class G1FindingScope(StrictModel):
    """Concrete scientific location of one finding.

    hypothesis_ids identify the hypothesis/hypotheses involved.

    assertion_ids are opaque upstream assertion identifiers. Generic
    core does not assume a domain-specific or producer-specific syntax.
    """

    kind: G1FindingScopeKind

    hypothesis_ids: list[str] = Field(
        min_length=1
    )

    assertion_ids: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _scope_shape(
        self,
    ) -> "G1FindingScope":
        if (
            len(self.hypothesis_ids)
            != len(set(self.hypothesis_ids))
        ):
            raise ValueError(
                "duplicate hypothesis_ids in G1 scope"
            )

        if (
            len(self.assertion_ids)
            != len(set(self.assertion_ids))
        ):
            raise ValueError(
                "duplicate assertion_ids in G1 scope"
            )

        if self.kind in {
            "portfolio",
            "hypothesis",
        }:
            if self.assertion_ids:
                raise ValueError(
                    f"{self.kind} scope must not carry "
                    "assertion_ids"
                )

            return self

        if not self.assertion_ids:
            raise ValueError(
                f"{self.kind} scope requires assertion_ids"
            )

        return self


class G1FindingRef(StrictModel):
    """Normalized, final-target-bound reference to an upstream finding.

    source_scope preserves what the producer actually reviewed.

    target_scope identifies where the finding lands in the target
    portfolio generation. They MUST remain distinct when IDs changed.

    Cross-generation binding is never implicit.
    """

    finding_ref_id: str = Field(
        min_length=1
    )

    source_kind: G1FindingSourceKind

    source_artifact_id: str = Field(
        min_length=1
    )

    source_finding_id: str = Field(
        min_length=1
    )

    source_status: str = Field(
        min_length=1
    )

    # Producer-specific metadata required by later lifecycle policy
    # without leaking producer semantics into generic G1 contracts.
    # Examples: external novelty claim importance or assessment level.
    source_attributes: dict[str, str] = Field(
        default_factory=dict
    )

    authority: G1FindingAuthority

    source_portfolio_id: str | None = None

    source_hypothesis_ids: list[str] = Field(
        min_length=1
    )

    source_scope: G1FindingScope

    target_portfolio_id: str = Field(
        min_length=1
    )

    target_hypothesis_id: str = Field(
        min_length=1
    )

    target_scope: G1FindingScope

    lineage_ref_ids: list[str] = Field(
        default_factory=list
    )

    rationale: str = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def _provenance_binding(
        self,
    ) -> "G1FindingRef":
        if any(
            not str(key).strip()
            or not str(value).strip()
            for key, value
            in self.source_attributes.items()
        ):
            raise ValueError(
                "source_attributes keys/values "
                "must be non-empty strings"
            )

        if (
            len(self.source_hypothesis_ids)
            != len(
                set(
                    self.source_hypothesis_ids
                )
            )
        ):
            raise ValueError(
                "duplicate source_hypothesis_ids"
            )

        if (
            set(
                self.source_scope.hypothesis_ids
            )
            != set(
                self.source_hypothesis_ids
            )
        ):
            raise ValueError(
                "source_scope hypothesis_ids must "
                "match source_hypothesis_ids"
            )

        if (
            self.target_scope.hypothesis_ids
            != [self.target_hypothesis_id]
        ):
            raise ValueError(
                "target_scope must bind exactly one "
                "target_hypothesis_id"
            )

        if (
            len(self.lineage_ref_ids)
            != len(
                set(
                    self.lineage_ref_ids
                )
            )
        ):
            raise ValueError(
                "duplicate lineage_ref_ids"
            )

        crosses_portfolio_generation = (
            self.source_portfolio_id
            is not None
            and self.source_portfolio_id
            != self.target_portfolio_id
        )

        target_not_directly_reviewed = (
            self.target_hypothesis_id
            not in self.source_hypothesis_ids
        )

        if (
            crosses_portfolio_generation
            or target_not_directly_reviewed
        ) and not self.lineage_ref_ids:
            raise ValueError(
                "cross-generation or rebound finding "
                "requires explicit lineage_ref_ids"
            )

        return self


# ----------------------------------------------------------------------
# Lifecycle decision vocabulary
# ----------------------------------------------------------------------

G1Disposition = Literal[
    "keep",
    "keep_with_warning",
    "repair_required",
    "reject",
    "unresolved",
]


G1LocalAction = Literal[
    "reframe",
    "downgrade",
    "remove_assertion",
]


class G1ActionDirective(StrictModel):
    """One bounded mutation requested by a G1 decision.

    This is declarative only. G1.1 does not mutate hypotheses.
    """

    directive_id: str = Field(
        min_length=1
    )

    action: G1LocalAction

    target_scope: G1FindingScope

    finding_ref_ids: list[str] = Field(
        min_length=1
    )

    rationale: str = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def _directive_shape(
        self,
    ) -> "G1ActionDirective":
        if (
            len(self.finding_ref_ids)
            != len(
                set(
                    self.finding_ref_ids
                )
            )
        ):
            raise ValueError(
                "duplicate finding_ref_ids in directive"
            )

        if (
            self.action
            == "remove_assertion"
            and self.target_scope.kind
            not in {
                "prediction",
                "assumption",
            }
        ):
            raise ValueError(
                "remove_assertion is limited to "
                "prediction/assumption scope in G1 v1"
            )

        return self


class G1ActionDecision(StrictModel):
    """Final-target-bound G1 adjudication contract.

    It records policy output but does not itself perform a mutation.
    A later G1 application/runtime artifact will own execution.
    """

    schema_version: Literal[
        "g1-action-decision-v1"
    ] = "g1-action-decision-v1"

    contract_version: Literal[
        "g1-action-contract-v1"
    ] = "g1-action-contract-v1"

    decision_id: str = Field(
        min_length=1
    )

    target_portfolio_id: str = Field(
        min_length=1
    )

    target_hypothesis_id: str = Field(
        min_length=1
    )

    findings: list[
        G1FindingRef
    ] = Field(
        default_factory=list
    )

    directives: list[
        G1ActionDirective
    ] = Field(
        default_factory=list
    )

    disposition: G1Disposition

    reason_codes: list[str] = Field(
        default_factory=list
    )

    interpretation: str = Field(
        min_length=1
    )

    mutation_applied: Literal[
        False
    ] = False

    @model_validator(mode="after")
    def _decision_consistency(
        self,
    ) -> "G1ActionDecision":
        finding_ids = [
            row.finding_ref_id
            for row in self.findings
        ]

        if (
            len(finding_ids)
            != len(set(finding_ids))
        ):
            raise ValueError(
                "duplicate finding_ref_id in G1 decision"
            )

        directive_ids = [
            row.directive_id
            for row in self.directives
        ]

        if (
            len(directive_ids)
            != len(set(directive_ids))
        ):
            raise ValueError(
                "duplicate directive_id in G1 decision"
            )

        available = set(
            finding_ids
        )

        for finding in self.findings:
            if (
                finding.target_portfolio_id
                != self.target_portfolio_id
            ):
                raise ValueError(
                    "finding target_portfolio_id "
                    "does not match decision"
                )

            if (
                finding.target_hypothesis_id
                != self.target_hypothesis_id
            ):
                raise ValueError(
                    "finding target_hypothesis_id "
                    "does not match decision"
                )

        for directive in self.directives:
            unknown = sorted(
                set(
                    directive.finding_ref_ids
                )
                - available
            )

            if unknown:
                raise ValueError(
                    "directive references unknown "
                    f"finding_ref_ids: {unknown}"
                )

            if (
                directive.target_scope.hypothesis_ids
                != [self.target_hypothesis_id]
            ):
                raise ValueError(
                    "directive target_scope must bind "
                    "exactly the decision target hypothesis"
                )

        authorities = {
            row.authority
            for row in self.findings
        }

        # Local repair directives exist only for a repair decision.
        if (
            self.disposition
            == "repair_required"
        ):
            if not self.directives:
                raise ValueError(
                    "repair_required decision requires "
                    "at least one local directive"
                )

            if (
                "actionable"
                not in authorities
            ):
                raise ValueError(
                    "repair_required decision requires "
                    "at least one actionable finding"
                )

        elif self.directives:
            raise ValueError(
                "local directives are only allowed for "
                "repair_required disposition"
            )

        # Safety boundaries. These do not decide scientific policy;
        # they prevent weaker evidence authority from silently
        # exercising stronger lifecycle power.
        if self.disposition == "keep":
            if authorities & {
                "advisory",
                "actionable",
                "terminal_candidate",
            }:
                raise ValueError(
                    "clean keep cannot contain unresolved "
                    "advisory/actionable/terminal findings"
                )

        if (
            self.disposition
            == "keep_with_warning"
        ):
            if "advisory" not in authorities:
                raise ValueError(
                    "keep_with_warning requires "
                    "at least one advisory finding"
                )

            if authorities & {
                "actionable",
                "terminal_candidate",
            }:
                raise ValueError(
                    "keep_with_warning cannot ignore "
                    "actionable/terminal findings"
                )

        if self.disposition == "reject":
            if (
                "terminal_candidate"
                not in authorities
            ):
                raise ValueError(
                    "reject requires at least one "
                    "terminal_candidate finding"
                )

        return self
