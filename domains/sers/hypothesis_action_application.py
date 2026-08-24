from __future__ import annotations

import hashlib
import json

from domains.sers.hypothesis_context_contracts import (
    expected_hypothesis_context_assertions,
)
from pipeline_core.discovery.hypothesis_action_application_contracts import (
    G1ApplicationAssertionSource,
    G1ApplicationPlan,
    G1NoveltyDispositionConstraint,
    G1ScientificRepairConstraint,
)
from pipeline_core.discovery.hypothesis_action_contracts import (
    G1ActionDecision,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisCard,
)


_POLICY_VERSION = (
    "sers-g1-application-plan-v1"
)


class SERSG1ApplicationPlanError(
    ValueError
):
    pass


def _canonical_json(
    value: object,
) -> str:
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


def _sha256_json(
    value: object,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            value
        ).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    payload: object,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            payload
        ).encode("utf-8")
    ).hexdigest()[:20]

    return f"{prefix}:{digest}"


class SERSG1ApplicationPlanBuilder:
    """Compile a declarative G1 decision into a bounded repair plan.

    This builder never generates scientific wording and never mutates
    a HypothesisCard.

    Scientific REFRAME directives become constraints for the existing
    HypothesisDraftBackend.repair mutation seam.

    DOWNGRADE remains lifecycle/novelty metadata in the application
    artifact and does not rewrite scientific text.

    Current G1 application v1 deliberately refuses REMOVE_ASSERTION.
    Removal semantics require an explicit patch/application contract.
    """

    policy_version = _POLICY_VERSION

    def build(
        self,
        *,
        source_card: HypothesisCard,
        decision: G1ActionDecision,
    ) -> G1ApplicationPlan:

        if (
            decision.target_hypothesis_id
            != source_card.hypothesis_id
        ):
            raise SERSG1ApplicationPlanError(
                "decision/card hypothesis mismatch"
            )

        if decision.mutation_applied:
            raise SERSG1ApplicationPlanError(
                "application planning requires "
                "an unapplied declarative decision"
            )

        if (
            decision.disposition
            != "repair_required"
        ):
            raise SERSG1ApplicationPlanError(
                "application-plan v1 accepts "
                "repair_required decisions only"
            )

        catalog = {
            row["assertion_id"]:
                row
            for row in (
                expected_hypothesis_context_assertions(
                    source_card
                )
            )
        }

        finding_by_id = {
            row.finding_ref_id:
                row
            for row in decision.findings
        }

        actionable_ids = {
            row.finding_ref_id
            for row in decision.findings
            if row.authority == "actionable"
        }

        scientific = []
        novelty = []

        covered_actionable = set()
        targeted_assertion_ids = set()

        for directive in sorted(
            decision.directives,
            key=lambda row:
                row.directive_id,
        ):
            unknown_refs = (
                set(
                    directive.finding_ref_ids
                )
                - set(finding_by_id)
            )

            if unknown_refs:
                raise SERSG1ApplicationPlanError(
                    "directive references unknown "
                    "finding IDs: "
                    + repr(
                        sorted(
                            unknown_refs
                        )
                    )
                )

            non_actionable = [
                ref_id
                for ref_id
                in directive.finding_ref_ids
                if (
                    finding_by_id[
                        ref_id
                    ].authority
                    != "actionable"
                )
            ]

            if non_actionable:
                raise SERSG1ApplicationPlanError(
                    "application directive contains "
                    "non-actionable finding refs: "
                    + repr(
                        sorted(
                            non_actionable
                        )
                    )
                )

            covered_actionable.update(
                directive.finding_ref_ids
            )

            if directive.action == "reframe":
                scope = directive.target_scope

                if scope.kind not in {
                    "central",
                    "bridge",
                    "prediction",
                    "assumption",
                }:
                    raise SERSG1ApplicationPlanError(
                        "REFRAME requires "
                        "assertion-level scope"
                    )

                if not scope.assertion_ids:
                    raise SERSG1ApplicationPlanError(
                        "REFRAME has no assertion IDs"
                    )

                sources = []

                for assertion_id in (
                    scope.assertion_ids
                ):
                    if (
                        assertion_id
                        in targeted_assertion_ids
                    ):
                        raise SERSG1ApplicationPlanError(
                            "multiple directives target "
                            "the same source assertion: "
                            + assertion_id
                        )

                    row = catalog.get(
                        assertion_id
                    )

                    if row is None:
                        raise SERSG1ApplicationPlanError(
                            "REFRAME assertion is absent "
                            "from source card: "
                            + assertion_id
                        )

                    if (
                        row["assertion_kind"]
                        != scope.kind
                    ):
                        raise SERSG1ApplicationPlanError(
                            "REFRAME assertion kind "
                            "does not match scope"
                        )

                    targeted_assertion_ids.add(
                        assertion_id
                    )

                    text = row[
                        "assertion_text"
                    ]

                    sources.append(
                        G1ApplicationAssertionSource(
                            assertion_id=
                                assertion_id,

                            assertion_kind=
                                scope.kind,

                            assertion_text=
                                text,

                            assertion_text_sha256=
                                hashlib.sha256(
                                    text.encode(
                                        "utf-8"
                                    )
                                ).hexdigest(),
                        )
                    )

                scientific.append(
                    G1ScientificRepairConstraint(
                        directive_id=
                            directive.directive_id,

                        source_scope=
                            scope,

                        source_assertions=
                            sources,

                        finding_ref_ids=
                            sorted(
                                directive.finding_ref_ids
                            ),

                        rationale=
                            directive.rationale,
                    )
                )

            elif directive.action == "downgrade":
                scope = directive.target_scope

                if (
                    scope.kind
                    != "hypothesis"
                    or scope.assertion_ids
                ):
                    raise SERSG1ApplicationPlanError(
                        "DOWNGRADE must be "
                        "hypothesis-level and "
                        "must not rewrite assertions"
                    )

                novelty.append(
                    G1NoveltyDispositionConstraint(
                        directive_id=
                            directive.directive_id,

                        finding_ref_ids=
                            sorted(
                                directive.finding_ref_ids
                            ),

                        rationale=
                            directive.rationale,
                    )
                )

            elif (
                directive.action
                == "remove_assertion"
            ):
                raise SERSG1ApplicationPlanError(
                    "REMOVE_ASSERTION application "
                    "is not implemented in v1"
                )

            else:
                raise SERSG1ApplicationPlanError(
                    "unsupported G1 application "
                    "action: "
                    + directive.action
                )

        if (
            covered_actionable
            != actionable_ids
        ):
            missing = sorted(
                actionable_ids
                - covered_actionable
            )

            extra = sorted(
                covered_actionable
                - actionable_ids
            )

            raise SERSG1ApplicationPlanError(
                "application directives do not "
                "exactly cover actionable findings; "
                f"missing={missing}, extra={extra}"
            )

        source_card_sha = (
            _sha256_json(
                source_card
            )
        )

        plan_id = _stable_id(
            "g1_application_plan",
            {
                "policy":
                    self.policy_version,

                "source_portfolio_id":
                    decision
                    .target_portfolio_id,

                "source_hypothesis_id":
                    source_card
                    .hypothesis_id,

                "source_decision_id":
                    decision
                    .decision_id,

                "source_card_sha256":
                    source_card_sha,

                "scientific_constraints": [
                    row.model_dump(
                        mode="json"
                    )
                    for row in scientific
                ],

                "novelty_constraints": [
                    row.model_dump(
                        mode="json"
                    )
                    for row in novelty
                ],
            },
        )

        return G1ApplicationPlan(
            plan_id=
                plan_id,

            source_portfolio_id=
                decision
                .target_portfolio_id,

            source_hypothesis_id=
                source_card
                .hypothesis_id,

            source_decision_id=
                decision
                .decision_id,

            source_card_sha256=
                source_card_sha,

            scientific_repair_constraints=
                scientific,

            novelty_disposition_constraints=
                novelty,

            source_generation_mutated=
                False,
        )
