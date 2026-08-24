from __future__ import annotations

import hashlib
import json

from pipeline_core.discovery.hypothesis_action_contracts import (
    G1ActionDecision,
    G1ActionDirective,
    G1FindingRef,
    G1FindingScope,
)


_POLICY_VERSION = (
    "sers-g1-combined-action-policy-v1"
)


class SERSCombinedActionPolicyError(
    ValueError
):
    pass


def _canonical_json(
    value: object,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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


def _require_target(
    finding: G1FindingRef,
    *,
    portfolio_id: str,
    hypothesis_id: str,
) -> None:
    if (
        finding.target_portfolio_id
        != portfolio_id
    ):
        raise SERSCombinedActionPolicyError(
            "finding target portfolio mismatch"
        )

    if (
        finding.target_hypothesis_id
        != hypothesis_id
    ):
        raise SERSCombinedActionPolicyError(
            "finding target hypothesis mismatch"
        )


class SERSCombinedLifecycleActionPolicy:
    """Combine normalized G1 authority lanes.

    v1 lanes:
      * S1 context action decision
      * calibrated external novelty findings
      * final semantic-critic findings

    Mutation semantics:
      * context actionable findings preserve the already-adjudicated
        local REFRAME directives from the S1 context policy;
      * external-novelty actionable findings fold into exactly one
        hypothesis-level DOWNGRADE directive;
      * semantic FAIL/actionable findings fold into one
        hypothesis-level REFRAME directive;
      * advisory findings remain warning pressure only;
      * terminal_candidate is deliberately not auto-adjudicated in
        combined-policy v1. A later explicit terminal policy must own
        rejection semantics.

    This class remains declarative. It never mutates the hypothesis.
    """

    policy_version = (
        _POLICY_VERSION
    )

    def decide(
        self,
        *,
        target_portfolio_id: str,
        target_hypothesis_id: str,
        context_decision:
            G1ActionDecision,
        external_findings:
            list[G1FindingRef],
        semantic_findings:
            list[G1FindingRef],
    ) -> G1ActionDecision:

        # ----------------------------------------------------------
        # Context lane must already be an S1-context adjudication for
        # exactly this target.
        # ----------------------------------------------------------

        if (
            context_decision.target_portfolio_id
            != target_portfolio_id
        ):
            raise SERSCombinedActionPolicyError(
                "context decision target portfolio mismatch"
            )

        if (
            context_decision.target_hypothesis_id
            != target_hypothesis_id
        ):
            raise SERSCombinedActionPolicyError(
                "context decision target hypothesis mismatch"
            )

        if context_decision.mutation_applied:
            raise SERSCombinedActionPolicyError(
                "combined policy accepts declarative "
                "context decisions only"
            )

        for finding in (
            context_decision.findings
        ):
            if (
                finding.source_kind
                != "context_review"
            ):
                raise SERSCombinedActionPolicyError(
                    "context decision contains "
                    "non-context finding"
                )

            _require_target(
                finding,
                portfolio_id=
                    target_portfolio_id,
                hypothesis_id=
                    target_hypothesis_id,
            )

        for finding in external_findings:
            if (
                finding.source_kind
                != "external_novelty"
            ):
                raise SERSCombinedActionPolicyError(
                    "external lane contains "
                    "non-external-novelty finding"
                )

            _require_target(
                finding,
                portfolio_id=
                    target_portfolio_id,
                hypothesis_id=
                    target_hypothesis_id,
            )

        for finding in semantic_findings:
            if (
                finding.source_kind
                != "semantic_review"
            ):
                raise SERSCombinedActionPolicyError(
                    "semantic lane contains "
                    "non-semantic-review finding"
                )

            _require_target(
                finding,
                portfolio_id=
                    target_portfolio_id,
                hypothesis_id=
                    target_hypothesis_id,
            )


        all_findings = sorted(
            [
                *context_decision.findings,
                *external_findings,
                *semantic_findings,
            ],
            key=lambda row:
                row.finding_ref_id,
        )

        finding_ids = [
            row.finding_ref_id
            for row in all_findings
        ]

        if (
            len(finding_ids)
            != len(set(finding_ids))
        ):
            raise SERSCombinedActionPolicyError(
                "duplicate finding_ref_id across "
                "combined G1 authority lanes"
            )


        terminal = [
            row
            for row in all_findings
            if row.authority
            == "terminal_candidate"
        ]

        if terminal:
            raise SERSCombinedActionPolicyError(
                "terminal_candidate requires explicit "
                "G1 terminal adjudication policy; "
                "combined-policy v1 does not auto-reject"
            )


        external_actionable = [
            row
            for row in external_findings
            if row.authority
            == "actionable"
        ]

        semantic_actionable = [
            row
            for row in semantic_findings
            if row.authority
            == "actionable"
        ]

        all_actionable = [
            row
            for row in all_findings
            if row.authority
            == "actionable"
        ]

        all_advisory = [
            row
            for row in all_findings
            if row.authority
            == "advisory"
        ]


        # ----------------------------------------------------------
        # Preserve already-adjudicated context directives exactly.
        # ----------------------------------------------------------

        directives = list(
            context_decision.directives
        )

        context_directive_ids = {
            row.directive_id
            for row in directives
        }

        if (
            len(context_directive_ids)
            != len(directives)
        ):
            raise SERSCombinedActionPolicyError(
                "duplicate context directive_id"
            )


        # ----------------------------------------------------------
        # External novelty actionable findings represent novelty
        # positioning pressure, not a scientific-content rewrite.
        #
        # Fold ALL such findings into one hypothesis-level downgrade
        # directive. Claim-level prior-art findings remain provenance
        # supporting that one lifecycle downgrade.
        # ----------------------------------------------------------

        if external_actionable:
            external_refs = sorted(
                row.finding_ref_id
                for row
                in external_actionable
            )

            directive_id = _stable_id(
                "g1_directive",
                {
                    "policy":
                        self.policy_version,

                    "lane":
                        "external_novelty",

                    "target_portfolio_id":
                        target_portfolio_id,

                    "target_hypothesis_id":
                        target_hypothesis_id,

                    "action":
                        "downgrade",

                    "finding_ref_ids":
                        external_refs,
                },
            )

            directives.append(
                G1ActionDirective(
                    directive_id=
                        directive_id,

                    action=
                        "downgrade",

                    target_scope=
                        G1FindingScope(
                            kind=
                                "hypothesis",

                            hypothesis_ids=[
                                target_hypothesis_id
                            ],
                        ),

                    finding_ref_ids=
                        external_refs,

                    rationale=(
                        "Calibrated external prior-art "
                        "evidence requires a bounded "
                        "downgrade of novelty positioning. "
                        "This does not assert scientific "
                        "falsity and does not rewrite the "
                        "scientific claim by itself."
                    ),
                )
            )


        # ----------------------------------------------------------
        # A semantic FAIL is actionable but deliberately nonterminal.
        # Since semantic findings are hypothesis-level diagnostics,
        # fold all semantic actionable findings into one bounded
        # hypothesis-level REFRAME directive.
        #
        # Historical H1/H2 contain no semantic actionable findings.
        # ----------------------------------------------------------

        if semantic_actionable:
            semantic_refs = sorted(
                row.finding_ref_id
                for row
                in semantic_actionable
            )

            directive_id = _stable_id(
                "g1_directive",
                {
                    "policy":
                        self.policy_version,

                    "lane":
                        "semantic_review",

                    "target_portfolio_id":
                        target_portfolio_id,

                    "target_hypothesis_id":
                        target_hypothesis_id,

                    "action":
                        "reframe",

                    "finding_ref_ids":
                        semantic_refs,
                },
            )

            directives.append(
                G1ActionDirective(
                    directive_id=
                        directive_id,

                    action=
                        "reframe",

                    target_scope=
                        G1FindingScope(
                            kind=
                                "hypothesis",

                            hypothesis_ids=[
                                target_hypothesis_id
                            ],
                        ),

                    finding_ref_ids=
                        semantic_refs,

                    rationale=(
                        "Final semantic critic identified "
                        "an actionable semantic failure; "
                        "bounded hypothesis-level reframing "
                        "is required before clean acceptance."
                    ),
                )
            )


        directive_ids = [
            row.directive_id
            for row in directives
        ]

        if (
            len(directive_ids)
            != len(set(directive_ids))
        ):
            raise SERSCombinedActionPolicyError(
                "duplicate directive_id in "
                "combined G1 decision"
            )


        # ----------------------------------------------------------
        # Combined disposition.
        # ----------------------------------------------------------

        reason_codes: list[str] = []


        if all_actionable:
            disposition = (
                "repair_required"
            )

            if (
                context_decision.disposition
                == "repair_required"
            ):
                reason_codes.append(
                    "G1_COMBINED_CONTEXT_REPAIR_REQUIRED"
                )

            if external_actionable:
                reason_codes.append(
                    "G1_COMBINED_EXTERNAL_NOVELTY_DOWNGRADE"
                )

            if semantic_actionable:
                reason_codes.append(
                    "G1_COMBINED_SEMANTIC_REPAIR_REQUIRED"
                )

            if all_advisory:
                reason_codes.append(
                    "G1_COMBINED_ADVISORY_FINDINGS_PRESENT"
                )

            interpretation = (
                "At least one normalized authority lane "
                "requires bounded lifecycle repair. "
                "Context defects remain local REFRAME "
                "actions, external prior-art pressure is "
                "represented as novelty DOWNGRADE, and "
                "semantic warnings remain advisory. "
                "No terminal rejection authority is "
                "exercised by this policy version."
            )

        elif all_advisory:
            disposition = (
                "keep_with_warning"
            )

            reason_codes.append(
                "G1_COMBINED_ADVISORY_ONLY"
            )

            interpretation = (
                "No actionable or terminal finding is "
                "present, but advisory evidence remains. "
                "The hypothesis is retained with warnings."
            )

        else:
            disposition = "keep"

            reason_codes.append(
                "G1_COMBINED_CLEAR"
            )

            interpretation = (
                "All normalized findings are "
                "informational; no lifecycle action "
                "is required."
            )


        decision_id = _stable_id(
            "g1_action_decision",
            {
                "policy":
                    self.policy_version,

                "target_portfolio_id":
                    target_portfolio_id,

                "target_hypothesis_id":
                    target_hypothesis_id,

                "finding_ref_ids":
                    finding_ids,

                "directive_ids":
                    sorted(
                        row.directive_id
                        for row
                        in directives
                    ),

                "disposition":
                    disposition,

                "reason_codes":
                    sorted(
                        set(reason_codes)
                    ),
            },
        )


        return G1ActionDecision(
            decision_id=
                decision_id,

            target_portfolio_id=
                target_portfolio_id,

            target_hypothesis_id=
                target_hypothesis_id,

            findings=
                all_findings,

            directives=
                sorted(
                    directives,
                    key=lambda row:
                        row.directive_id,
                ),

            disposition=
                disposition,

            reason_codes=
                sorted(
                    set(reason_codes)
                ),

            interpretation=
                interpretation,

            mutation_applied=
                False,
        )
