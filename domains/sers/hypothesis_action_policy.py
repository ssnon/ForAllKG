from __future__ import annotations

from collections import defaultdict
import hashlib
import json

from pipeline_core.discovery.hypothesis_action_contracts import (
    G1ActionDecision,
    G1ActionDirective,
    G1FindingRef,
)


_POLICY_VERSION = (
    "sers-g1-context-action-policy-v1"
)


_INFORMATIONAL = frozenset({
    "match",
    "intentional_variation",
    "compatible_extension",
})

_ADVISORY = frozenset({
    "unknown",
})

_ACTIONABLE = frozenset({
    "role_mismatch",
    "context_conflation",
    "conflict",
})


class SERSContextActionPolicyError(
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


def _expected_authority(
    status: str,
) -> str:
    if status in _INFORMATIONAL:
        return "informational"

    if status in _ADVISORY:
        return "advisory"

    if status in _ACTIONABLE:
        return "actionable"

    raise SERSContextActionPolicyError(
        "unsupported S1 context status: "
        f"{status!r}"
    )


class SERSContextLifecycleActionPolicy:
    """Deterministic S1-context lifecycle subpolicy.

    Scientific boundary:

      * informational S1 findings do not require action;
      * UNKNOWN remains advisory and cannot cause rejection;
      * ROLE_MISMATCH / CONTEXT_CONFLATION / CONFLICT require
        bounded local reframing;
      * S1 context evidence alone never rejects a hypothesis.

    This policy only emits declarative G1ActionDecision objects.
    It never mutates hypothesis content.
    """

    policy_version = (
        _POLICY_VERSION
    )

    def decide(
        self,
        *,
        target_portfolio_id: str,
        target_hypothesis_id: str,
        findings: list[
            G1FindingRef
        ],
    ) -> G1ActionDecision:

        normalized = sorted(
            findings,
            key=lambda row:
                row.finding_ref_id,
        )

        ids = [
            row.finding_ref_id
            for row in normalized
        ]

        if len(ids) != len(set(ids)):
            raise SERSContextActionPolicyError(
                "duplicate G1 finding_ref_id"
            )

        for finding in normalized:
            if (
                finding.source_kind
                != "context_review"
            ):
                raise SERSContextActionPolicyError(
                    "SERS context action policy "
                    "accepts context_review findings only"
                )

            if (
                finding.target_portfolio_id
                != target_portfolio_id
            ):
                raise SERSContextActionPolicyError(
                    "finding target portfolio mismatch"
                )

            if (
                finding.target_hypothesis_id
                != target_hypothesis_id
            ):
                raise SERSContextActionPolicyError(
                    "finding target hypothesis mismatch"
                )

            expected = (
                _expected_authority(
                    finding.source_status
                )
            )

            if finding.authority != expected:
                raise SERSContextActionPolicyError(
                    "S1 status/authority mismatch: "
                    f"status={finding.source_status!r}, "
                    f"expected={expected!r}, "
                    f"actual={finding.authority!r}"
                )

            if (
                finding.authority
                == "terminal_candidate"
            ):
                raise SERSContextActionPolicyError(
                    "S1 context finding must never "
                    "carry terminal_candidate authority"
                )

        actionable = [
            row
            for row in normalized
            if row.authority
            == "actionable"
        ]

        advisory = [
            row
            for row in normalized
            if row.authority
            == "advisory"
        ]

        informational = [
            row
            for row in normalized
            if row.authority
            == "informational"
        ]

        directives: list[
            G1ActionDirective
        ] = []

        # ----------------------------------------------------------
        # Local-action folding.
        #
        # Multiple actionable findings that target the exact same
        # final scientific scope become one REFRAME directive.
        # This prevents duplicated mutations while preserving every
        # finding_ref_id as provenance.
        # ----------------------------------------------------------

        grouped: dict[
            tuple[
                str,
                tuple[str, ...],
            ],
            list[G1FindingRef],
        ] = defaultdict(list)

        for finding in actionable:
            key = (
                finding
                .target_scope
                .kind,

                tuple(
                    finding
                    .target_scope
                    .assertion_ids
                ),
            )

            grouped[
                key
            ].append(
                finding
            )

        for (
            scope_key,
            scope_findings,
        ) in sorted(
            grouped.items(),
            key=lambda item:
                repr(item[0]),
        ):
            exemplar = (
                scope_findings[0]
            )

            finding_ref_ids = sorted(
                row.finding_ref_id
                for row
                in scope_findings
            )

            directive_id = _stable_id(
                "g1_directive",
                {
                    "policy":
                        self.policy_version,

                    "target_portfolio_id":
                        target_portfolio_id,

                    "target_hypothesis_id":
                        target_hypothesis_id,

                    "action":
                        "reframe",

                    "scope_kind":
                        scope_key[0],

                    "assertion_ids":
                        list(
                            scope_key[1]
                        ),

                    "finding_ref_ids":
                        finding_ref_ids,
                },
            )

            statuses = sorted({
                row.source_status
                for row
                in scope_findings
            })

            directives.append(
                G1ActionDirective(
                    directive_id=
                        directive_id,

                    action=
                        "reframe",

                    target_scope=
                        exemplar
                        .target_scope,

                    finding_ref_ids=
                        finding_ref_ids,

                    rationale=(
                        "S1 context compatibility "
                        "requires bounded local "
                        "reframing at this scope; "
                        "source statuses="
                        + ",".join(
                            statuses
                        )
                    ),
                )
            )

        if actionable:
            disposition = (
                "repair_required"
            )

            reason_codes = [
                "S1_CONTEXT_LOCAL_REPAIR_REQUIRED",
            ]

            reason_codes.extend(
                "S1_CONTEXT_"
                + status.upper()
                for status in sorted({
                    row.source_status
                    for row in actionable
                })
            )

            if advisory:
                reason_codes.append(
                    "S1_CONTEXT_UNKNOWN_WARNING_PRESENT"
                )

            interpretation = (
                "One or more S1 context findings "
                "require bounded local reframing. "
                "S1 context evidence alone does not "
                "authorize whole-hypothesis rejection."
            )

        elif advisory:
            disposition = (
                "keep_with_warning"
            )

            reason_codes = [
                "S1_CONTEXT_UNKNOWN_WARNING",
            ]

            interpretation = (
                "No actionable S1 context defect "
                "is present, but typed context "
                "coverage remains unresolved. "
                "UNKNOWN is retained as a warning."
            )

        else:
            disposition = "keep"

            reason_codes = [
                "S1_CONTEXT_CLEAR",
            ]

            interpretation = (
                "S1 context findings are "
                "informational only; no lifecycle "
                "repair is required."
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
                    ids,

                "directive_ids": [
                    row.directive_id
                    for row in directives
                ],

                "disposition":
                    disposition,

                "reason_codes":
                    sorted(
                        set(
                            reason_codes
                        )
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
                normalized,

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
                    set(
                        reason_codes
                    )
                ),

            interpretation=
                interpretation,

            mutation_applied=
                False,
        )
