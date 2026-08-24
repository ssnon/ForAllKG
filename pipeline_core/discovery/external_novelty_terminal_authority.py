from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pipeline_core.discovery.hypothesis_action_contracts import (
    G1FindingRef,
)
from pipeline_core.discovery.novelty_refinement_contracts import (
    NoveltyRefinementReport,
)


_POLICY_VERSION = (
    "external-novelty-terminal-authority-v1"
)

_DESTRUCTIVE_EXTERNAL = frozenset({
    "WELL_ESTABLISHED",
    "CONFLICTING_PRIOR_ART",
})


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class ExternalNoveltyTerminalAuthorityError(
    ValueError
):
    pass


TerminalResolutionKind = Literal[
    "superseded_by_final_reassessment",
]


class ExternalNoveltyTerminalResolution(
    StrictModel
):
    original_finding_ref_id: str = Field(
        min_length=1
    )

    resolved_finding_ref_id: str = Field(
        min_length=1
    )

    source_status: str = Field(
        min_length=1
    )

    original_authority: Literal[
        "terminal_candidate"
    ] = "terminal_candidate"

    resolved_authority: Literal[
        "informational"
    ] = "informational"

    resolution: TerminalResolutionKind

    refinement_report_id: str = Field(
        min_length=1
    )

    refinement_decision: Literal[
        "kept_original"
    ] = "kept_original"

    final_external_status: str = Field(
        min_length=1
    )

    reject_authorized: Literal[
        False
    ] = False


class ExternalNoveltyTerminalResolutionBundle(
    StrictModel
):
    schema_version: Literal[
        "g1-external-terminal-resolution-v1"
    ] = "g1-external-terminal-resolution-v1"

    policy_version: Literal[
        "external-novelty-terminal-authority-v1"
    ] = _POLICY_VERSION

    target_portfolio_id: str = Field(
        min_length=1
    )

    target_hypothesis_id: str = Field(
        min_length=1
    )

    refinement_report_id: str = Field(
        min_length=1
    )

    refinement_decision: Literal[
        "kept_original"
    ] = "kept_original"

    final_external_status: str = Field(
        min_length=1
    )

    original_terminal_count: int = Field(
        ge=0
    )

    resolved_terminal_count: int = Field(
        ge=0
    )

    resolutions: list[
        ExternalNoveltyTerminalResolution
    ] = Field(
        default_factory=list
    )

    findings: list[
        G1FindingRef
    ] = Field(
        default_factory=list
    )

    reject_authorized: Literal[
        False
    ] = False

    mutation_applied: Literal[
        False
    ] = False

    @model_validator(mode="after")
    def _resolution_consistency(
        self,
    ) -> "ExternalNoveltyTerminalResolutionBundle":
        if (
            self.original_terminal_count
            != self.resolved_terminal_count
        ):
            raise ValueError(
                "all terminal candidates must "
                "be explicitly resolved"
            )

        if (
            self.resolved_terminal_count
            != len(self.resolutions)
        ):
            raise ValueError(
                "resolved_terminal_count does not "
                "match resolution records"
            )

        if any(
            row.authority
            == "terminal_candidate"
            for row in self.findings
        ):
            raise ValueError(
                "resolved terminal bundle must not "
                "retain terminal_candidate authority"
            )

        finding_ids = [
            row.finding_ref_id
            for row in self.findings
        ]

        if (
            len(finding_ids)
            != len(set(finding_ids))
        ):
            raise ValueError(
                "duplicate finding_ref_id in "
                "terminal resolution bundle"
            )

        return self


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


class ExternalNoveltyTerminalAuthorityResolver:
    """Resolve stale external terminal pressure against R6 final state.

    Ownership boundary:

    * R6 owns destructive external rejection.
    * This resolver operates only on final kept-original survivors.
    * A final survivor carrying WELL_ESTABLISHED or
      CONFLICTING_PRIOR_ART is an invariant violation.
    * Pre-refinement terminal_candidate findings may be retained only
      as informational provenance when R6's final external reassessment
      is non-destructive.
    * accepted_refinement is intentionally unsupported here because
      pre-refinement external findings cannot be rebound onto changed
      scientific wording; fresh final novelty assessment is required.
    """

    policy_version = _POLICY_VERSION

    destructive_external_statuses = (
        _DESTRUCTIVE_EXTERNAL
    )

    def resolve(
        self,
        *,
        target_portfolio_id: str,
        target_hypothesis_id: str,
        findings: list[
            G1FindingRef
        ],
        refinement_report:
            NoveltyRefinementReport,
    ) -> ExternalNoveltyTerminalResolutionBundle:

        if (
            refinement_report.schema_version
            != "novelty-refinement-report-v2"
        ):
            raise ExternalNoveltyTerminalAuthorityError(
                "terminal resolution requires "
                "novelty-refinement-report-v2"
            )

        if (
            refinement_report.final_portfolio_id
            != target_portfolio_id
        ):
            raise ExternalNoveltyTerminalAuthorityError(
                "refinement final portfolio mismatch"
            )

        for finding in findings:
            if (
                finding.source_kind
                != "external_novelty"
            ):
                raise ExternalNoveltyTerminalAuthorityError(
                    "terminal resolver accepts only "
                    "external_novelty findings"
                )

            if (
                finding.target_portfolio_id
                != target_portfolio_id
            ):
                raise ExternalNoveltyTerminalAuthorityError(
                    "external finding target "
                    "portfolio mismatch"
                )

            if (
                finding.target_hypothesis_id
                != target_hypothesis_id
            ):
                raise ExternalNoveltyTerminalAuthorityError(
                    "external finding target "
                    "hypothesis mismatch"
                )

        survivors = [
            attempt
            for attempt
            in refinement_report.attempts
            if (
                attempt.final_hypothesis_id
                == target_hypothesis_id
            )
        ]

        if len(survivors) != 1:
            raise ExternalNoveltyTerminalAuthorityError(
                "target final hypothesis must map "
                "to exactly one R6 surviving attempt"
            )

        attempt = survivors[0]

        if (
            attempt.decision
            != "kept_original"
        ):
            raise ExternalNoveltyTerminalAuthorityError(
                "pre-refinement terminal resolution "
                "supports kept_original only; "
                "accepted refinement requires fresh "
                "final novelty assessment"
            )

        if (
            attempt.final_external_status
            is None
        ):
            raise ExternalNoveltyTerminalAuthorityError(
                "R6 survivor missing final external status"
            )

        if (
            attempt.final_external_status
            in self.destructive_external_statuses
        ):
            raise ExternalNoveltyTerminalAuthorityError(
                "R6 final survivor carries destructive "
                "external status; final-portfolio "
                "invariant is violated"
            )

        resolved_findings: list[
            G1FindingRef
        ] = []

        resolutions: list[
            ExternalNoveltyTerminalResolution
        ] = []

        original_terminal_count = 0

        for finding in sorted(
            findings,
            key=lambda row:
                row.finding_ref_id,
        ):
            if (
                finding.authority
                != "terminal_candidate"
            ):
                resolved_findings.append(
                    finding
                )
                continue

            original_terminal_count += 1

            resolved_id = _stable_id(
                "g1_finding_ref",
                {
                    "policy":
                        self.policy_version,

                    "original_finding_ref_id":
                        finding.finding_ref_id,

                    "refinement_report_id":
                        refinement_report.report_id,

                    "refinement_decision":
                        attempt.decision,

                    "final_external_status":
                        attempt.final_external_status,

                    "resolution":
                        "superseded_by_final_reassessment",
                },
            )

            source_attributes = dict(
                finding.source_attributes
            )

            source_attributes.update({
                "original_authority":
                    "terminal_candidate",

                "terminal_resolution":
                    "superseded_by_final_reassessment",

                "terminal_rejection_owner":
                    "novelty_refinement_runtime",

                "r6_refinement_decision":
                    attempt.decision,

                "r6_final_external_status":
                    attempt.final_external_status,

                "r6_refinement_report_id":
                    refinement_report.report_id,

                "reject_authorized":
                    "false",
            })

            lineage = list(
                finding.lineage_ref_ids
            )

            if (
                refinement_report.report_id
                not in lineage
            ):
                lineage.append(
                    refinement_report.report_id
                )

            resolved_payload = (
                finding.model_dump(
                    mode="json"
                )
            )

            resolved_payload.update({
                "finding_ref_id":
                    resolved_id,

                "source_attributes":
                    source_attributes,

                "authority":
                    "informational",

                "lineage_ref_ids":
                    lineage,

                "rationale": (
                    finding.rationale
                    + " Terminal authority from the "
                    "pre-refinement external assessment "
                    "is superseded for lifecycle purposes "
                    "by the non-destructive R6 final "
                    "external reassessment. R6 remains "
                    "the sole owner of destructive "
                    "external rejection."
                ),
            })

            resolved = (
                G1FindingRef
                .model_validate(
                    resolved_payload
                )
            )

            resolved_findings.append(
                resolved
            )

            resolutions.append(
                ExternalNoveltyTerminalResolution(
                    original_finding_ref_id=
                        finding.finding_ref_id,

                    resolved_finding_ref_id=
                        resolved.finding_ref_id,

                    source_status=
                        finding.source_status,

                    resolution=(
                        "superseded_by_final_reassessment"
                    ),

                    refinement_report_id=
                        refinement_report.report_id,

                    refinement_decision=
                        "kept_original",

                    final_external_status=
                        attempt.final_external_status,

                    reject_authorized=
                        False,
                )
            )

        return (
            ExternalNoveltyTerminalResolutionBundle(
                target_portfolio_id=
                    target_portfolio_id,

                target_hypothesis_id=
                    target_hypothesis_id,

                refinement_report_id=
                    refinement_report.report_id,

                refinement_decision=
                    "kept_original",

                final_external_status=
                    attempt.final_external_status,

                original_terminal_count=
                    original_terminal_count,

                resolved_terminal_count=
                    len(resolutions),

                resolutions=
                    resolutions,

                findings=
                    sorted(
                        resolved_findings,
                        key=lambda row:
                            row.finding_ref_id,
                    ),

                reject_authorized=
                    False,

                mutation_applied=
                    False,
            )
        )
