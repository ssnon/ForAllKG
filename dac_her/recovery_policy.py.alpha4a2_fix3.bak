from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dac_her.validation_issues import IssueCode, ValidationReport


class RecoveryAction(str, Enum):
    ACCEPT = "accept"
    NORMALIZE = "normalize"
    SEMANTIC_PATCH = "semantic_patch"
    RECHUNK = "rechunk"
    MICRO_REEXTRACT = "micro_reextract"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    issue_codes: tuple[str, ...]

LOSSLESS_CODES = {
    IssueCode.DUPLICATE_NODE_ID,
    IssueCode.INVALID_MEASURED_FOR_COUNT,
    IssueCode.MEASURED_FOR_TARGET_MISMATCH,
    IssueCode.MISSING_MEASUREMENT_GROUP_EDGE,
    IssueCode.UNEXPECTED_MEASUREMENT_GROUP_EDGE,
    IssueCode.SINGLETON_MEASUREMENT_GROUP,
    IssueCode.DUPLICATE_MEASUREMENT_GROUP_MEMBER,
}


def decide_recovery(
    *,
    report: ValidationReport,
    normalization_attempted: bool,
    patch_attempts: int,
    micro_reextract_attempts: int,
    post_micro_patch_attempts: int,
    split_depth: int,
    source_tokens: int,
    max_patch_attempts: int,
    max_micro_reextract_attempts: int,
    max_post_micro_patch_attempts: int,
    micro_reextract_max_source_tokens: int,
    max_split_depth: int,
    min_rechunk_source_tokens: int,
    isolated_rechunk_threshold: int,
    issue_family_rechunk_threshold: int,
    undefined_endpoint_rechunk_threshold: int,
) -> RecoveryDecision:
    codes = tuple(sorted(item.value for item in report.codes()))
    if report.valid:
        return RecoveryDecision(RecoveryAction.ACCEPT, "No validation errors.", codes)

    if not normalization_attempted and report.codes() & LOSSLESS_CODES:
        return RecoveryDecision(
            RecoveryAction.NORMALIZE,
            "At least one issue can be normalized without scientific inference.",
            codes,
        )

    cannot_rechunk_safely = (
        split_depth >= max_split_depth
        or source_tokens <= min_rechunk_source_tokens
    )

    if cannot_rechunk_safely:
        # ---------------------------------------------------------
        # Phase 1: constrained semantic patch before micro-reextract
        # ---------------------------------------------------------
        if (
            micro_reextract_attempts == 0
            and patch_attempts < max_patch_attempts
        ):
            return RecoveryDecision(
                RecoveryAction.SEMANTIC_PATCH,
                (
                    "The chunk cannot be safely split; "
                    "attempt a constrained pre-micro patch."
                ),
                codes,
            )

        # ---------------------------------------------------------
        # Phase 2: one-shot micro re-extraction
        # ---------------------------------------------------------
        if (
            source_tokens
            <= micro_reextract_max_source_tokens
            and micro_reextract_attempts
            < max_micro_reextract_attempts
        ):
            return RecoveryDecision(
                RecoveryAction.MICRO_REEXTRACT,
                (
                    "Pre-micro semantic patch attempts were exhausted "
                    "for a small unsplittable source leaf."
                ),
                codes,
            )

        # ---------------------------------------------------------
        # Phase 3: patch small residual errors left by micro-reextract
        # ---------------------------------------------------------
        if (
            micro_reextract_attempts > 0
            and post_micro_patch_attempts
            < max_post_micro_patch_attempts
        ):
            return RecoveryDecision(
                RecoveryAction.SEMANTIC_PATCH,
                (
                    "Micro-reextract completed but left a small "
                    "strict-validation residual; attempt one "
                    "post-micro constrained patch."
                ),
                codes,
            )

        # ---------------------------------------------------------
        # Terminal state
        # ---------------------------------------------------------
        return RecoveryDecision(
            RecoveryAction.QUARANTINE,
            (
                "Pre-micro patch, micro-reextract, and "
                "post-micro patch budgets were exhausted "
                "and the chunk cannot be split safely."
            ),
            codes,
        )
    
    isolated_count = report.count(IssueCode.ISOLATED_NODE)
    undefined_count = (
        report.count(IssueCode.UNDEFINED_EDGE_SOURCE)
        + report.count(IssueCode.UNDEFINED_EDGE_TARGET)
    )

    if isolated_count >= isolated_rechunk_threshold:
        return RecoveryDecision(
            RecoveryAction.RECHUNK,
            f"Graph assembly failure: {isolated_count} isolated nodes.",
            codes,
        )

    if report.family_count() >= issue_family_rechunk_threshold:
        return RecoveryDecision(
            RecoveryAction.RECHUNK,
            f"Graph contains {report.family_count()} independent issue families.",
            codes,
        )

    if undefined_count >= undefined_endpoint_rechunk_threshold:
        return RecoveryDecision(
            RecoveryAction.RECHUNK,
            f"Graph contains {undefined_count} undefined edge endpoints.",
            codes,
        )

    if patch_attempts < max_patch_attempts:
        return RecoveryDecision(
            RecoveryAction.SEMANTIC_PATCH,
            "The remaining issue set is small enough for a constrained patch.",
            codes,
        )

    return RecoveryDecision(
        RecoveryAction.RECHUNK,
        "Constrained patch attempts were exhausted.",
        codes,
    )
