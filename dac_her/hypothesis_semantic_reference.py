from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dac_her.hypothesis_semantic_contracts import (
    HypothesisSemanticReviewDraft,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SemanticReferenceDrop(_StrictModel):
    location: str
    namespace: Literal[
        "hypothesis_ids",
        "statement_ids",
    ]
    original_ids: list[str]
    kept_ids: list[str]
    dropped_ids: list[str]
    all_supplied_references_dropped: bool = False


class SemanticReferenceAudit(_StrictModel):
    schema_version: Literal[
        "hypothesis-semantic-reference-audit-v1"
    ] = "hypothesis-semantic-reference-audit-v1"
    applied: bool = False
    drop_count: int = 0
    fatal: bool = False
    fatal_reasons: list[str] = Field(default_factory=list)
    drops: list[SemanticReferenceDrop] = Field(
        default_factory=list
    )


@dataclass(frozen=True)
class SemanticReferenceSanitizationResult:
    draft: HypothesisSemanticReviewDraft
    audit: SemanticReferenceAudit


def _filter_known(
    values: list[str],
    valid: set[str],
) -> tuple[list[str], list[str]]:
    kept = [
        value for value in values
        if value in valid
    ]
    dropped = [
        value for value in values
        if value not in valid
    ]
    return kept, dropped


class HypothesisSemanticReferenceSanitizer:
    """Exact-ID safe-drop sanitizer.

    Mixed valid/invalid lists are sanitized deterministically. No fuzzy matching
    is ever attempted. If a non-empty reference list loses every supplied ID,
    the audit is fatal and the runtime must reject the review rather than
    silently accepting an unreferenced critic judgment.
    """

    def sanitize(
        self,
        draft: HypothesisSemanticReviewDraft,
        *,
        valid_hypothesis_ids: set[str],
        valid_statement_ids: set[str],
    ) -> SemanticReferenceSanitizationResult:
        drops: list[SemanticReferenceDrop] = []
        fatal_reasons: list[str] = []
        dimensions = []

        for index, row in enumerate(draft.dimensions):
            h_kept, h_dropped = _filter_known(
                row.hypothesis_ids,
                valid_hypothesis_ids,
            )
            s_kept, s_dropped = _filter_known(
                row.statement_ids,
                valid_statement_ids,
            )

            if h_dropped:
                all_dropped = bool(
                    row.hypothesis_ids
                    and not h_kept
                )
                location = (
                    f"dimensions[{index}].hypothesis_ids"
                )
                drops.append(
                    SemanticReferenceDrop(
                        location=location,
                        namespace="hypothesis_ids",
                        original_ids=list(
                            row.hypothesis_ids
                        ),
                        kept_ids=h_kept,
                        dropped_ids=h_dropped,
                        all_supplied_references_dropped=(
                            all_dropped
                        ),
                    )
                )
                if all_dropped:
                    fatal_reasons.append(
                        f"{location}: every supplied "
                        "hypothesis reference was unknown"
                    )

            if s_dropped:
                all_dropped = bool(
                    row.statement_ids
                    and not s_kept
                )
                location = (
                    f"dimensions[{index}].statement_ids"
                )
                drops.append(
                    SemanticReferenceDrop(
                        location=location,
                        namespace="statement_ids",
                        original_ids=list(
                            row.statement_ids
                        ),
                        kept_ids=s_kept,
                        dropped_ids=s_dropped,
                        all_supplied_references_dropped=(
                            all_dropped
                        ),
                    )
                )
                if all_dropped:
                    fatal_reasons.append(
                        f"{location}: every supplied "
                        "statement reference was unknown"
                    )

            dimensions.append(
                row.model_copy(
                    update={
                        "hypothesis_ids": h_kept,
                        "statement_ids": s_kept,
                    }
                )
            )

        sanitized = draft.model_copy(
            update={"dimensions": dimensions}
        )
        audit = SemanticReferenceAudit(
            applied=bool(drops),
            drop_count=sum(
                len(row.dropped_ids)
                for row in drops
            ),
            fatal=bool(fatal_reasons),
            fatal_reasons=fatal_reasons,
            drops=drops,
        )
        return SemanticReferenceSanitizationResult(
            draft=sanitized,
            audit=audit,
        )
