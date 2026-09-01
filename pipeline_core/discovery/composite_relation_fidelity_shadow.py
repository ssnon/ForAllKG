from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


_COMPOSITE_SOURCE_MODE = (
    "task_conditioned_composite_bridge_projection"
)

_COMPOSITE_RELATION = (
    "MAY_RELATE_TO_VIA_COMPOSED_CANDIDATE_BRIDGE"
)


_GENERIC_TOKENS = {
    "the",
    "and",
    "with",
    "within",
    "from",
    "into",
    "through",
    "between",
    "across",
    "that",
    "this",
    "local",
    "effective",
    "structural",
    "structure",
    "relation",
    "relationship",
    "behavior",
    "measured",
    "interpreted",
}


_COMPARATOR_PATTERNS = (
    r"\bsimilar\b",
    r"\bcomparable\b",
    r"\bsame\b",
    r"\bretained\b",
    r"\bremain(?:s|ed)?\b",
    r"\bheld (?:fixed|constant)\b",
    r"\botherwise comparable\b",
)


_MEDIATOR_CONTRAST_PATTERNS = (
    r"\bdifferent\b",
    r"\bdiffer(?:s|ed|ing)?\b",
    r"\bvar(?:y|ies|ied|ying)\b",
    r"\bacross\b",
    r"\bwhen\b",
)


_OUTCOME_CONTRAST_PATTERNS = (
    r"\bdistinct\b",
    r"\bdifferent\b",
    r"\bdiffer(?:s|ed|ing)?\b",
    r"\bnon[- ]?equivalent\b",
    r"\bnot equivalent\b",
    r"\bchange\b",
    r"\bvar(?:y|ies|ied|ying)\b",
    r"\bdepends?\b",
    r"\bnot invariant\b",
    r"\brather than (?:being )?invariant\b",
)


_SHARED_MEDIATOR_RE = re.compile(
    r"\[SHARED MEDIATOR:\s*([^\]]+)\]",
    flags=re.I,
)


@dataclass(frozen=True)
class CompositeObservationStructuralDiagnostic:
    observation_index: int

    source_matches: tuple[str, ...]
    mediator_matches: tuple[str, ...]
    outcome_matches: tuple[str, ...]

    matched_source_state: bool
    mediator_contrast: bool
    outcome_contrast: bool

    complete_conditional_consequence: bool


@dataclass(frozen=True)
class CompositeRelationFidelityShadowReview:
    axis_id: str
    hypothesis_id: str

    applicable: bool

    status: Literal[
        "pass",
        "fail",
        "not_applicable",
    ]

    shared_mediator: str | None

    source_tokens: tuple[str, ...]
    mediator_tokens: tuple[str, ...]
    outcome_tokens: tuple[str, ...]

    observations: tuple[
        CompositeObservationStructuralDiagnostic,
        ...
    ]

    reason_codes: tuple[str, ...]


def _normalize(
    text: object,
) -> str:
    value = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    ).lower()

    value = re.sub(
        r"[^a-z0-9α-ω+/-]+",
        " ",
        value,
    )

    return " ".join(
        value.split()
    )


def _tokens(
    text: object,
) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z0-9α-ω]+",
            _normalize(text),
        )
        if (
            len(token) >= 3
            and
            token not in _GENERIC_TOKENS
        )
    }


def _role_matches(
    role_tokens: set[str],
    text_tokens: set[str],
) -> tuple[str, ...]:
    """
    Conservative lexical role matching.

    Exact tokens are preferred. A narrow trailing-s equivalence
    covers ordinary plural realization (shape/shapes, size/sizes)
    without introducing semantic inference.
    """

    matches: set[str] = set()

    for role in role_tokens:
        for observed in text_tokens:

            if observed == role:
                matches.add(
                    role
                )
                break

            if (
                role != "sers"
                and
                (
                    observed.rstrip("s")
                    == role.rstrip("s")
                )
            ):
                matches.add(
                    role
                )
                break

    return tuple(
        sorted(matches)
    )


def _has_any(
    text: str,
    patterns: tuple[str, ...],
) -> bool:
    return any(
        re.search(
            pattern,
            text,
            flags=re.I,
        )
        is not None
        for pattern in patterns
    )


def _has_role_local_signal(
    text: str,
    role_matches: tuple[str, ...],
    patterns: tuple[str, ...],
    *,
    max_gap_tokens: int = 2,
) -> bool:
    """
    Require the linguistic signal to occur locally around the
    scientific role it is supposed to qualify.

    This prevents a contrast attached to one role from being
    borrowed by another role merely because both occur somewhere
    in the same observable.
    """

    if not role_matches:
        return False

    value = _normalize(text)

    token_gap = (
        rf"(?:\s+[a-z0-9α-ω+/-]+)"
        rf"{{0,{max_gap_tokens}}}\s+"
    )

    for role in role_matches:
        role_re = (
            rf"\b{re.escape(role)}\b"
        )

        for marker in patterns:

            marker_before_role = (
                rf"(?:{marker})"
                rf"{token_gap}"
                rf"{role_re}"
            )

            role_before_marker = (
                rf"{role_re}"
                rf"{token_gap}"
                rf"(?:{marker})"
            )

            if (
                re.search(
                    marker_before_role,
                    value,
                    flags=re.I,
                )
                is not None
                or
                re.search(
                    role_before_marker,
                    value,
                    flags=re.I,
                )
                is not None
            ):
                return True

    return False


class CompositeRelationFidelityShadowCritic:
    """
    Shadow-only structural diagnostic for composed discovery axes.

    This review does NOT assess:
      - scientific truth,
      - external novelty,
      - semantic distinctiveness,
      - mechanism switching,
      - counterfactual novelty.

    It asks one narrower provenance/fidelity question:

    Does at least one predicted observable preserve the composed
    relation as a complete conditional consequence?

        comparable source state
        + variation in shared mediator
        + contrasting requested outcome

    The review is intentionally not wired into production selection.
    """

    def review(
        self,
        axis,
        card,
    ) -> CompositeRelationFidelityShadowReview:

        axis_id = str(
            getattr(
                axis,
                "axis_id",
                "",
            )
        )

        hypothesis_id = str(
            getattr(
                card,
                "hypothesis_id",
                "",
            )
        )

        source_mode = str(
            getattr(
                axis,
                "source_mode",
                "",
            )
        )

        proposed_relation = str(
            getattr(
                axis,
                "proposed_relation",
                "",
            )
        )

        if (
            source_mode
            != _COMPOSITE_SOURCE_MODE
            or
            proposed_relation
            != _COMPOSITE_RELATION
        ):
            return (
                CompositeRelationFidelityShadowReview(
                    axis_id=axis_id,
                    hypothesis_id=hypothesis_id,
                    applicable=False,
                    status="not_applicable",
                    shared_mediator=None,
                    source_tokens=(),
                    mediator_tokens=(),
                    outcome_tokens=(),
                    observations=(),
                    reason_codes=(
                        "not_composite_task_bridge_axis",
                    ),
                )
            )

        rendered_path = str(
            getattr(
                axis,
                "rendered_path",
                "",
            )
        )

        mediator_match = (
            _SHARED_MEDIATOR_RE.search(
                rendered_path
            )
        )

        if mediator_match is None:
            return (
                CompositeRelationFidelityShadowReview(
                    axis_id=axis_id,
                    hypothesis_id=hypothesis_id,
                    applicable=True,
                    status="fail",
                    shared_mediator=None,
                    source_tokens=tuple(
                        sorted(
                            _tokens(
                                getattr(
                                    axis,
                                    "proposed_subject",
                                    "",
                                )
                            )
                        )
                    ),
                    mediator_tokens=(),
                    outcome_tokens=tuple(
                        sorted(
                            _tokens(
                                getattr(
                                    axis,
                                    "proposed_object",
                                    "",
                                )
                            )
                        )
                    ),
                    observations=(),
                    reason_codes=(
                        "shared_mediator_not_recoverable",
                    ),
                )
            )

        shared_mediator = (
            mediator_match
            .group(1)
            .strip()
        )

        source_tokens = _tokens(
            getattr(
                axis,
                "proposed_subject",
                "",
            )
        )

        mediator_tokens = _tokens(
            shared_mediator
        )

        outcome_tokens = _tokens(
            getattr(
                axis,
                "proposed_object",
                "",
            )
        )

        diagnostics = []

        for index, observation in enumerate(
            getattr(
                card,
                "predicted_observations",
                (),
            ),
            start=1,
        ):
            observable = str(
                getattr(
                    observation,
                    "observable",
                    "",
                )
            )

            observable_tokens = _tokens(
                observable
            )

            source_matches = (
                _role_matches(
                    source_tokens,
                    observable_tokens,
                )
            )

            mediator_matches = (
                _role_matches(
                    mediator_tokens,
                    observable_tokens,
                )
            )

            outcome_matches = (
                _role_matches(
                    outcome_tokens,
                    observable_tokens,
                )
            )

            matched_source_state = (
                _has_role_local_signal(
                    observable,
                    source_matches,
                    _COMPARATOR_PATTERNS,
                )
            )

            mediator_contrast = (
                _has_role_local_signal(
                    observable,
                    mediator_matches,
                    _MEDIATOR_CONTRAST_PATTERNS,
                )
            )

            outcome_contrast = (
                _has_role_local_signal(
                    observable,
                    outcome_matches,
                    _OUTCOME_CONTRAST_PATTERNS,
                )
            )

            complete = (
                matched_source_state
                and
                mediator_contrast
                and
                outcome_contrast
            )

            diagnostics.append(
                CompositeObservationStructuralDiagnostic(
                    observation_index=index,
                    source_matches=
                        source_matches,
                    mediator_matches=
                        mediator_matches,
                    outcome_matches=
                        outcome_matches,
                    matched_source_state=
                        matched_source_state,
                    mediator_contrast=
                        mediator_contrast,
                    outcome_contrast=
                        outcome_contrast,
                    complete_conditional_consequence=
                        complete,
                )
            )

        passed = any(
            row.complete_conditional_consequence
            for row in diagnostics
        )

        reasons = []

        if not diagnostics:
            reasons.append(
                "no_predicted_observations"
            )

        if not passed:
            reasons.append(
                "no_complete_conditional_consequence"
            )

        return (
            CompositeRelationFidelityShadowReview(
                axis_id=axis_id,
                hypothesis_id=hypothesis_id,
                applicable=True,
                status=(
                    "pass"
                    if passed
                    else "fail"
                ),
                shared_mediator=
                    shared_mediator,
                source_tokens=tuple(
                    sorted(
                        source_tokens
                    )
                ),
                mediator_tokens=tuple(
                    sorted(
                        mediator_tokens
                    )
                ),
                outcome_tokens=tuple(
                    sorted(
                        outcome_tokens
                    )
                ),
                observations=tuple(
                    diagnostics
                ),
                reason_codes=tuple(
                    sorted(
                        set(
                            reasons
                        )
                    )
                ),
            )
        )
