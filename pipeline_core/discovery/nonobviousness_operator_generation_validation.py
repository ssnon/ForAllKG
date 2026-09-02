from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from pydantic import (
    BaseModel,
    ConfigDict,
)

from pipeline_core.discovery.nonobviousness_mechanism_semantics_contracts import (
    MechanismSearchOperator,
)
from pipeline_core.discovery.nonobviousness_operator_generation_contracts import (
    N11OperatorGenerationDraft,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class N11OperatorGenerationIssue(
    StrictModel
):
    code: str
    location: str
    message: str


class N11OperatorGenerationValidation(
    StrictModel
):
    passes: bool
    issues: list[
        N11OperatorGenerationIssue
    ]


@dataclass(frozen=True)
class N11OperatorGenerationAuthority:
    requested_operator: MechanismSearchOperator

    eligible_operators: tuple[
        MechanismSearchOperator,
        ...,
    ]

    allowed_baseline_statement_ids: tuple[
        str,
        ...,
    ]

    allowed_supplemental_node_ids: tuple[
        str,
        ...,
    ]

    allowed_gap_statement_ids: tuple[
        str,
        ...,
    ]

    allowed_shared_component_ids: tuple[
        str,
        ...,
    ]

    allowed_supplemental_only_component_ids: tuple[
        str,
        ...,
    ]


_NOVELTY_PATTERNS = (
    re.compile(
        r"\bnovel\b",
        re.I,
    ),
    re.compile(
        r"\bunprecedented\b",
        re.I,
    ),
    re.compile(
        r"\bfirst\s+(?:report|demonstration|observation)\b",
        re.I,
    ),
)


_FORBIDDEN_STRONG_OPERATOR_PATTERNS = (
    (
        "UNAUTHORIZED_COMPETITION_CLAIM",
        re.compile(
            r"\b(?:compete|competition|outcompete)\w*\b",
            re.I,
        ),
    ),
    (
        "UNAUTHORIZED_SWITCH_CLAIM",
        re.compile(
            r"\b(?:mechanism\s+switch|switches?\s+(?:between|from|to))\b",
            re.I,
        ),
    ),
    (
        "UNAUTHORIZED_THRESHOLD_CLAIM",
        re.compile(
            r"\bthreshold\b",
            re.I,
        ),
    ),
    (
        "UNAUTHORIZED_REVERSAL_CLAIM",
        re.compile(
            r"\b(?:reversal|reverses?|sign\s+change)\b",
            re.I,
        ),
    ),
    (
        "UNAUTHORIZED_NONMONOTONIC_CLAIM",
        re.compile(
            r"\bnon[\s-]?monotonic\b",
            re.I,
        ),
    ),
)


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?\d+(?:\.\d+)?"
    r"(?![A-Za-z0-9_])"
)



_NEGATED_OPERATOR_CONTEXT_PATTERNS = (
    # "rather than as evidence of pathway competition"
    re.compile(
        r"\brather\s+than\b.{0,120}$",
        re.I,
    ),

    # "does not imply/establish a mechanism switch"
    re.compile(
        (
            r"\b(?:does|do|did|is|are|was|were|can|could|"
            r"would|should|may|might)\s+not\b.{0,120}$"
        ),
        re.I,
    ),

    # "not evidence of competition"
    # "without evidence for a switch"
    re.compile(
        (
            r"\b(?:not|without)\s+(?:as\s+)?"
            r"(?:evidence|proof|support|a\s+claim)\b"
            r".{0,120}$"
        ),
        re.I,
    ),

    # "without implying competition"
    re.compile(
        (
            r"\bwithout\s+"
            r"(?:implying|assuming|claiming|establishing|requiring)"
            r"\b.{0,120}$"
        ),
        re.I,
    ),

    # "no evidence for mechanism switching"
    re.compile(
        (
            r"\bno\s+(?:evidence|support|basis)\s+"
            r"(?:of|for)\b.{0,120}$"
        ),
        re.I,
    ),
)


def _operator_mention_is_explicitly_negated(
    text: str,
    match_start: int,
) -> bool:
    """Return True only for an explicit local negation/boundary mention.

    Strong-operator vocabulary remains prohibited when asserted
    affirmatively. This exception only prevents phrases such as
    "rather than evidence of pathway competition" from being
    misclassified as positive competition claims.
    """

    prefix = str(text)[
        max(
            0,
            int(match_start) - 180,
        ):
        int(match_start)
    ]

    # Restrict negation scope to the current local clause.
    # A negation in an earlier sentence must not license a later
    # affirmative strong-operator claim.
    local_clause = re.split(
        r"[\n.;!?]",
        prefix,
    )[-1]

    return any(
        pattern.search(
            local_clause
        )
        is not None
        for pattern
        in _NEGATED_OPERATOR_CONTEXT_PATTERNS
    )


def _norm(
    text: str,
) -> str:
    value = unicodedata.normalize(
        "NFKC",
        str(text),
    ).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _observable_matches(
    first: str,
    second: str,
) -> bool:
    a = _norm(first)
    b = _norm(second)

    if not a or not b:
        return False

    return (
        a == b
        or (
            len(a) >= 5
            and a in b
        )
        or (
            len(b) >= 5
            and b in a
        )
    )


def _candidate_text(
    candidate,
) -> str:
    parts = [
        candidate.title,
        candidate.hypothesis_statement,
        candidate.relative_contribution_claim,
        candidate.inferential_bridge,
        *candidate.assumptions,
    ]

    for row in (
        candidate.predicted_observations
    ):
        parts.extend(
            [
                row.observable,
                row.rationale,
            ]
        )

    for row in (
        candidate.falsification_criteria
    ):
        # Structural reference IDs are provenance/linkage metadata,
        # not generated scientific prose. Excluding prediction_local_id
        # prevents digits inside IDs such as "prediction:1" from being
        # misclassified as unsupported numeric scientific predictions.
        parts.append(
            row.falsifying_outcome
        )

    return "\n".join(
        str(x)
        for x in parts
    )


def _unknown(
    actual: Iterable[str],
    allowed: Iterable[str],
) -> list[str]:
    return sorted(
        set(
            map(
                str,
                actual,
            )
        )
        - set(
            map(
                str,
                allowed,
            )
        )
    )


class N11OperatorGenerationValidator:
    def validate(
        self,
        *,
        authority: N11OperatorGenerationAuthority,
        draft: N11OperatorGenerationDraft,
    ) -> N11OperatorGenerationValidation:
        issues: list[
            N11OperatorGenerationIssue
        ] = []

        def error(
            code: str,
            location: str,
            message: str,
        ) -> None:
            issues.append(
                N11OperatorGenerationIssue(
                    code=code,
                    location=location,
                    message=message,
                )
            )

        # Abstention is always permitted.
        if draft.candidate is None:
            return (
                N11OperatorGenerationValidation(
                    passes=True,
                    issues=[],
                )
            )

        candidate = draft.candidate

        # --------------------------------------------------------
        # Operator authority
        # --------------------------------------------------------

        if (
            authority.requested_operator
            not in authority.eligible_operators
        ):
            error(
                "REQUESTED_OPERATOR_NOT_AUTHORIZED",
                "authority.requested_operator",
                authority.requested_operator,
            )

        if (
            candidate.operator
            != authority.requested_operator
        ):
            error(
                "OPERATOR_MISMATCH",
                "candidate.operator",
                (
                    f"requested={authority.requested_operator}, "
                    f"actual={candidate.operator}"
                ),
            )

        # --------------------------------------------------------
        # Provenance / ID authority
        # --------------------------------------------------------

        id_surfaces = (
            (
                "UNKNOWN_BASELINE_PREMISE",
                (
                    "candidate."
                    "baseline_premise_statement_ids"
                ),
                candidate.baseline_premise_statement_ids,
                authority.allowed_baseline_statement_ids,
            ),
            (
                "UNKNOWN_SUPPLEMENTAL_MECHANISM_NODE",
                (
                    "candidate."
                    "supplemental_mechanism_node_ids"
                ),
                candidate.supplemental_mechanism_node_ids,
                authority.allowed_supplemental_node_ids,
            ),
            (
                "UNKNOWN_GAP_STATEMENT",
                "candidate.gap_statement_ids",
                candidate.gap_statement_ids,
                authority.allowed_gap_statement_ids,
            ),
            (
                "UNKNOWN_SHARED_COMPONENT",
                "candidate.shared_component_ids",
                candidate.shared_component_ids,
                authority.allowed_shared_component_ids,
            ),
            (
                "UNKNOWN_SUPPLEMENTAL_ONLY_COMPONENT",
                (
                    "candidate."
                    "supplemental_only_component_ids"
                ),
                candidate.supplemental_only_component_ids,
                (
                    authority
                    .allowed_supplemental_only_component_ids
                ),
            ),
        )

        for (
            code,
            location,
            actual,
            allowed,
        ) in id_surfaces:
            unknown = _unknown(
                actual,
                allowed,
            )

            if unknown:
                error(
                    code,
                    location,
                    str(unknown),
                )

        # --------------------------------------------------------
        # Operator-specific semantic contract
        # --------------------------------------------------------

        if (
            authority.requested_operator
            == "RELATIVE_CONTRIBUTION_SHIFT"
        ):
            normalized_claim = _norm(
                candidate
                .relative_contribution_claim
            )

            has_relative = (
                "relative"
                in normalized_claim
            )

            has_contribution_concept = any(
                token in normalized_claim
                for token in (
                    "contribution",
                    "balance",
                    "fraction",
                    "weight",
                )
            )

            if not (
                has_relative
                and has_contribution_concept
            ):
                error(
                    (
                        "RELATIVE_CONTRIBUTION_"
                        "SEMANTICS_MISSING"
                    ),
                    (
                        "candidate."
                        "relative_contribution_claim"
                    ),
                    (
                        "RELATIVE_CONTRIBUTION_SHIFT "
                        "requires an explicit relative "
                        "contribution/balance claim"
                    ),
                )

        # --------------------------------------------------------
        # Generated-text safety
        # --------------------------------------------------------

        generated_text = _candidate_text(
            candidate
        )

        for pattern in _NOVELTY_PATTERNS:
            if pattern.search(
                generated_text
            ):
                error(
                    "EXTERNAL_NOVELTY_CLAIM",
                    "candidate",
                    (
                        "external novelty is not "
                        "established during N11 generation"
                    ),
                )
                break

        for (
            code,
            pattern,
        ) in (
            _FORBIDDEN_STRONG_OPERATOR_PATTERNS
        ):
            for match in pattern.finditer(
                generated_text
            ):
                if (
                    _operator_mention_is_explicitly_negated(
                        generated_text,
                        match.start(),
                    )
                ):
                    continue

                error(
                    code,
                    "candidate",
                    (
                        "generated candidate introduces "
                        "an unauthorized stronger operator"
                    ),
                )

                # One issue per unauthorized operator class is enough.
                break

        # C1 deliberately bans generated numeric values entirely.
        if _NUMBER_RE.search(
            generated_text
        ):
            error(
                "NUMERIC_PREDICTION_NOT_ALLOWED_IN_C1",
                "candidate",
                (
                    "C1 operator generation must "
                    "remain qualitative"
                ),
            )

        # --------------------------------------------------------
        # Prediction / falsification consistency
        # --------------------------------------------------------

        predictions = {
            row.local_id: row
            for row in (
                candidate.predicted_observations
            )
        }

        discriminating = predictions[
            candidate
            .discriminating_observation_local_id
        ]

        if not str(
            discriminating.observable
        ).strip():
            error(
                "EMPTY_DISCRIMINATING_OBSERVABLE",
                (
                    "candidate."
                    "discriminating_observation_local_id"
                ),
                (
                    candidate
                    .discriminating_observation_local_id
                ),
            )

        # Falsifier-to-prediction linkage is structural rather than
        # lexical. N11OperatorCandidateDraft already guarantees that
        # every prediction_local_id resolves to an exact prediction.

        return (
            N11OperatorGenerationValidation(
                passes=not issues,
                issues=issues,
            )
        )
