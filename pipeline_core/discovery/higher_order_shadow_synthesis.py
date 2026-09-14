from __future__ import annotations

import hashlib
import re
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.higher_order_synthesis_context import (
    HigherOrderSynthesisContext,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HigherOrderShadowHypothesisDraft(StrictModel):
    """
    One shadow-only higher-order hypothesis realization.

    The draft deliberately carries no positive novelty status and no directional
    interaction claim. Any interaction among the known components remains a
    hypothesis requiring downstream prior-art and N10 review.
    """

    schema_version: str = "higher-order-shadow-hypothesis-draft-v1"

    decision: Literal["propose", "abstain"]

    hypothesis_statement: str | None = None
    mechanistic_bridge: str | None = None
    predicted_observation: str | None = None
    falsification_condition: str | None = None

    assumptions: list[str] = Field(default_factory=list)

    expected_direction: Literal["unspecified"] = "unspecified"
    evidence_status: Literal["hypothesis_only"] = "hypothesis_only"
    novelty_status: Literal["not_assessed"] = "not_assessed"

    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_decision_shape(
        self,
    ) -> "HigherOrderShadowHypothesisDraft":
        proposal_fields = (
            self.hypothesis_statement,
            self.mechanistic_bridge,
            self.predicted_observation,
            self.falsification_condition,
        )

        if self.decision == "propose":
            if not all(
                isinstance(value, str) and value.strip()
                for value in proposal_fields
            ):
                raise ValueError(
                    "proposed higher-order shadow hypothesis requires "
                    "statement, bridge, prediction, and falsification condition"
                )
            if self.abstention_reason is not None:
                raise ValueError(
                    "proposed higher-order shadow hypothesis cannot carry "
                    "abstention_reason"
                )
        else:
            if any(
                value is not None
                for value in proposal_fields
            ):
                raise ValueError(
                    "abstained higher-order shadow hypothesis cannot carry "
                    "proposal fields"
                )
            if not (
                isinstance(self.abstention_reason, str)
                and self.abstention_reason.strip()
            ):
                raise ValueError(
                    "abstained higher-order shadow hypothesis requires "
                    "abstention_reason"
                )

        return self


class HigherOrderShadowAuthorityAudit(StrictModel):
    schema_version: str = "higher-order-shadow-authority-audit-v1"

    novelty_language_hits: list[str] = Field(default_factory=list)
    evidence_upgrade_hits: list[str] = Field(default_factory=list)
    directional_interaction_hits: list[str] = Field(default_factory=list)
    mediator_identity_language_hits: list[str] = Field(default_factory=list)

    fixed_hypothesis_only_status: bool
    fixed_mediator_identity_not_authorized: bool
    fixed_novelty_not_assessed_status: bool
    fixed_unspecified_direction: bool

    authority_safe: bool


_NOVELTY_PATTERNS = (
    re.compile(r"\bnovel\b", flags=re.IGNORECASE),
    re.compile(r"\bunprecedented\b", flags=re.IGNORECASE),
    re.compile(r"\bpreviously unknown\b", flags=re.IGNORECASE),
    re.compile(r"\bnewly discovered\b", flags=re.IGNORECASE),
    re.compile(
        r"\bfirst(?:-|\s)+(?:ever|report|reported|demonstration|demonstrated)\b",
        flags=re.IGNORECASE,
    ),
)

_EVIDENCE_UPGRADE_PATTERNS = (
    re.compile(r"\bdemonstrates?\b", flags=re.IGNORECASE),
    re.compile(r"\bproves?\b", flags=re.IGNORECASE),
    re.compile(r"\bconfirms?\b", flags=re.IGNORECASE),
    re.compile(r"\bestablish(?:es|ed)?\b", flags=re.IGNORECASE),
    re.compile(r"\bshows that\b", flags=re.IGNORECASE),
    re.compile(r"\bis known to\b", flags=re.IGNORECASE),
)

# These terms are treated as unsafe for the higher-order interaction itself.
# Nouns such as "enhancement" remain allowed because they occur in the recorded
# SERS mediator vocabulary; the audit targets directional verbs/adjectives.
_DIRECTION_PATTERNS = (
    re.compile(r"\bincrease(?:s|d)?\b", flags=re.IGNORECASE),
    re.compile(r"\bdecrease(?:s|d)?\b", flags=re.IGNORECASE),
    re.compile(
        r"\bhigher\b(?![-\s]+order\b)",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\blower\b", flags=re.IGNORECASE),
    re.compile(r"\bstronger\b", flags=re.IGNORECASE),
    re.compile(r"\bweaker\b", flags=re.IGNORECASE),
    re.compile(r"\benhance(?:s|d)?\b", flags=re.IGNORECASE),
    re.compile(r"\bsuppress(?:es|ed)?\b", flags=re.IGNORECASE),
)

# The current higher-order contract proves compatibility among mediator-like
# relation arguments, not scientific identity and not an explicit M1->M2 edge.
# These phrases are therefore fail-closed until a separate mediator-equivalence
# authority is introduced.
_MEDIATOR_IDENTITY_PATTERNS = (
    re.compile(
        r"\bshared\b[^.\n]{0,100}\bmediator\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bsame\b[^.\n]{0,100}\bmediator\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bcommon mediator\b", flags=re.IGNORECASE),
    re.compile(r"\bmediator path\b", flags=re.IGNORECASE),
    re.compile(r"\blinked expressions\b", flags=re.IGNORECASE),
    re.compile(
        r"\bconnects?\b[^.\n]{0,180}\bthrough\b",
        flags=re.IGNORECASE,
    ),
)


def _proposal_text(
    draft: HigherOrderShadowHypothesisDraft,
) -> str:
    return "\n".join(
        value
        for value in (
            draft.hypothesis_statement,
            draft.mechanistic_bridge,
            draft.predicted_observation,
            draft.falsification_condition,
        )
        if value
    )


_NEGATION_SCOPE_RE = re.compile(
    r"(?:^|\b)(?:"
    r"no|not|without|never|neither|nor|cannot|can't|"
    r"do\s+not|does\s+not|did\s+not|"
    r"is\s+not|are\s+not|was\s+not|were\s+not"
    r")(?:\b|$)",
    flags=re.IGNORECASE,
)

_ADVERSATIVE_SCOPE_RE = re.compile(
    r"\b(?:but|however|yet|although|though|instead)\b",
    flags=re.IGNORECASE,
)


def _is_explicitly_negated_match(
    text: str,
    match: re.Match[str],
) -> bool:
    """Return True only for a locally explicit negation of the hit.

    Authority audits should not fail on guard language such as
    "no mediator path is assumed" or "without establishing ...".
    The scope is restricted to the current sentence/clause and reset after
    an adversative so that "not X, but Y establishes ..." still flags Y.
    """

    start = max(
        text.rfind(".", 0, match.start()),
        text.rfind("\n", 0, match.start()),
        text.rfind(";", 0, match.start()),
    ) + 1
    prefix = text[start:match.start()]

    adversatives = list(
        _ADVERSATIVE_SCOPE_RE.finditer(prefix)
    )
    if adversatives:
        prefix = prefix[adversatives[-1].end():]

    # "not only" is not a negation of the following assertion.
    normalized = re.sub(
        r"\bnot\s+only\b",
        "",
        prefix,
        flags=re.IGNORECASE,
    )

    if _NEGATION_SCOPE_RE.search(normalized):
        return True

    # Contrastive exclusion such as
    # "treated as separate expressions rather than as ... a mediator path"
    # negates the complement after "rather than". Keep the scope local:
    # a comma terminates the exclusion so that
    # "Rather than X, the topology establishes a mediator path"
    # remains an authority violation.
    contrastive = list(
        re.finditer(
            r"\brather\s+than(?:\s+as)?\b",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    if contrastive:
        suffix = normalized[contrastive[-1].end():]
        if "," not in suffix:
            return True

    return False


def _pattern_hits(
    text: str,
    patterns: Sequence[re.Pattern[str]],
) -> list[str]:
    hits = []
    seen = set()

    for pattern in patterns:
        for match in pattern.finditer(text):
            if _is_explicitly_negated_match(
                text,
                match,
            ):
                continue

            token = match.group(0)
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            hits.append(token)

    return hits


def audit_higher_order_shadow_hypothesis(
    draft: HigherOrderShadowHypothesisDraft,
) -> HigherOrderShadowAuthorityAudit:
    text = (
        _proposal_text(draft)
        if draft.decision == "propose"
        else ""
    )

    novelty_hits = _pattern_hits(
        text,
        _NOVELTY_PATTERNS,
    )
    evidence_hits = _pattern_hits(
        text,
        _EVIDENCE_UPGRADE_PATTERNS,
    )
    direction_hits = _pattern_hits(
        text,
        _DIRECTION_PATTERNS,
    )
    mediator_identity_hits = _pattern_hits(
        text,
        _MEDIATOR_IDENTITY_PATTERNS,
    )

    fixed_hypothesis = (
        draft.evidence_status == "hypothesis_only"
    )
    fixed_novelty = (
        draft.novelty_status == "not_assessed"
    )
    fixed_direction = (
        draft.expected_direction == "unspecified"
    )
    fixed_mediator_identity = True

    authority_safe = (
        not novelty_hits
        and not evidence_hits
        and not direction_hits
        and not mediator_identity_hits
        and fixed_hypothesis
        and fixed_novelty
        and fixed_direction
    )

    return HigherOrderShadowAuthorityAudit(
        novelty_language_hits=novelty_hits,
        evidence_upgrade_hits=evidence_hits,
        directional_interaction_hits=direction_hits,
        mediator_identity_language_hits=mediator_identity_hits,
        fixed_hypothesis_only_status=fixed_hypothesis,
        fixed_mediator_identity_not_authorized=fixed_mediator_identity,
        fixed_novelty_not_assessed_status=fixed_novelty,
        fixed_unspecified_direction=fixed_direction,
        authority_safe=authority_safe,
    )


def select_shadow_synthesis_contexts(
    *,
    contexts: Sequence[HigherOrderSynthesisContext],
    max_contexts: int,
) -> tuple[HigherOrderSynthesisContext, ...]:
    """
    Deterministic smoke-sample selection with no scientific ranking.

    At most one context is retained per modifier component. Among duplicate
    backbone realizations for one modifier, the lexicographically smallest
    context id is retained. Unique modifiers are then ordered by a stable hash
    of modifier_component_id. This is a coverage sample, not a quality score.
    """

    if max_contexts < 1:
        raise ValueError(
            "max_contexts must be at least 1"
        )

    by_modifier: dict[
        str,
        HigherOrderSynthesisContext,
    ] = {}

    for context in contexts:
        modifier_id = (
            context.lineage.modifier_component_id
        )
        previous = by_modifier.get(modifier_id)

        if (
            previous is None
            or context.context_id < previous.context_id
        ):
            by_modifier[modifier_id] = context

    ranked = sorted(
        by_modifier.values(),
        key=lambda context: (
            hashlib.sha256(
                context.lineage.modifier_component_id.encode(
                    "utf-8"
                )
            ).hexdigest(),
            context.context_id,
        ),
    )

    return tuple(
        ranked[:max_contexts]
    )
