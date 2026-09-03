from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pipeline_core.discovery.novelty_residue import (
    NoveltyResidueClaim,
    assess_residual_specification,
)


ClosureTargetSlot = Literal[
    "BASE_RELATION",
    "DISTINGUISHING_FACTOR_EFFECT",
    "BRIDGE_RELATION",
    "FULL_RELATION",
]

ClosureTargetBasis = Literal[
    "RELATION_NUCLEUS",
    "IDENTITY_PLUS_RELATION_CONTEXT",
    "EXTRACTIVE_REQUIRED_BRIDGE",
    "FULL_RESIDUAL_CLAIM",
]


@dataclass(frozen=True)
class ClosureRetrievalTarget:
    slot: ClosureTargetSlot
    source_claim_id: str
    target_basis: ClosureTargetBasis
    search_terms: tuple[str, ...]
    search_query: str
    source_text: str
    identity_anchor_terms: tuple[str, ...] = ()
    inference_provenance: dict[str, object] | None = None
    evidence_status: Literal["UNASSESSED"] = "UNASSESSED"


@dataclass(frozen=True)
class ClosureRetrievalPlan:
    hypothesis_id: str
    claim_id: str
    claim_text: str
    targets: tuple[ClosureRetrievalTarget, ...]
    policy_version: Literal[
        "n9-closure-retrieval-policy-v3"
    ] = "n9-closure-retrieval-policy-v3"


def _normalize_text(
    value: str,
    *,
    limit: int = 600,
) -> str:
    text = str(value or "")
    text = re.sub(
        r"[‐-‒–—−-]+",
        " ",
        text,
    )
    text = re.sub(
        r"[^\w\s+*/().,]",
        " ",
        text,
        flags=re.UNICODE,
    )
    return " ".join(
        text.split()
    )[:limit].strip()


def _unique_terms(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    rows: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned = _normalize_text(
            value,
            limit=140,
        )

        key = cleaned.lower()

        if cleaned and key not in seen:
            seen.add(key)
            rows.append(cleaned)

    return tuple(rows)


def _query_from_terms(
    terms: tuple[str, ...],
) -> str:
    return _normalize_text(
        " ".join(terms),
        limit=600,
    )


_MODERATOR_RELATION_ONLY_TERMS = frozenset(
    {
        "conditional dependence",
        "conditional association",
        "conditional relationship",
        "moderation",
        "moderator interaction",
        "interaction",
        "conditioned relationship",
        "conditioned association",
    }
)


def _remove_moderator_structure_terms(
    nucleus: tuple[str, ...],
) -> tuple[str, ...]:
    """Remove terms that encode only the higher-order moderator form.

    BASE_RELATION must represent the unconditioned lower-order
    relation. This function performs exact normalized filtering only;
    it does not add synonyms or infer scientific relations.
    """

    output: list[str] = []

    for term in nucleus:
        normalized = _normalize_text(
            term,
            limit=140,
        ).casefold()

        if (
            normalized
            in _MODERATOR_RELATION_ONLY_TERMS
        ):
            continue

        output.append(term)

    return tuple(output)


def _remove_exact_identity_terms(
    *,
    nucleus: tuple[str, ...],
    identity: tuple[str, ...],
) -> tuple[str, ...]:
    """Remove exact normalized moderator identity from base terms.

    This is intentionally narrower than semantic or substring
    matching. It prevents an explicitly duplicated moderator identity
    from contaminating the BASE_RELATION target without inventing or
    reinterpreting scientific content.
    """

    identity_keys = {
        _normalize_text(
            term,
            limit=140,
        ).casefold()
        for term in identity
        if _normalize_text(
            term,
            limit=140,
        )
    }

    return tuple(
        term
        for term in nucleus
        if (
            _normalize_text(
                term,
                limit=140,
            ).casefold()
            not in identity_keys
        )
    )


def _source_token_keys(
    text: str,
) -> set[str]:
    """Return normalized source vocabulary without semantic expansion."""

    normalized = _normalize_text(
        text,
        limit=8000,
    )

    return {
        token.casefold()
        for token in re.findall(
            r"\w+",
            normalized,
            flags=re.UNICODE,
        )
        if token
    }


def _source_supported_retrieval_terms(
    *,
    source_text: str,
    candidates: tuple[str, ...],
) -> tuple[str, ...]:
    """Keep only structured terms whose tokens occur in source text.

    Terms are retrieval intent only. This function does not assert
    that separated source tokens form an established scientific
    relation, and it performs no synonym or semantic expansion.
    """

    source_keys = _source_token_keys(
        source_text
    )

    output: list[str] = []

    for term in _unique_terms(candidates):
        term_tokens = {
            token.casefold()
            for token in re.findall(
                r"\w+",
                term,
                flags=re.UNICODE,
            )
            if token
        }

        if (
            term_tokens
            and term_tokens <= source_keys
        ):
            output.append(term)

    return tuple(output)


def build_closure_retrieval_plan(
    claim: NoveltyResidueClaim,
) -> ClosureRetrievalPlan:
    """Build four provenance-preserving closure retrieval targets.

    Planner output is search intent only, never scientific evidence.

    BASE:
        relation_nucleus_terms only.

    FACTOR:
        prior_art_identity_terms plus relation context.
        This remains a retrieval target and does not assert a new
        factor-effect proposition.

    BRIDGE:
        only the already provenance-checked required_bridge.

    FULL:
        the original residual claim.
    """

    specification = assess_residual_specification(
        claim
    )

    if (
        specification.status
        != "READY_FOR_CLOSURE"
    ):
        raise ValueError(
            "Closure retrieval planning requires a "
            "READY_FOR_CLOSURE atomic residue."
        )

    identity = _unique_terms(
        claim.prior_art_identity_terms
    )

    nucleus = _unique_terms(
        claim.relation_nucleus_terms
    )

    if not identity:
        raise ValueError(
            "Closure retrieval planning requires "
            "prior_art_identity_terms."
        )

    if not nucleus:
        raise ValueError(
            "Closure retrieval planning requires "
            "relation_nucleus_terms."
        )

    base_nucleus = nucleus

    # For a moderator interaction, prior_art_identity_terms carries
    # the moderator/factor while relation_nucleus_terms carries the
    # underlying relation. LLM decomposition may occasionally repeat
    # the exact moderator identity in both fields. Do not allow that
    # duplication to collapse BASE_RELATION and
    # DISTINGUISHING_FACTOR_EFFECT into the same retrieval target.
    #
    # Only exact normalized equality is removed. No substring,
    # synonym, embedding, or scientific-semantic inference is used.
    if claim.claim_kind == "moderator_interaction":
        base_nucleus = _remove_exact_identity_terms(
            nucleus=nucleus,
            identity=identity,
        )

        base_nucleus = (
            _remove_moderator_structure_terms(
                base_nucleus
            )
        )

        if not base_nucleus:
            raise ValueError(
                "Moderator identity normalization leaves "
                "an empty base relation nucleus."
            )

    base_query = _query_from_terms(
        base_nucleus
    )

    factor_terms = _unique_terms(
        (
            *identity,
            *base_nucleus,
        )
    )

    factor_query = _query_from_terms(
        factor_terms
    )

    bridge_query = _normalize_text(
        claim.required_bridge
    )

    full_query = _normalize_text(
        claim.claim_text
    )

    structured_retrieval_terms = _unique_terms(
        (
            *identity,
            *base_nucleus,
            *claim.distinguishing_terms,
        )
    )

    bridge_retrieval_terms = (
        _source_supported_retrieval_terms(
            source_text=claim.required_bridge,
            candidates=structured_retrieval_terms,
        )
    )

    full_retrieval_terms = (
        _source_supported_retrieval_terms(
            source_text=claim.claim_text,
            candidates=structured_retrieval_terms,
        )
    )

    if not all(
        (
            base_query,
            factor_query,
            bridge_query,
            full_query,
        )
    ):
        raise ValueError(
            "Closure retrieval planning produced "
            "an empty search target."
        )

    targets = (
        ClosureRetrievalTarget(
            slot="BASE_RELATION",
            source_claim_id=claim.claim_id,
            inference_provenance=(
                claim.inference_provenance
            ),
            target_basis="RELATION_NUCLEUS",
            search_terms=base_nucleus,
            search_query=base_query,
            identity_anchor_terms=(),
            source_text=" | ".join(base_nucleus),
        ),
        ClosureRetrievalTarget(
            slot="DISTINGUISHING_FACTOR_EFFECT",
            source_claim_id=claim.claim_id,
            inference_provenance=(
                claim.inference_provenance
            ),
            target_basis=(
                "IDENTITY_PLUS_RELATION_CONTEXT"
            ),
            search_terms=factor_terms,
            search_query=factor_query,
            identity_anchor_terms=identity,
            source_text=(
                "identity="
                + " | ".join(identity)
                + "; relation_context="
                + " | ".join(base_nucleus)
            ),
        ),
        ClosureRetrievalTarget(
            slot="BRIDGE_RELATION",
            source_claim_id=claim.claim_id,
            inference_provenance=(
                claim.inference_provenance
            ),
            target_basis=(
                "EXTRACTIVE_REQUIRED_BRIDGE"
            ),
            search_terms=bridge_retrieval_terms,
            search_query=bridge_query,
            identity_anchor_terms=identity,
            source_text=claim.required_bridge,
        ),
        ClosureRetrievalTarget(
            slot="FULL_RELATION",
            source_claim_id=claim.claim_id,
            inference_provenance=(
                claim.inference_provenance
            ),
            target_basis="FULL_RESIDUAL_CLAIM",
            search_terms=full_retrieval_terms,
            search_query=full_query,
            identity_anchor_terms=identity,
            source_text=claim.claim_text,
        ),
    )

    return ClosureRetrievalPlan(
        hypothesis_id=claim.hypothesis_id,
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        targets=targets,
    )
