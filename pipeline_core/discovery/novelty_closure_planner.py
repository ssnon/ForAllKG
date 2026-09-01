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
    evidence_status: Literal["UNASSESSED"] = "UNASSESSED"


@dataclass(frozen=True)
class ClosureRetrievalPlan:
    hypothesis_id: str
    claim_id: str
    claim_text: str
    targets: tuple[ClosureRetrievalTarget, ...]
    policy_version: Literal[
        "n9-closure-retrieval-policy-v1"
    ] = "n9-closure-retrieval-policy-v1"


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

    base_query = _query_from_terms(
        nucleus
    )

    factor_terms = _unique_terms(
        (
            *identity,
            *nucleus,
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
            target_basis="RELATION_NUCLEUS",
            search_terms=nucleus,
            search_query=base_query,
            identity_anchor_terms=(),
            source_text=" | ".join(nucleus),
        ),
        ClosureRetrievalTarget(
            slot="DISTINGUISHING_FACTOR_EFFECT",
            source_claim_id=claim.claim_id,
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
                + " | ".join(nucleus)
            ),
        ),
        ClosureRetrievalTarget(
            slot="BRIDGE_RELATION",
            source_claim_id=claim.claim_id,
            target_basis=(
                "EXTRACTIVE_REQUIRED_BRIDGE"
            ),
            search_terms=(),
            search_query=bridge_query,
            identity_anchor_terms=identity,
            source_text=claim.required_bridge,
        ),
        ClosureRetrievalTarget(
            slot="FULL_RELATION",
            source_claim_id=claim.claim_id,
            target_basis="FULL_RESIDUAL_CLAIM",
            search_terms=(),
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
