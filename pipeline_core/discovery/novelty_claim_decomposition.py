from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

from pipeline_core.discovery.external_novelty_contracts import (
    HypothesisNoveltyClaims,
    LiteratureQuery,
    LiteratureQueryPlan,
    NoveltyClaim,
    NoveltyClaimDecompositionDraft,
)
from pipeline_core.discovery.hypothesis_contracts import HypothesisCard, HypothesisPortfolio
from pipeline_core.discovery.novelty_structure_validation import (
    compile_claim_scientific_structure,
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _clean_query(text: str, *, limit: int = 280) -> str:
    value = str(text or "")
    value = value.replace("ΔG_H*", "hydrogen adsorption free energy")
    value = value.replace("ΔG_H", "hydrogen adsorption free energy")
    value = value.replace("ΔG", "free energy")
    value = re.sub(r"[‐‑‒–—−-]+", " ", value)
    value = re.sub(r"[^\w\s+*/().,]", " ", value, flags=re.UNICODE)
    value = " ".join(value.split())
    return value[:limit].strip()


def _clean_diagnostic_terms(
    values: list[str],
) -> list[str]:
    """Normalize structured diagnostic terms without semantic expansion."""

    rows: list[str] = []

    for value in values:
        cleaned = _clean_query(
            value,
            limit=120,
        )

        if (
            cleaned
            and cleaned.lower()
            not in {
                row.lower()
                for row in rows
            }
        ):
            rows.append(cleaned)

    return rows


def _clean_branch_specific_specification(
    text: str,
    identity_terms: list[str],
) -> str:
    """Preserve a specification only when it names this atomic branch.

    This is deliberately conservative. An umbrella hypothesis-level
    statement such as "laser excitation conditions ..." must not be
    silently instantiated as a "laser power" or "excitation wavelength"
    bridge unless that branch identity is explicitly represented.

    Empty identity terms retain the cleaned text for backward
    compatibility; claims with a usable branch identity are guarded.
    """

    cleaned = " ".join(
        str(text or "").split()
    )

    if not cleaned:
        return ""

    identities = _clean_diagnostic_terms(
        identity_terms
    )

    if not identities:
        return cleaned

    normalized_text = _clean_query(
        cleaned,
        limit=2000,
    ).lower()

    for identity in identities:
        normalized_identity = _clean_query(
            identity,
            limit=120,
        ).lower()

        if (
            normalized_identity
            and normalized_identity
            in normalized_text
        ):
            return cleaned

    return ""


def _clean_branch_specific_bridge(
    text: str,
    identity_terms: list[str],
    source_texts: list[str],
) -> str:
    """Preserve only an extractively supported branch-specific bridge.

    A bridge is stronger than a branch-specific prediction/falsifier:
    it asserts the scientific proposition connecting the atomic factor
    to the residual relation.

    Therefore it must satisfy BOTH:
      1. the atomic branch identity is explicitly named; and
      2. the proposed bridge is an extractive span of the original
         hypothesis inferential bridge or assumptions.

    Paraphrasing an umbrella bridge into a new branch-specific
    proposition is intentionally rejected.
    """

    cleaned = " ".join(
        str(text or "").split()
    )

    if not cleaned:
        return ""

    identities = _clean_diagnostic_terms(
        identity_terms
    )

    if not identities:
        return ""

    normalized_bridge = _clean_query(
        cleaned,
        limit=4000,
    ).lower()

    identity_present = False

    for identity in identities:
        normalized_identity = _clean_query(
            identity,
            limit=120,
        ).lower()

        if (
            normalized_identity
            and normalized_identity
            in normalized_bridge
        ):
            identity_present = True
            break

    if not identity_present:
        return ""

    for source in source_texts:
        normalized_source = _clean_query(
            source,
            limit=8000,
        ).lower()

        if (
            normalized_bridge
            and normalized_bridge
            in normalized_source
        ):
            return cleaned

    return ""


def _assemble_diagnostic_relation_query(
    structural_terms: list[str],
    relation_terms: list[str],
    *,
    fallback: str,
) -> tuple[str, list[str], list[str]]:
    """Build a relation-first query without deleting or inventing terms."""

    structural = _clean_diagnostic_terms(
        structural_terms
    )

    relation = _clean_diagnostic_terms(
        relation_terms
    )

    candidate = _clean_query(
        " ".join(
            [
                *structural,
                *relation,
            ]
        )
    )

    # Fail safe for incomplete structured output.
    if len(candidate.split()) < 3:
        candidate = _clean_query(
            fallback
        )

    return (
        candidate,
        structural,
        relation,
    )


class NoveltyClaimBackend(Protocol):
    def decompose(
        self,
        hypothesis: HypothesisCard,
        *,
        max_claims: int,
    ) -> NoveltyClaimDecompositionDraft: ...


class NoveltyClaimDecomposer:
    def __init__(
        self,
        backend: NoveltyClaimBackend,
        *,
        max_claims_per_hypothesis: int = 4,
        max_queries_per_claim: int = 2,
    ) -> None:
        self.backend = backend
        self.max_claims = int(max_claims_per_hypothesis)
        self.max_queries = int(max_queries_per_claim)
        if self.max_claims < 1:
            raise ValueError("max_claims_per_hypothesis must be >= 1")
        if self.max_queries < 1:
            raise ValueError("max_queries_per_claim must be >= 1")

    def decompose(self, hypothesis: HypothesisCard) -> HypothesisNoveltyClaims:
        draft = self.backend.decompose(hypothesis, max_claims=self.max_claims)
        rows: list[NoveltyClaim] = []
        for rank, row in enumerate(draft.claims[: self.max_claims], start=1):
            concepts = []
            for value in row.search_concepts:
                cleaned = _clean_query(value, limit=120)
                if cleaned and cleaned not in concepts:
                    concepts.append(cleaned)
            queries = []
            for value in row.search_queries:
                cleaned = _clean_query(value)
                if cleaned and cleaned not in queries:
                    queries.append(cleaned)
                if len(queries) >= self.max_queries:
                    break
            if not queries:
                fallback = _clean_query(row.text)
                if fallback:
                    queries.append(fallback)
            if len(queries) < self.max_queries and concepts:
                concept_query = _clean_query(" ".join(concepts))
                if concept_query and concept_query not in queries:
                    queries.append(concept_query)

            diagnostic_kind = row.diagnostic_query_kind

            diagnostic_source_query = _clean_query(
                row.diagnostic_search_query or ""
            )

            (
                diagnostic_execution_query,
                diagnostic_structural_terms,
                diagnostic_relation_terms,
            ) = _assemble_diagnostic_relation_query(
                row.diagnostic_structural_terms,
                row.diagnostic_relation_terms,
                fallback=diagnostic_source_query,
            )

            prior_art_identity_terms = (
                _clean_diagnostic_terms(
                    row.prior_art_identity_terms
                )
            )

            relation_nucleus_terms = (
                _clean_diagnostic_terms(
                    row.relation_nucleus_terms
                )
            )

            scientific_structure, structure_reason_codes = (
                compile_claim_scientific_structure(
                    row.scientific_structure,
                    identity_terms=prior_art_identity_terms,
                    source_texts=[
                        hypothesis.hypothesis_statement,
                        hypothesis.inferential_bridge,
                        *hypothesis.assumptions,
                        *[
                            item.observable
                            for item
                            in hypothesis.predicted_observations
                        ],
                        *[
                            item.rationale
                            for item
                            in hypothesis.predicted_observations
                        ],
                        *[
                            item.observable
                            for item
                            in hypothesis.falsification_criteria
                        ],
                        *[
                            item.falsifying_outcome
                            for item
                            in hypothesis.falsification_criteria
                        ],
                    ],
                )
            )

            rows.append(
                NoveltyClaim(
                    claim_id=_stable_id(
                        "external_novelty_claim",
                        hypothesis.hypothesis_id,
                        rank,
                        row.kind,
                        row.text,
                    ),
                    hypothesis_id=hypothesis.hypothesis_id,
                    claim_rank=rank,
                    kind=row.kind,
                    importance=row.importance,
                    text=row.text,
                    rationale=row.rationale,
                    search_concepts=concepts,
                    search_queries=queries[: self.max_queries],
                    distinguishing_terms=_clean_diagnostic_terms(
                        row.distinguishing_terms
                    ),
                    prior_art_identity_terms=(
                        prior_art_identity_terms
                    ),
                    relation_nucleus_terms=(
                        relation_nucleus_terms
                    ),
                    required_bridge=(
                        _clean_branch_specific_bridge(
                            row.required_bridge,
                            prior_art_identity_terms,
                            [
                                hypothesis.inferential_bridge,
                                *hypothesis.assumptions,
                            ],
                        )
                    ),
                    predicted_observation=(
                        _clean_branch_specific_specification(
                            row.predicted_observation,
                            prior_art_identity_terms,
                        )
                    ),
                    falsification_condition=(
                        _clean_branch_specific_specification(
                            row.falsification_condition,
                            prior_art_identity_terms,
                        )
                    ),
                    scientific_structure=scientific_structure,
                    diagnostic_query_kind=diagnostic_kind,
                    diagnostic_search_query=(
                        diagnostic_source_query or None
                    ),
                    diagnostic_execution_query=(
                        diagnostic_execution_query or None
                    ),
                    diagnostic_structural_terms=(
                        diagnostic_structural_terms
                    ),
                    diagnostic_relation_terms=(
                        diagnostic_relation_terms
                    ),
                    scientific_structure_reason_codes=list(
                        structure_reason_codes
                    ),
                )
            )
        return HypothesisNoveltyClaims(
            hypothesis_id=hypothesis.hypothesis_id,
            title=hypothesis.title,
            claims=rows,
            decomposition_notes=draft.decomposition_notes,
        )


class LiteratureQueryPlanner:
    def __init__(self, *, include_hypothesis_composite: bool = True) -> None:
        self.include_hypothesis_composite = bool(include_hypothesis_composite)

    def build(
        self,
        portfolio: HypothesisPortfolio,
        decompositions: list[HypothesisNoveltyClaims],
    ) -> LiteratureQueryPlan:
        by_hypothesis = {row.hypothesis_id: row for row in decompositions}
        queries: list[LiteratureQuery] = []
        seen: set[tuple[str, str | None, str]] = set()

        for hypothesis in portfolio.hypotheses:
            row = by_hypothesis.get(hypothesis.hypothesis_id)
            if row is None:
                raise ValueError(
                    f"missing novelty-claim decomposition for {hypothesis.hypothesis_id}"
                )
            for claim in row.claims:
                for index, query_text in enumerate(claim.search_queries):
                    cleaned = _clean_query(query_text)
                    if not cleaned:
                        continue

                    if index == 0:
                        kind = "claim_primary"
                    else:
                        kind = "claim_variant"
                    key = (hypothesis.hypothesis_id, claim.claim_id, cleaned.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    queries.append(
                        LiteratureQuery(
                            query_id=_stable_id(
                                "literature_query",
                                hypothesis.hypothesis_id,
                                claim.claim_id,
                                kind,
                                cleaned,
                            ),
                            hypothesis_id=hypothesis.hypothesis_id,
                            claim_id=claim.claim_id,
                            query_kind=kind,
                            query_text=cleaned,
                        )
                    )
            if self.include_hypothesis_composite:
                composite = _clean_query(
                    " ".join(
                        [
                            hypothesis.title,
                            hypothesis.hypothesis_statement,
                        ]
                    )
                )
                if composite:
                    key = (hypothesis.hypothesis_id, None, composite.lower())
                    if key not in seen:
                        seen.add(key)
                        queries.append(
                            LiteratureQuery(
                                query_id=_stable_id(
                                    "literature_query",
                                    hypothesis.hypothesis_id,
                                    "composite",
                                    composite,
                                ),
                                hypothesis_id=hypothesis.hypothesis_id,
                                claim_id=None,
                                query_kind="hypothesis_composite",
                                query_text=composite,
                            )
                        )

        payload = {
            "schema_version": "literature-query-plan-v1",
            "source_portfolio_id": portfolio.portfolio_id,
            "queries": [row.model_dump(mode="json") for row in queries],
            "claims": [row.model_dump(mode="json") for row in decompositions],
            "policy_version": "external-novelty-query-policy-v1",
        }
        plan_id = _stable_id(
            "literature_query_plan",
            portfolio.portfolio_id,
            *[row.query_id for row in queries],
        )
        body = {**payload, "plan_id": plan_id}
        return LiteratureQueryPlan(**body, plan_sha256=_sha256_json(body))
