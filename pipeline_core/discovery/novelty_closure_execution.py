from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict

from pipeline_core.discovery.external_novelty_contracts import (
    ClaimPriorArtCandidateSet,
    LiteratureQuery,
    NoveltyClaim,
    PriorArtPacket,
)
from pipeline_core.discovery.novelty_closure_planner import (
    ClosureRetrievalPlan,
    ClosureRetrievalTarget,
)
from pipeline_core.discovery.prior_art_matching import (
    PriorArtRanker,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExecutableClosureTarget(StrictModel):
    target_id: str
    slot: str
    source_claim_id: str
    target_basis: str
    search_terms: tuple[str, ...]
    search_query: str
    source_text: str
    identity_anchor_terms: tuple[str, ...] = ()
    inference_provenance: dict[str, object] | None = None
    evidence_status: str = "UNASSESSED"


class ClosureLiteratureQueryPlan(StrictModel):
    schema_version: str = "closure-literature-query-plan-v1"
    plan_id: str
    plan_sha256: str
    source_portfolio_id: str
    source_hypothesis_id: str
    source_claim_id: str
    queries: list[LiteratureQuery]
    targets: list[ExecutableClosureTarget]
    policy_version: str = "n9-closure-execution-policy-v1"


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(x) for x in parts
    ).encode("utf-8")

    return (
        f"{prefix}:"
        f"{hashlib.sha256(raw).hexdigest()[:length]}"
    )



_SOURCE_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "at",
        "that",
        "how",
        "in",
        "on",
        "for",
        "and",
        "or",
        "as",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
    }
)


def _source_query_tokens(
    text: str,
) -> tuple[str, ...]:
    """Tokenize without adding scientific vocabulary."""

    cleaned = "".join(
        char
        if char.isalnum()
        else " "
        for char in str(text or "")
    )

    return tuple(
        token
        for token in cleaned.split()
        if token
    )


def _query_uses_only_source_vocabulary(
    query: str,
    source_text: str,
) -> bool:
    """Fail closed if a retrieval query adds source-external tokens."""

    query_keys = {
        token.casefold()
        for token in _source_query_tokens(
            query
        )
    }

    source_keys = {
        token.casefold()
        for token in _source_query_tokens(
            source_text
        )
    }

    return bool(
        query_keys
        and query_keys <= source_keys
    )


def _dedupe_query_texts(
    rows: list[str],
) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for row in rows:
        normalized = " ".join(
            str(row or "").split()
        )

        if not normalized:
            continue

        key = normalized.casefold()

        if key in seen:
            continue

        seen.add(key)
        output.append(normalized)

    return output


def _source_preserving_variants_for_target(
    target: ExecutableClosureTarget,
    *,
    max_queries_per_target: int = 3,
) -> tuple[str, ...]:
    """Derive bounded retrieval variants from existing source text only.

    No synonym expansion, mechanism invention, or external vocabulary
    is permitted here. These strings are retrieval formulations only,
    never scientific propositions or evidence.
    """

    candidates: list[str] = [
        target.search_query,
    ]

    terms = tuple(
        " ".join(str(term).split())
        for term in target.search_terms
        if str(term).strip()
    )

    # BASE:
    # remove the generic relation descriptor while preserving the
    # two source variables/entities.
    if (
        target.slot == "BASE_RELATION"
        and len(terms) >= 2
    ):
        candidates.append(
            " ".join(terms[:2])
        )

    # FACTOR:
    # query the atomic identity against each principal relation-side
    # term separately. This broadens retrieval without asserting that
    # either pair is scientifically related.
    elif (
        target.slot
        == "DISTINGUISHING_FACTOR_EFFECT"
        and target.identity_anchor_terms
        and len(terms) >= 3
    ):
        identity = " ".join(
            target.identity_anchor_terms
        )

        identity_keys = {
            term.casefold()
            for term
            in target.identity_anchor_terms
        }

        relation_terms = [
            term
            for term in terms
            if term.casefold()
            not in identity_keys
        ]

        # The final relation descriptor (e.g. "dependence") is kept
        # in the primary query but not required in pairwise variants.
        principal_terms = (
            relation_terms[:-1]
            if len(relation_terms) >= 3
            else relation_terms
        )

        for term in principal_terms[:2]:
            candidates.append(
                f"{identity} {term}"
            )

    # BRIDGE / FULL:
    # Prefer structured terms only when every token is already present
    # in the exact slot source text. These are retrieval formulations,
    # not reconstructed scientific propositions.
    #
    # If structured terms are unavailable, retain the older
    # source-only compression as a fail-safe fallback.
    elif target.slot in {
        "BRIDGE_RELATION",
        "FULL_RELATION",
    }:
        source_tokens = list(
            _source_query_tokens(
                target.source_text
            )
        )

        content_tokens = [
            token
            for token in source_tokens
            if token.casefold()
            not in _SOURCE_QUERY_STOPWORDS
        ]

        identity_keys = {
            str(term).strip().casefold()
            for term
            in target.identity_anchor_terms
            if str(term).strip()
        }

        structured_anchor_terms = [
            term
            for term in terms
            if term.casefold()
            in identity_keys
        ]

        structured_relation_terms = [
            term
            for term in terms
            if term.casefold()
            not in identity_keys
        ]

        if (
            structured_anchor_terms
            and structured_relation_terms
        ):
            identity_text = " ".join(
                structured_anchor_terms
            )

            head_terms = (
                structured_relation_terms[:2]
            )

            tail_terms = (
                structured_relation_terms[-2:]
                if len(
                    structured_relation_terms
                ) > 2
                else structured_relation_terms
            )

            for relation_slice in (
                head_terms,
                tail_terms,
            ):
                structured_query = " ".join(
                    [
                        identity_text,
                        *relation_slice,
                    ]
                )

                if (
                    _query_uses_only_source_vocabulary(
                        structured_query,
                        target.source_text,
                    )
                ):
                    candidates.append(
                        structured_query
                    )

        # Fill any remaining bounded retrieval slots using only the
        # literal source sentence. This preserves behavior for older
        # plans with no structured BRIDGE/FULL search terms.
        if len(
            _dedupe_query_texts(candidates)
        ) < max_queries_per_target:
            if content_tokens:
                candidates.append(
                    " ".join(content_tokens)
                )

        if len(
            _dedupe_query_texts(candidates)
        ) < max_queries_per_target:
            anchor_tokens = list(
                _source_query_tokens(
                    " ".join(
                        target.identity_anchor_terms
                    )
                )
            )

            anchor_keys = {
                token.casefold()
                for token in anchor_tokens
            }

            remainder = [
                token
                for token in content_tokens
                if token.casefold()
                not in anchor_keys
            ]

            if anchor_tokens and remainder:
                head = remainder[:4]
                tail = (
                    remainder[-4:]
                    if len(remainder) > 4
                    else []
                )

                fallback_query = " ".join(
                    anchor_tokens
                    + head
                    + tail
                )

                if (
                    _query_uses_only_source_vocabulary(
                        fallback_query,
                        target.source_text,
                    )
                ):
                    candidates.append(
                        fallback_query
                    )

    return tuple(
        _dedupe_query_texts(
            candidates
        )[: max(
            1,
            int(max_queries_per_target),
        )]
    )


def expand_closure_query_plan_source_preserving(
    *,
    plan: ClosureLiteratureQueryPlan,
    max_queries_per_target: int = 3,
) -> ClosureLiteratureQueryPlan:
    """Expand each closure target into bounded source-preserving queries.

    Target identity and scientific content are unchanged. Only the
    retrieval formulation is expanded.
    """

    queries: list[LiteratureQuery] = []

    for target in plan.targets:
        variants = (
            _source_preserving_variants_for_target(
                target,
                max_queries_per_target=(
                    max_queries_per_target
                ),
            )
        )

        for variant_index, query_text in enumerate(
            variants
        ):
            query_id = _stable_id(
                "closure_query",
                target.target_id,
                query_text,
            )

            queries.append(
                LiteratureQuery(
                    query_id=query_id,
                    hypothesis_id=(
                        plan.source_hypothesis_id
                    ),
                    claim_id=target.target_id,
                    query_kind=(
                        "claim_primary"
                        if variant_index == 0
                        else "claim_variant"
                    ),
                    query_text=query_text,
                )
            )

    body = {
        "schema_version":
            plan.schema_version,
        "source_portfolio_id":
            plan.source_portfolio_id,
        "source_hypothesis_id":
            plan.source_hypothesis_id,
        "source_claim_id":
            plan.source_claim_id,
        "queries": [
            row.model_dump(mode="json")
            for row in queries
        ],
        "targets": [
            row.model_dump(mode="json")
            for row in plan.targets
        ],
        "policy_version":
            "n9-closure-source-preserving-query-expansion-v2",
    }

    plan_sha = _sha256_json(body)

    plan_id = _stable_id(
        "closure_literature_query_plan",
        plan.source_portfolio_id,
        plan.source_claim_id,
        plan_sha,
    )

    return ClosureLiteratureQueryPlan(
        **body,
        plan_id=plan_id,
        plan_sha256=plan_sha,
    )


def build_closure_execution_plan(
    *,
    source_portfolio_id: str,
    closure_plan: ClosureRetrievalPlan,
) -> ClosureLiteratureQueryPlan:
    targets: list[ExecutableClosureTarget] = []
    queries: list[LiteratureQuery] = []

    for target in closure_plan.targets:
        target_id = _stable_id(
            "closure_target",
            closure_plan.claim_id,
            target.slot,
            target.search_query,
        )

        query_id = _stable_id(
            "closure_query",
            target_id,
            target.search_query,
        )

        targets.append(
            ExecutableClosureTarget(
                target_id=target_id,
                slot=target.slot,
                source_claim_id=target.source_claim_id,
                target_basis=target.target_basis,
                inference_provenance=(
                    target.inference_provenance
                ),
                search_terms=target.search_terms,
                search_query=target.search_query,
                source_text=target.source_text,
                identity_anchor_terms=(
                    target.identity_anchor_terms
                ),
            )
        )

        queries.append(
            LiteratureQuery(
                query_id=query_id,
                hypothesis_id=closure_plan.hypothesis_id,
                claim_id=target_id,
                query_kind="claim_primary",
                query_text=target.search_query,
            )
        )

    body = {
        "schema_version":
            "closure-literature-query-plan-v1",
        "source_portfolio_id":
            source_portfolio_id,
        "source_hypothesis_id":
            closure_plan.hypothesis_id,
        "source_claim_id":
            closure_plan.claim_id,
        "queries": [
            row.model_dump(mode="json")
            for row in queries
        ],
        "targets": [
            row.model_dump(mode="json")
            for row in targets
        ],
        "policy_version":
            "n9-closure-execution-policy-v1",
    }

    plan_sha = _sha256_json(body)

    plan_id = _stable_id(
        "closure_literature_query_plan",
        source_portfolio_id,
        closure_plan.claim_id,
        plan_sha,
    )

    return ClosureLiteratureQueryPlan(
        **body,
        plan_id=plan_id,
        plan_sha256=plan_sha,
    )


def build_ranking_surrogate(
    *,
    hypothesis_id: str,
    target: ExecutableClosureTarget,
    rank: int,
) -> NoveltyClaim:
    """Construct a retrieval-ranking surrogate only.

    Its text is search intent, not a scientific proposition and it
    must never be passed to the ordinary claim reviewer.
    """

    concepts = list(
        target.search_terms
    )

    if not concepts:
        concepts = [
            target.search_query
        ]

    return NoveltyClaim(
        claim_id=target.target_id,
        hypothesis_id=hypothesis_id,
        claim_rank=rank,
        kind="composite",
        importance="supporting",
        text=target.search_query,
        rationale=(
            "Closure retrieval-ranking surrogate only; "
            "not a scientific assertion."
        ),
        search_concepts=concepts,
        search_queries=[
            target.search_query
        ],
        distinguishing_terms=[],
        prior_art_identity_terms=[],
        relation_nucleus_terms=[],
    )


def _select_with_abstract_reserve(
    *,
    expanded: ClaimPriorArtCandidateSet,
    packet: PriorArtPacket,
    max_ranked: int,
    min_abstract: int = 3,
    identity_anchor_terms: tuple[str, ...] = (),
    min_identity_abstract: int = 1,
) -> ClaimPriorArtCandidateSet:
    """Keep relevance ranking while reserving abstract-backed evidence.

    This is closure-specific. It does not alter the global prior-art
    ranker. Abstract availability is only a reviewability constraint;
    it is never scientific evidence by itself.
    """

    works = {
        row.work_id: row
        for row in packet.works
    }

    rows = list(
        expanded.ranked_works
    )

    if not rows:
        return expanded

    max_ranked = max(
        1,
        int(max_ranked),
    )

    min_abstract = max(
        0,
        min(
            int(min_abstract),
            max_ranked,
        ),
    )

    selected = list(
        rows[:max_ranked]
    )

    def has_abstract(row) -> bool:
        work = works.get(
            row.work_id
        )
        return bool(
            work
            and work.abstract
        )

    current_abstract = sum(
        1
        for row in selected
        if has_abstract(row)
    )

    needed = max(
        0,
        min_abstract
        - current_abstract,
    )

    if needed:
        selected_ids = {
            row.work_id
            for row in selected
        }

        reserve = [
            row
            for row in rows[max_ranked:]
            if (
                row.work_id
                not in selected_ids
                and has_abstract(row)
            )
        ]

        for candidate in reserve:
            if needed <= 0:
                break

            replace_index = None

            # Replace the lowest-ranked non-abstract candidate first.
            for index in range(
                len(selected) - 1,
                -1,
                -1,
            ):
                if not has_abstract(
                    selected[index]
                ):
                    replace_index = index
                    break

            if replace_index is None:
                break

            selected[
                replace_index
            ] = candidate

            needed -= 1

    # Positive evidence is asymmetric: one explicit abstract-backed
    # identity match may establish a slot. Therefore, for non-base
    # targets, keep at least one identity-anchored abstract available
    # to the reviewer when one exists in the ranked retrieval pool.
    #
    # This is reviewability only. It does NOT increase material or
    # negative-closure coverage by itself.
    def normalized(value: str) -> str:
        return " ".join(
            "".join(
                char.casefold()
                if char.isalnum()
                else " "
                for char in str(value or "")
            ).split()
        )

    normalized_anchors = tuple(
        normalized(anchor)
        for anchor in identity_anchor_terms
        if normalized(anchor)
    )

    def has_identity_abstract(row) -> bool:
        if not normalized_anchors:
            return False

        work = works.get(row.work_id)

        if not work or not work.abstract:
            return False

        abstract = normalized(
            work.abstract
        )

        return any(
            anchor in abstract
            for anchor in normalized_anchors
        )

    current_identity_abstract = sum(
        1
        for row in selected
        if has_identity_abstract(row)
    )

    identity_needed = max(
        0,
        min(
            int(min_identity_abstract),
            max_ranked,
        )
        - current_identity_abstract,
    )

    if identity_needed:
        selected_ids = {
            row.work_id
            for row in selected
        }

        identity_reserve = [
            row
            for row in rows[max_ranked:]
            if (
                row.work_id not in selected_ids
                and has_identity_abstract(row)
            )
        ]

        for candidate in identity_reserve:
            if identity_needed <= 0:
                break

            replace_index = None

            # First replace a title-only, non-identity candidate.
            for index in range(
                len(selected) - 1,
                -1,
                -1,
            ):
                if (
                    not has_abstract(selected[index])
                    and not has_identity_abstract(
                        selected[index]
                    )
                ):
                    replace_index = index
                    break

            # Otherwise replace the lowest-ranked non-identity row.
            if replace_index is None:
                for index in range(
                    len(selected) - 1,
                    -1,
                    -1,
                ):
                    if not has_identity_abstract(
                        selected[index]
                    ):
                        replace_index = index
                        break

            if replace_index is None:
                break

            selected[
                replace_index
            ] = candidate

            identity_needed -= 1

    # Restore original expanded relevance order among retained rows.
    position = {
        row.work_id: index
        for index, row
        in enumerate(rows)
    }

    selected.sort(
        key=lambda row: position[
            row.work_id
        ]
    )

    return ClaimPriorArtCandidateSet(
        hypothesis_id=(
            expanded.hypothesis_id
        ),
        claim_id=(
            expanded.claim_id
        ),
        ranked_works=selected,
    )


def rank_closure_candidates(
    *,
    plan: ClosureLiteratureQueryPlan,
    packet: PriorArtPacket,
    ranker: PriorArtRanker,
) -> dict[str, ClaimPriorArtCandidateSet]:
    if (
        packet.source_query_plan_id
        != plan.plan_id
    ):
        raise ValueError(
            "Closure packet/query-plan provenance mismatch."
        )

    rows: dict[
        str,
        ClaimPriorArtCandidateSet,
    ] = {}

    # Closure review needs abstract-backed metadata to distinguish
    # relation evidence from title similarity. Rank a broader pool
    # with the SAME scoring model, then retain the ordinary top-N
    # while reserving a small number of abstract-backed candidates.
    expanded_ranker = PriorArtRanker(
        ranker.encoder,
        max_ranked_works_per_claim=max(
            24,
            min(
                96,
                len(packet.works),
            ),
            ranker.max_ranked,
        ),
        domain_profile=(
            ranker.domain_profile
        ),
    )

    for rank, target in enumerate(
        plan.targets,
        start=1,
    ):
        surrogate = build_ranking_surrogate(
            hypothesis_id=(
                plan.source_hypothesis_id
            ),
            target=target,
            rank=rank,
        )

        expanded = expanded_ranker.rank(
            surrogate,
            packet,
            plan,
        )

        rows[target.target_id] = (
            _select_with_abstract_reserve(
                expanded=expanded,
                packet=packet,
                max_ranked=(
                    ranker.max_ranked
                ),
                min_abstract=3,
                identity_anchor_terms=(
                    target.identity_anchor_terms
                ),
                min_identity_abstract=3,
            )
        )

    return rows
