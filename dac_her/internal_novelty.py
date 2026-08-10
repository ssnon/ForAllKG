from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dac_her.discovery_contracts import DiscoveryBundle
from dac_her.dual_hypothesis_context import DualHypothesisContext
from dac_her.hypothesis_contracts import HypothesisCard, HypothesisPortfolio
from dac_her.node_mapping import QueryConcept


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


InternalNoveltyStatus = Literal[
    "reconstructs_existing_corpus_claim",
    "reconstructs_existing_corpus_chain",
    "corpus_supported_extension",
    "new_combination_within_corpus",
    "corpus_distinct_candidate",
    "insufficient_internal_evidence",
]


class InternalNoveltyNodeMatch(StrictModel):
    node_id: str
    label: str
    source_paper_id: str = ""
    semantic_similarity: float
    node_type: str = ""
    requires_verification: bool = False


class InternalNoveltyRouteMatch(StrictModel):
    route_id: str
    paper_ids: list[str] = Field(default_factory=list)
    premise_coverage: float
    matched_statement_ids: list[str] = Field(default_factory=list)
    single_paper: bool


class InternalNoveltyCard(StrictModel):
    hypothesis_id: str
    status: InternalNoveltyStatus
    external_novelty_status: Literal["not_assessed"] = "not_assessed"
    max_node_similarity: float = 0.0
    strongest_node_matches: list[InternalNoveltyNodeMatch] = Field(default_factory=list)
    strongest_route_match: InternalNoveltyRouteMatch | None = None
    source_paper_count: int = 0
    dominant_premise_paper_fraction: float = 0.0
    discovery_inspiration_similarity: float = 0.0
    discovery_inspiration_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    interpretation: str


class InternalNoveltyReport(StrictModel):
    schema_version: Literal["internal-novelty-report-v1"] = "internal-novelty-report-v1"
    report_id: str
    report_sha256: str
    source_dual_context_id: str
    source_dual_context_sha256: str
    source_portfolio_id: str
    corpus_id: str
    cards: list[InternalNoveltyCard] = Field(default_factory=list)
    status_counts: dict[str, int] = Field(default_factory=dict)
    external_novelty_status: Literal["not_assessed"] = "not_assessed"
    policy_version: Literal["internal-novelty-policy-v1"] = "internal-novelty-policy-v1"


class MapperProtocol(Protocol):
    encoder: Any

    def map(self, concept: QueryConcept) -> list[Any]: ...


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:length]}"


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalize_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"expected 1D embedding, got {array.shape}")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return array
    return array / norm


def _discovery_similarity(
    text: str,
    bundle: DiscoveryBundle,
    mapper: MapperProtocol,
) -> tuple[float, str | None]:
    if not bundle.inspirations:
        return 0.0, None
    query = _normalize_vector(mapper.encoder.encode_query(text))
    texts = [item.rendered_path for item in bundle.inspirations]
    encode_documents = getattr(mapper.encoder, "encode_documents", None)
    if callable(encode_documents):
        matrix = np.asarray(encode_documents(texts, batch_size=max(1, min(32, len(texts)))), dtype=np.float32)
        candidates = [_normalize_vector(row) for row in matrix]
    else:
        candidates = [
            _normalize_vector(mapper.encoder.encode_query(item.rendered_path))
            for item in bundle.inspirations
        ]
    best = 0.0
    best_id: str | None = None
    for item, candidate in zip(bundle.inspirations, candidates, strict=True):
        score = float(np.dot(query, candidate)) if query.size and candidate.size else 0.0
        if score > best:
            best = score
            best_id = item.inspiration_id
    return max(0.0, min(1.0, best)), best_id


def _route_match(
    hypothesis: HypothesisCard,
    dual: DualHypothesisContext,
) -> InternalNoveltyRouteMatch | None:
    premises = set(map(str, hypothesis.premise_statement_ids))
    if not premises:
        return None
    best: InternalNoveltyRouteMatch | None = None
    for route in dual.grounded_context.mechanism_routes:
        matched = sorted(premises & set(map(str, route.statement_ids)))
        coverage = len(matched) / len(premises)
        card = InternalNoveltyRouteMatch(
            route_id=route.route_id,
            paper_ids=list(route.paper_ids),
            premise_coverage=coverage,
            matched_statement_ids=matched,
            single_paper=len(set(route.paper_ids)) == 1,
        )
        if best is None or (card.premise_coverage, card.single_paper) > (
            best.premise_coverage,
            best.single_paper,
        ):
            best = card
    return best


def _dominant_premise_paper_fraction(
    hypothesis: HypothesisCard,
    dual: DualHypothesisContext,
) -> float:
    statements = {
        row.statement_id: row
        for row in dual.grounded_context.evidence_statements
    }
    counts: Counter[str] = Counter()
    premise_count = 0
    for statement_id in hypothesis.premise_statement_ids:
        row = statements.get(statement_id)
        if row is None:
            continue
        premise_count += 1
        for paper_id in set(row.paper_ids):
            counts[str(paper_id)] += 1
    if premise_count == 0 or not counts:
        return 0.0
    return max(counts.values()) / premise_count


class InternalNoveltyAssessor:
    """Corpus-internal prior-art check; never makes external novelty claims.

    It combines three deterministic signals:
    1. semantic near-duplication against indexed corpus nodes,
    2. reconstruction of an already exposed single-paper mechanism route,
    3. whether a multi-paper hypothesis aligns with a discovery inspiration.

    The statuses mean only "relative to this frozen corpus". External literature
    novelty remains explicitly not_assessed.
    """

    def __init__(
        self,
        *,
        node_near_duplicate_threshold: float = 0.88,
        node_extension_threshold: float = 0.80,
        route_reconstruction_threshold: float = 0.80,
        route_extension_threshold: float = 0.50,
        discovery_alignment_threshold: float = 0.70,
        top_k_nodes: int = 12,
    ) -> None:
        self.node_near_duplicate_threshold = float(node_near_duplicate_threshold)
        self.node_extension_threshold = float(node_extension_threshold)
        self.route_reconstruction_threshold = float(route_reconstruction_threshold)
        self.route_extension_threshold = float(route_extension_threshold)
        self.discovery_alignment_threshold = float(discovery_alignment_threshold)
        self.top_k_nodes = int(top_k_nodes)

    def _node_matches(
        self,
        hypothesis: HypothesisCard,
        mapper: MapperProtocol,
    ) -> list[InternalNoveltyNodeMatch]:
        matches = mapper.map(
            QueryConcept(
                text=hypothesis.hypothesis_statement,
                allow_candidates=True,
                allow_alignment_hubs=False,
                top_k=self.top_k_nodes,
                min_similarity=-1.0,
            )
        )
        rows = [
            InternalNoveltyNodeMatch(
                node_id=str(_get(match, "node_id", "")),
                label=str(_get(match, "label", "")),
                source_paper_id=str(_get(match, "source_paper_id", "")),
                semantic_similarity=float(_get(match, "semantic_similarity", 0.0) or 0.0),
                node_type=str(_get(match, "node_type", "")),
                requires_verification=bool(_get(match, "requires_verification", False)),
            )
            for match in matches
        ]
        return sorted(rows, key=lambda row: (-row.semantic_similarity, row.node_id))

    def _card(
        self,
        hypothesis: HypothesisCard,
        dual: DualHypothesisContext,
        mapper: MapperProtocol,
    ) -> InternalNoveltyCard:
        node_matches = self._node_matches(hypothesis, mapper)
        max_node = node_matches[0].semantic_similarity if node_matches else 0.0
        route = _route_match(hypothesis, dual)
        route_coverage = route.premise_coverage if route is not None else 0.0
        dominant_fraction = _dominant_premise_paper_fraction(hypothesis, dual)
        discovery_similarity, discovery_id = _discovery_similarity(
            hypothesis.hypothesis_statement,
            dual.discovery_bundle,
            mapper,
        )
        source_papers = sorted(set(map(str, hypothesis.source_paper_ids)))

        reasons: list[str] = []
        if max_node >= self.node_near_duplicate_threshold:
            reasons.append("near_duplicate_corpus_node")
        elif max_node >= self.node_extension_threshold:
            reasons.append("strong_corpus_node_overlap")

        if route is not None and route.single_paper and route_coverage >= self.route_reconstruction_threshold:
            reasons.append("single_paper_route_reconstruction")
        elif route is not None and route_coverage >= self.route_extension_threshold:
            reasons.append("partial_existing_route_reconstruction")

        if len(source_papers) >= 2:
            reasons.append("multi_paper_positive_premises")
        if discovery_similarity >= self.discovery_alignment_threshold:
            reasons.append("aligned_with_discovery_inspiration")
        if dominant_fraction >= 0.999 and hypothesis.premise_statement_ids:
            reasons.append("premises_dominated_by_one_paper")

        if max_node >= self.node_near_duplicate_threshold:
            status: InternalNoveltyStatus = "reconstructs_existing_corpus_claim"
            interpretation = (
                "The hypothesis is semantically very close to an indexed corpus claim/node. "
                "Treat it as corpus-internal prior art unless the proposed condition or prediction materially differs."
            )
        elif route is not None and route.single_paper and route_coverage >= self.route_reconstruction_threshold:
            status = "reconstructs_existing_corpus_chain"
            interpretation = (
                "Most selected premises already occur together in one exposed mechanism route from one paper. "
                "The hypothesis mainly reconstructs an existing corpus chain."
            )
        elif max_node >= self.node_extension_threshold or route_coverage >= self.route_extension_threshold:
            status = "corpus_supported_extension"
            interpretation = (
                "The hypothesis extends a strongly overlapping corpus claim/route rather than forming a clearly distinct internal combination."
            )
        elif len(source_papers) >= 2 and discovery_similarity >= self.discovery_alignment_threshold:
            status = "new_combination_within_corpus"
            interpretation = (
                "No strong single-node/single-route reconstruction was detected, while the hypothesis combines multiple grounded papers "
                "and aligns with a separate discovery inspiration. This is a corpus-internal combination signal, not an external novelty claim."
            )
        elif node_matches:
            status = "corpus_distinct_candidate"
            interpretation = (
                "The hypothesis is not a near-duplicate of the strongest indexed nodes or exposed routes under current thresholds. "
                "It is only distinct relative to this corpus; external novelty remains unassessed."
            )
        else:
            status = "insufficient_internal_evidence"
            interpretation = (
                "The corpus index produced no usable semantic comparison, so internal novelty could not be assessed reliably."
            )

        return InternalNoveltyCard(
            hypothesis_id=hypothesis.hypothesis_id,
            status=status,
            max_node_similarity=max_node,
            strongest_node_matches=node_matches[:5],
            strongest_route_match=route,
            source_paper_count=len(source_papers),
            dominant_premise_paper_fraction=dominant_fraction,
            discovery_inspiration_similarity=discovery_similarity,
            discovery_inspiration_id=discovery_id,
            reason_codes=sorted(set(reasons)),
            interpretation=interpretation,
        )

    def assess(
        self,
        dual: DualHypothesisContext,
        portfolio: HypothesisPortfolio,
        mapper: MapperProtocol,
    ) -> InternalNoveltyReport:
        if portfolio.source_context_id != dual.grounded_context.context_id:
            raise ValueError("portfolio source_context_id does not match grounded context")
        if portfolio.source_context_sha256 != dual.grounded_context.context_sha256:
            raise ValueError("portfolio source_context_sha256 does not match grounded context")

        cards = [self._card(hypothesis, dual, mapper) for hypothesis in portfolio.hypotheses]
        counts: defaultdict[str, int] = defaultdict(int)
        for card in cards:
            counts[card.status] += 1

        report_id = _stable_id(
            "internal_novelty_report",
            dual.dual_context_sha256,
            portfolio.portfolio_id,
            *[f"{card.hypothesis_id}:{card.status}:{card.max_node_similarity:.6f}" for card in cards],
        )
        payload = {
            "schema_version": "internal-novelty-report-v1",
            "report_id": report_id,
            "source_dual_context_id": dual.dual_context_id,
            "source_dual_context_sha256": dual.dual_context_sha256,
            "source_portfolio_id": portfolio.portfolio_id,
            "corpus_id": dual.grounded_context.corpus_id,
            "cards": [card.model_dump(mode="json") for card in cards],
            "status_counts": dict(sorted(counts.items())),
            "external_novelty_status": "not_assessed",
            "policy_version": "internal-novelty-policy-v1",
        }
        return InternalNoveltyReport(**payload, report_sha256=_sha256_json(payload))
