from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dac_her.discovery_contracts import DiscoveryBundle, DiscoveryInspiration, DiscoveryScoreBreakdown
from dac_her.dual_hypothesis_context import DualHypothesisContext
from dac_her.hypothesis_contracts import (
    FalsificationCriterion,
    HypothesisCard,
    HypothesisContext,
    HypothesisEvidenceProfile,
    HypothesisEvidenceStatement,
    HypothesisPortfolio,
    HypothesisRouteContext,
    PredictedObservation,
)
from dac_her.internal_novelty import InternalNoveltyAssessor


@dataclass
class FakeMatch:
    node_id: str
    label: str
    semantic_similarity: float
    source_paper_id: str = "A"
    node_type: str = "Mechanism"
    requires_verification: bool = False


class FakeEncoder:
    def encode_query(self, text: str) -> np.ndarray:
        if "different" in text.lower():
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeMapper:
    encoder = FakeEncoder()

    def __init__(self, similarity: float) -> None:
        self.similarity = similarity

    def map(self, concept):
        return [FakeMatch("paper::A::claim", "Existing claim", self.similarity)]


def _dual() -> DualHypothesisContext:
    context = HypothesisContext(
        context_id="ctx",
        context_sha256="ctxsha",
        source_packet_id="p",
        source_packet_sha256="psha",
        source_report_id="r",
        source_report_sha256="rsha",
        task_id="task",
        question="q",
        corpus_id="c1",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="s1",
                text="A premise",
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["A"],
                scientific_support_node_ids=["paper::A::claim"],
                eligible_as_premise=True,
            )
        ],
        mechanism_routes=[
            HypothesisRouteContext(
                route_id="route1",
                statement_ids=["s1"],
                paper_ids=["A"],
                structural_type="DIRECT_MECHANISTIC",
            )
        ],
    )
    score = DiscoveryScoreBreakdown(
        endpoint_relevance=1,
        mechanistic_content=1,
        cross_paper_span=1,
        community_span=1,
        relation_rarity=1,
        exploratory_mode_bonus=0,
        grounding_redundancy_penalty=0,
        navigation_burden_penalty=0,
        reverse_burden_penalty=0,
        total=1,
    )
    bundle = DiscoveryBundle(
        bundle_id="b",
        bundle_sha256="bsha",
        corpus_id="c1",
        query_signature="q",
        inspirations=[
            DiscoveryInspiration(
                inspiration_id="i",
                source_path_id="path",
                source_corpus_id="c1",
                source_mode="mechanism",
                path_type="CROSS_PAPER_MECHANISTIC",
                paper_ids=["A", "B"],
                rendered_path="A premise causes an additional mechanism",
                exploration_score=1,
                score_breakdown=score,
            )
        ],
        candidate_count=1,
        selected_count=1,
        used_candidate_pool=True,
    )
    return DualHypothesisContext.build(context, bundle)


def _portfolio(statement: str = "A premise causes an effect") -> HypothesisPortfolio:
    card = HypothesisCard(
        hypothesis_id="h1",
        source_context_id="ctx",
        source_context_sha256="ctxsha",
        source_report_id="r",
        source_report_sha256="rsha",
        title="H",
        hypothesis_statement=statement,
        hypothesis_type="mechanistic_extension",
        premise_statement_ids=["s1"],
        inferential_bridge="bridge",
        predicted_observations=[
            PredictedObservation(
                observation_id="o1",
                observable="obs",
                expected_direction="qualitative_change",
                rationale="r",
            )
        ],
        falsification_criteria=[
            FalsificationCriterion(
                criterion_id="f1",
                observable="obs",
                falsifying_outcome="no change",
            )
        ],
        source_paper_ids=["A"],
        evidence_profile=HypothesisEvidenceProfile(
            premise_count=1,
            gap_count=0,
            source_paper_count=1,
            candidate_premise_count=0,
            reported_premise_count=1,
            synthesis_premise_count=0,
        ),
    )
    return HypothesisPortfolio(
        portfolio_id="portfolio",
        source_context_id="ctx",
        source_context_sha256="ctxsha",
        source_report_id="r",
        source_report_sha256="rsha",
        hypotheses=[card],
    )


def test_internal_novelty_detects_near_duplicate() -> None:
    report = InternalNoveltyAssessor().assess(_dual(), _portfolio(), FakeMapper(0.93))
    assert report.cards[0].status == "reconstructs_existing_corpus_claim"
    assert report.external_novelty_status == "not_assessed"


def test_internal_novelty_detects_existing_single_paper_chain() -> None:
    report = InternalNoveltyAssessor(
        node_near_duplicate_threshold=0.99,
        node_extension_threshold=0.99,
    ).assess(_dual(), _portfolio(), FakeMapper(0.50))
    assert report.cards[0].status == "reconstructs_existing_corpus_chain"
