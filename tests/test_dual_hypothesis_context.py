from __future__ import annotations

from dac_her.discovery_contracts import DiscoveryBundle, DiscoveryInspiration, DiscoveryScoreBreakdown
from dac_her.discovery_hypothesis_prompt import DiscoveryAwareHypothesisPromptAssembler
from dac_her.dual_hypothesis_context import DualHypothesisContext
from dac_her.hypothesis_contracts import HypothesisContext, HypothesisEvidenceStatement


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id="ctx",
        context_sha256="ctxsha",
        source_packet_id="packet",
        source_packet_sha256="packetsha",
        source_report_id="report",
        source_report_sha256="reportsha",
        task_id="task",
        question="How does X affect Y?",
        corpus_id="c1",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="s1",
                text="X changes descriptor D.",
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["A"],
                scientific_support_node_ids=["paper::A::m1"],
                eligible_as_premise=True,
            )
        ],
    )


def _bundle() -> DiscoveryBundle:
    inspiration = DiscoveryInspiration(
        inspiration_id="i1",
        source_path_id="p1",
        source_corpus_id="c1",
        source_mode="exploratory",
        path_type="CROSS_PAPER_MECHANISTIC",
        paper_ids=["A", "B"],
        rendered_path="X --MODULATES--> Z --INFLUENCES--> Y",
        exploration_score=0.8,
        score_breakdown=DiscoveryScoreBreakdown(
            endpoint_relevance=0.8,
            mechanistic_content=1.0,
            cross_paper_span=0.5,
            community_span=0.5,
            relation_rarity=0.5,
            exploratory_mode_bonus=1.0,
            grounding_redundancy_penalty=0.0,
            navigation_burden_penalty=0.0,
            reverse_burden_penalty=0.0,
            total=0.8,
        ),
    )
    return DiscoveryBundle(
        bundle_id="b1",
        bundle_sha256="bsha",
        corpus_id="c1",
        query_signature="X||Y",
        inspirations=[inspiration],
        candidate_count=1,
        selected_count=1,
        used_candidate_pool=True,
    )


def test_dual_context_preserves_grounded_contract_and_prompt_separation() -> None:
    dual = DualHypothesisContext.build(_context(), _bundle())
    assert dual.grounded_context.evidence_statements[0].statement_id == "s1"
    prompt = DiscoveryAwareHypothesisPromptAssembler(
        dual.discovery_bundle,
        max_hypotheses=2,
    ).build(dual.grounded_context)
    assert "DISCOVERY INSPIRATIONS (NOT POSITIVE PREMISES)" in prompt.user_prompt
    assert "eligible_as_positive_premise=false" in prompt.user_prompt
    assert "i1" in prompt.user_prompt
    assert "X changes descriptor D." in prompt.user_prompt
