from __future__ import annotations

from dac_her.hypothesis_benchmark_evaluator import HypothesisBenchmarkEvaluator
from pipeline_core.discovery.hypothesis_compiler import HypothesisCompiler
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_semantic_contracts import SEMANTIC_DIMENSIONS
from dac_her.hypothesis_semantic_prompt import HypothesisSemanticPromptAssembler


def fixture():
    context = HypothesisContext(
        context_id="ctx",
        context_sha256="csha",
        source_packet_id="packet",
        source_packet_sha256="psha",
        source_report_id="report",
        source_report_sha256="rsha",
        task_id="task",
        question="How might coordination affect adsorption?",
        corpus_id="corpus",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="s:1",
                text="Coordination changes adsorption geometry.",
                epistemic_role="reported",
                claim_kind="mechanism",
                paper_ids=["Kiwook_1"],
                scientific_support_node_ids=["n:1"],
                eligible_as_premise=True,
            )
        ],
    )
    draft = HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h",
                title="h",
                hypothesis_statement="Coordination may alter adsorption.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=["s:1"],
                inferential_bridge="A hypothetical electronic bridge may mediate the relation.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p",
                        observable="adsorption response",
                        expected_direction="qualitative_change",
                        rationale="linked prediction",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f",
                        observable="adsorption response",
                        falsifying_outcome="no response",
                    )
                ],
            )
        ]
    )
    portfolio = HypothesisCompiler().compile(context, draft)
    evaluation = HypothesisBenchmarkEvaluator().evaluate(context, portfolio)
    return context, portfolio, evaluation


def test_prompt_contains_core_critic_invariant_and_all_dimensions():
    context, portfolio, evaluation = fixture()
    prompt = HypothesisSemanticPromptAssembler().build(context, portfolio, evaluation)
    assert "Evaluate the hypothesis; do not repair or rewrite it." in prompt.system_prompt
    assert "A hypothesis is not evidence." in prompt.system_prompt
    assert "do NOT penalize a hypothesis merely because" in prompt.system_prompt
    for dimension in SEMANTIC_DIMENSIONS:
        assert dimension in prompt.system_prompt
        assert dimension in prompt.user_prompt
