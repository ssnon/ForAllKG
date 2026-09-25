from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
    NoveltyClaimDraft,
    NoveltyClaimSemanticFidelityBindingDraft,
)
from pipeline_core.discovery.hypothesis_contracts import (
    FalsificationCriterionDraft,
    HypothesisContext,
    HypothesisEvidenceStatement,
    HypothesisPolicy,
    HypothesisPortfolioDraft,
    HypothesisProposalDraft,
    PredictedObservationDraft,
)
from pipeline_core.discovery.hypothesis_llm import HypothesisDraftGeneration
from pipeline_core.discovery.hypothesis_semantic_contracts import (
    HypothesisSemanticDimensionDraft,
    HypothesisSemanticReviewDraft,
    SEMANTIC_DIMENSIONS,
)
from pipeline_core.discovery.hypothesis_semantic_llm import (
    HypothesisSemanticGeneration,
)
from pipeline_core.discovery.hypothesis_semantic_runtime import (
    HypothesisSemanticCriticRuntime,
)
from pipeline_core.discovery.pre_n10_regeneration_reentry_v1 import (
    execute_pre_n10_regeneration_reentry_v1,
)
from pipeline_core.discovery.pre_n10_regeneration_v1 import (
    execute_pre_n10_regeneration_v1,
)
from pipeline_core.discovery.pre_n10_source_alignment_primary_v1 import (
    PreN10SourceAlignmentHypothesisResultV1,
    PreN10SourceAlignmentPrimaryReportV1,
)
from pipeline_core.discovery.prospective_regeneration_unit_v2 import (
    build_regeneration_unit_v2_freeze,
)


def _sha(value):
    import hashlib
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id="context:1",
        context_sha256="a" * 64,
        source_packet_id="packet:1",
        source_packet_sha256="b" * 64,
        source_report_id="report:1",
        source_report_sha256="c" * 64,
        task_id="task:1",
        question="How are X and Y related?",
        corpus_id="corpus:1",
        domain_profile_id="domain:1",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="statement:1",
                text="X is associated with Y.",
                epistemic_role="reported",
                claim_kind="reported_relation",
                paper_ids=["paper:1"],
                eligible_as_premise=True,
            )
        ],
        policy=HypothesisPolicy(),
    )


def _draft(*, hard_gate_bad: bool = False) -> HypothesisPortfolioDraft:
    premise_ids = [] if hard_gate_bad else ["statement:1"]
    return HypothesisPortfolioDraft(
        hypotheses=[
            HypothesisProposalDraft(
                local_id="h1",
                title="X-Y relation",
                hypothesis_statement="X modulates Y.",
                hypothesis_type="mechanistic_extension",
                premise_statement_ids=premise_ids,
                inferential_bridge="X may modulate Y.",
                predicted_observations=[
                    PredictedObservationDraft(
                        local_id="p1",
                        observable="Y response under X",
                        expected_direction="qualitative_change",
                        rationale="Tests X-dependent Y response.",
                    )
                ],
                falsification_criteria=[
                    FalsificationCriterionDraft(
                        local_id="f1",
                        observable="Y response under X",
                        falsifying_outcome=(
                            "Y response under X is unchanged"
                        ),
                    )
                ],
            )
        ]
    )


class _GenerationBackend:
    backend_name = "generation"
    model_name = "generation"

    def __init__(self, draft):
        self.draft = draft

    def generate(self, prompt):
        return HypothesisDraftGeneration(
            draft=self.draft,
            input_tokens=1,
            output_tokens=1,
        )


class _SemanticBackend:
    backend_name = "semantic"
    model_name = "semantic"

    def __init__(self, portfolio, *, accept=True):
        self.portfolio = portfolio
        self.accept = accept
        self.calls = 0

    def review(self, prompt):
        self.calls += 1
        hid = self.portfolio.hypotheses[0].hypothesis_id
        rows = []
        for dimension in SEMANTIC_DIMENSIONS:
            verdict = (
                "not_applicable"
                if dimension == "hypothesis_distinctness"
                else ("pass" if self.accept else "fail")
            )
            rows.append(
                HypothesisSemanticDimensionDraft(
                    dimension=dimension,
                    verdict=verdict,
                    rationale="fixture",
                    hypothesis_ids=(
                        []
                        if dimension == "abstention_appropriateness"
                        else [hid]
                    ),
                    statement_ids=(
                        ["statement:1"]
                        if dimension == "premise_fidelity"
                        else []
                    ),
                )
            )
        return HypothesisSemanticGeneration(
            draft=HypothesisSemanticReviewDraft(
                dimensions=rows,
                overall_summary="fixture",
            ),
            input_tokens=1,
            output_tokens=1,
        )


class _DecompositionBackend:
    backend_name = "decomposition"
    model_name = "decomposition"

    def __init__(self, *, malformed=False):
        self.malformed = malformed
        self.calls = 0

    def decompose(self, hypothesis, *, max_claims):
        self.calls += 1
        prediction = hypothesis.predicted_observations[0]
        falsifier = hypothesis.falsification_criteria[0]
        return NoveltyClaimDecompositionDraft(
            claims=[
                NoveltyClaimDraft(
                    local_id="c1",
                    kind="mechanistic_link",
                    importance="core",
                    novelty_selection_role="NOVELTY_BEARING",
                    text="X modulates Y.",
                    rationale="Bounded relation.",
                    search_concepts=["X", "Y"],
                    search_queries=["X Y modulation"],
                    prior_art_identity_terms=["Y"],
                    relation_nucleus_terms=["X", "Y", "modulates"],
                    semantic_fidelity_binding=(
                        NoveltyClaimSemanticFidelityBindingDraft(
                            proposition_basis="X modulates Y.",
                            relation_endpoint_anchors=["X", "Y"],
                            prediction_observation_id=(
                                prediction.observation_id
                            ),
                            falsification_criterion_id=(
                                falsifier.criterion_id
                            ),
                        )
                    ),
                    required_bridge=(
                        ""
                        if self.malformed
                        else "X may modulate Y."
                    ),
                    predicted_observation=prediction.observable,
                    falsification_condition=falsifier.falsifying_outcome,
                )
            ]
        )


def _primary():
    hypothesis = PreN10SourceAlignmentHypothesisResultV1(
        hypothesis_id="hypothesis:source",
        source_contract_status="REQUIRES_PRE_N10_INTERVENTION",
        source_alignment_results=[],
        post_contract_status="REQUIRES_PRE_N10_INTERVENTION",
        recovered_for_n10=False,
        regeneration_fallback_required=True,
    )
    body = {
        "schema_version":
            "pre-n10-source-alignment-primary-report-v1",
        "source_contract_report_id": "contract:1",
        "source_contract_report_sha256": "d" * 64,
        "source_query_plan_id": "plan:1",
        "source_query_plan_sha256": "e" * 64,
        "aligned_query_plan_id": "plan:2",
        "aligned_query_plan_sha256": "f" * 64,
        "post_contract_report_id": "contract:2",
        "post_contract_report_sha256": "1" * 64,
        "hypotheses": [hypothesis.model_dump(mode="json")],
        "hypothesis_count": 1,
        "materialized_alignment_count": 0,
        "unavailable_alignment_count": 0,
        "ambiguous_alignment_count": 0,
        "semantic_reject_count": 0,
        "audit_failure_count": 0,
        "recovered_for_n10_count": 0,
        "regeneration_fallback_required_count": 1,
        "audit_llm_call_count": 0,
        "claim_generation_llm_calls": 0,
        "retrieval_performed": False,
        "novelty_assessment_performed": False,
        "n9_performed": False,
        "n10_performed": False,
        "regeneration_performed": False,
        "endpoint_binding_performed": False,
        "verifier_performed": False,
        "production_selection_changed": False,
        "canonical_graph_mutated": False,
    }
    digest = _sha(body)
    return PreN10SourceAlignmentPrimaryReportV1(
        **body,
        report_id=(
            "pre_n10_source_alignment_primary_v1:"
            + digest[:20]
        ),
        report_sha256=digest,
    )


def _regenerate(tmp_path: Path, *, draft=None):
    context = _context()
    freeze = build_regeneration_unit_v2_freeze(
        repository_head_sha="2" * 40,
        repository_tracked_worktree_dirty=False,
    )
    report, _ = execute_pre_n10_regeneration_v1(
        context=context,
        primary_report=_primary(),
        unit_freeze=freeze,
        backend_factory=lambda _source, _dir: _GenerationBackend(
            draft or _draft()
        ),
        output_root=tmp_path / "regen",
    )
    return context, report


def test_reentry_semantic_acceptance_then_contract_ready(
    tmp_path: Path,
):
    context, regeneration = _regenerate(tmp_path)
    portfolio_path = Path(
        regeneration.lineages[0].regenerated_portfolio_path
    )
    from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    semantic_backend = _SemanticBackend(portfolio, accept=True)
    decomposition_backend = _DecompositionBackend(malformed=False)

    report, _ = execute_pre_n10_regeneration_reentry_v1(
        context=context,
        regeneration_report=regeneration,
        semantic_runner_factory=lambda _source, _dir: (
            HypothesisSemanticCriticRuntime(semantic_backend)
        ),
        decomposition_backend_factory=lambda _source, _dir: (
            decomposition_backend
        ),
        output_root=tmp_path / "reentry",
    )

    assert semantic_backend.calls == 1
    assert decomposition_backend.calls == 1
    assert report.semantic_accepted_count == 1
    assert report.claim_decomposition_request_count == 1
    assert report.ready_for_n10_count == 1
    assert report.intervention_required_count == 0
    assert report.lineages[0].final_status == "PRE_N10_READY"
    assert report.external_novelty_performed is False
    assert report.n10_performed is False


def test_reentry_semantic_acceptance_but_contract_failure_stays_before_n10(
    tmp_path: Path,
):
    context, regeneration = _regenerate(tmp_path)
    portfolio_path = Path(
        regeneration.lineages[0].regenerated_portfolio_path
    )
    from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolio
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    semantic_backend = _SemanticBackend(portfolio, accept=True)
    decomposition_backend = _DecompositionBackend(malformed=True)

    report, _ = execute_pre_n10_regeneration_reentry_v1(
        context=context,
        regeneration_report=regeneration,
        semantic_runner_factory=lambda _source, _dir: (
            HypothesisSemanticCriticRuntime(semantic_backend)
        ),
        decomposition_backend_factory=lambda _source, _dir: (
            decomposition_backend
        ),
        output_root=tmp_path / "reentry",
    )

    assert report.semantic_accepted_count == 1
    assert report.ready_for_n10_count == 0
    assert report.intervention_required_count == 1
    assert (
        report.lineages[0].final_status
        == "PRE_N10_INTERVENTION_REQUIRED"
    )
    assert report.n10_performed is False


def test_reentry_hard_gate_failure_skips_claim_decomposition(
    tmp_path: Path,
):
    context, regeneration = _regenerate(tmp_path)
    portfolio_path = Path(
        regeneration.lineages[0].regenerated_portfolio_path
    )
    from pipeline_core.discovery.hypothesis_contracts import (
        HypothesisEvidenceProfile,
        HypothesisPortfolio,
    )
    portfolio = HypothesisPortfolio.model_validate_json(
        portfolio_path.read_text(encoding="utf-8")
    )
    card = portfolio.hypotheses[0]
    bad_profile = HypothesisEvidenceProfile(
        premise_count=0,
        gap_count=0,
        source_paper_count=0,
        candidate_premise_count=0,
        reported_premise_count=0,
        synthesis_premise_count=0,
    )
    bad_card = card.model_copy(
        update={
            "premise_statement_ids": [],
            "source_paper_ids": [],
            "evidence_profile": bad_profile,
        }
    )
    bad_portfolio = portfolio.model_copy(
        update={"hypotheses": [bad_card]}
    )
    portfolio_path.write_text(
        bad_portfolio.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    class _ShouldNotDecompose:
        def decompose(self, hypothesis, *, max_claims):
            raise AssertionError("decomposition must not run")

    class _SemanticMustNotCall:
        backend_name = "semantic"
        model_name = "semantic"
        def review(self, prompt):
            raise AssertionError(
                "semantic LLM must not run after deterministic hard gate"
            )

    report, _ = execute_pre_n10_regeneration_reentry_v1(
        context=context,
        regeneration_report=regeneration,
        semantic_runner_factory=lambda _source, _dir: (
            HypothesisSemanticCriticRuntime(_SemanticMustNotCall())
        ),
        decomposition_backend_factory=lambda _source, _dir: (
            _ShouldNotDecompose()
        ),
        output_root=tmp_path / "reentry",
    )

    assert report.semantic_terminal_count == 1
    assert report.semantic_critic_llm_invocation_count == 0
    assert report.claim_decomposition_request_count == 0
    assert report.ready_for_n10_count == 0
    assert (
        report.lineages[0].final_status
        == "SEMANTIC_HARD_GATE_FAILED"
    )
