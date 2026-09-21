from __future__ import annotations

import pytest

from pipeline_core.discovery.reframing.cross_lane_synthesis import (
    CrossLaneSynthesisBatchDraft,
    CrossLaneSynthesisGeneration,
    CrossLaneSynthesisRuntime,
    CrossLaneSynthesizedHypothesisDraft,
    SynthesisDiscriminatingTestDraft,
    SynthesisFalsifierDraft,
    SynthesisPredictionDraft,
    build_cross_lane_synthesis_prompt,
    compile_cross_lane_synthesis,
)
from pipeline_core.discovery.reframing.production_candidate_contract import (
    ProductionCandidateFalsifier,
    ProductionCandidatePrediction,
    ProductionFacingScientificCandidatePortfolio,
    ProductionScientificCandidate,
    ProductionCandidateLineageValidation,
)


def _rel(candidate_id: str = "pc:rel") -> ProductionScientificCandidate:
    return ProductionScientificCandidate(
        candidate_id=candidate_id,
        source_entry_id=f"entry:{candidate_id}",
        source_lane="RELATIONAL_DISCOVERY",
        source_object_kind="relational_hypothesis",
        source_schema_version="hypothesis-card-v1",
        source_object_id=f"hypothesis:{candidate_id}",
        source_epistemic_status="grounded",
        title="Matrix loading controls signal stability",
        reasoning_label="descriptor_interaction",
        scientific_proposal="Matrix loading changes SERS signal stability through aggregation-dependent response.",
        reasoning_rationale="Observed loading and stability relations motivate a conditional interaction.",
        premise_statement_ids=["p1", "p2"],
        gap_statement_ids=["g1"],
        assumptions=["matrix composition is otherwise comparable"],
        predictions=[
            ProductionCandidatePrediction(
                observable="signal variance",
                expected_direction="qualitative_change",
                rationale="variance should change across loading regimes",
            )
        ],
        falsifiers=[
            ProductionCandidateFalsifier(
                observable="signal variance",
                falsifying_outcome="variance is invariant across matched loading regimes",
            )
        ],
    )


def _ref(candidate_id: str = "pc:ref") -> ProductionScientificCandidate:
    return ProductionScientificCandidate(
        candidate_id=candidate_id,
        source_entry_id=f"entry:{candidate_id}",
        source_lane="SCIENTIFIC_REFRAMING",
        source_object_kind="scientific_reframe",
        source_schema_version="scientific-reframe-candidate-v1",
        source_object_id=f"scientific_reframe:{candidate_id}",
        source_epistemic_status="grounded_shadow",
        title="Accessible hotspot fraction is latent",
        reasoning_label="LATENT_VARIABLE",
        scientific_proposal="Accessible hotspot fraction mediates the observed matrix-loading response.",
        reasoning_rationale="Bulk loading may not equal the locally accessible hotspot population.",
        premise_statement_ids=["p2", "p3"],
        gap_statement_ids=["g1"],
        assumptions=["local accessibility can vary independently of bulk loading"],
        predictions=[
            ProductionCandidatePrediction(
                observable="accessible hotspot fraction",
                baseline_expectation="tracks bulk loading",
                alternative_expectation="can diverge from bulk loading",
                discriminating_outcome="local accessibility explains residual signal variance",
            )
        ],
        falsifiers=[
            ProductionCandidateFalsifier(
                falsifying_outcome="bulk loading fully predicts local accessibility and signal variance"
            )
        ],
    )


def _portfolio(*candidates: ProductionScientificCandidate) -> ProductionFacingScientificCandidatePortfolio:
    rows = list(candidates) or [_rel(), _ref()]
    rel = sum(row.source_lane == "RELATIONAL_DISCOVERY" for row in rows)
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row.source_object_kind] = kinds.get(row.source_object_kind, 0) + 1
    return ProductionFacingScientificCandidatePortfolio(
        portfolio_id="production_facing:1",
        source_cross_lane_portfolio_id="cross:1",
        source_cross_lane_portfolio_file_sha256="a" * 64,
        source_task_id="task:1",
        source_context_id="ctx:1",
        source_context_sha256="c" * 64,
        question="What controls matrix-dependent SERS reliability?",
        domain_profile_id="sers",
        candidates=rows,
        lineage_validation=ProductionCandidateLineageValidation(),
        candidate_count=len(rows),
        relational_candidate_count=rel,
        reframing_candidate_count=len(rows) - rel,
        source_kind_counts=kinds,
    )


def _draft(**overrides) -> CrossLaneSynthesizedHypothesisDraft:
    kwargs = dict(
        local_id="s1",
        title="Matrix loading acts through accessible hotspot fraction",
        source_candidate_ids=["pc:rel", "pc:ref"],
        synthesis_kind="complementary_mechanism_integration",
        hypothesis_statement=(
            "Matrix loading affects SERS reliability conditionally through the fraction of locally accessible hotspots, "
            "so matched bulk loading can yield different signal variance when hotspot accessibility differs."
        ),
        synthesis_rationale=(
            "The relational loading-stability effect becomes mechanistically conditional when combined with the latent accessibility construct."
        ),
        premise_statement_ids=["p1", "p2", "p3"],
        gap_statement_ids=["g1"],
        assumptions=["matrix composition is otherwise comparable"],
        predictions=[
            SynthesisPredictionDraft(
                observable="signal variance at matched bulk loading",
                expected_result="variance follows accessible hotspot fraction rather than bulk loading alone",
                rationale="the latent construct should explain residual variation",
                source_candidate_ids=["pc:rel", "pc:ref"],
            )
        ],
        falsifiers=[
            SynthesisFalsifierDraft(
                observable="signal variance",
                falsifying_outcome="bulk loading predicts variance after local accessibility is measured",
                source_candidate_ids=["pc:rel", "pc:ref"],
            )
        ],
        discriminating_test=SynthesisDiscriminatingTestDraft(
            test_design="hold bulk loading comparable while varying or measuring local hotspot accessibility",
            primary_observables=["bulk loading", "accessible hotspot fraction", "signal variance"],
            source_candidate_ids=["pc:rel", "pc:ref"],
            favoring_outcomes=[
                "bulk-loading model favored if accessibility adds no explanatory value",
                "synthesized model favored if accessibility explains residual variance",
            ],
        ),
        unresolved_questions=["which measurement best estimates accessible hotspot fraction?"],
    )
    kwargs.update(overrides)
    return CrossLaneSynthesizedHypothesisDraft(**kwargs)


class FakeBackend:
    backend_name = "fake"
    model_name = "fake-model"

    def __init__(self, draft: CrossLaneSynthesisBatchDraft):
        self.draft = draft
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return CrossLaneSynthesisGeneration(
            draft=self.draft,
            input_tokens=100,
            output_tokens=50,
        )


def test_prompt_exposes_short_candidate_aliases_without_canonical_ids():
    prompt = build_cross_lane_synthesis_prompt(_portfolio(), max_syntheses=2)
    assert "CANDIDATE_01" in prompt.user_prompt
    assert "CANDIDATE_02" in prompt.user_prompt
    assert "pc:rel" not in prompt.user_prompt
    assert "pc:ref" not in prompt.user_prompt
    assert "candidate_ref aliases" in prompt.system_prompt
    assert "NOT to rank" in prompt.system_prompt
    assert "BOTH source lanes" in prompt.system_prompt


def test_alias_refs_compile_to_canonical_candidate_ids():
    base = _draft()
    prediction = base.predictions[0].model_copy(
        update={"source_candidate_ids": ["CANDIDATE_01", "CANDIDATE_02"]}
    )
    falsifier = base.falsifiers[0].model_copy(
        update={"source_candidate_ids": ["CANDIDATE_01", "CANDIDATE_02"]}
    )
    test = base.discriminating_test.model_copy(
        update={"source_candidate_ids": ["CANDIDATE_01", "CANDIDATE_02"]}
    )
    draft = base.model_copy(
        update={
            "source_candidate_ids": ["CANDIDATE_01", "CANDIDATE_02"],
            "predictions": [prediction],
            "falsifiers": [falsifier],
            "discriminating_test": test,
        }
    )
    report = compile_cross_lane_synthesis(
        portfolio=_portfolio(),
        draft=CrossLaneSynthesisBatchDraft(hypotheses=[draft]),
        backend_name="fake",
        model_name="fake",
        max_syntheses=3,
        llm_calls_performed=1,
    )
    row = report.hypotheses[0]
    assert row.source_candidate_ids == ["pc:rel", "pc:ref"]
    assert row.predictions[0].source_candidate_ids == ["pc:rel", "pc:ref"]
    assert row.falsifiers[0].source_candidate_ids == ["pc:rel", "pc:ref"]
    assert row.discriminating_test.source_candidate_ids == ["pc:rel", "pc:ref"]


def test_unknown_alias_still_fails_closed():
    draft = _draft(source_candidate_ids=["CANDIDATE_01", "CANDIDATE_99"])
    with pytest.raises(ValueError, match="unknown source candidate IDs"):
        compile_cross_lane_synthesis(
            portfolio=_portfolio(),
            draft=CrossLaneSynthesisBatchDraft(hypotheses=[draft]),
            backend_name="fake",
            model_name="fake",
            max_syntheses=3,
            llm_calls_performed=1,
        )


def test_runtime_compiles_genuine_cross_lane_synthesis():
    backend = FakeBackend(CrossLaneSynthesisBatchDraft(hypotheses=[_draft()]))
    report, prompt = CrossLaneSynthesisRuntime(backend).run(portfolio=_portfolio())
    assert prompt is not None
    assert backend.calls == 1
    assert report.llm_calls_performed == 1
    assert report.synthesized_hypothesis_count == 1
    row = report.hypotheses[0]
    assert set(row.source_lanes) == {"RELATIONAL_DISCOVERY", "SCIENTIFIC_REFRAMING"}
    assert row.new_evidence_asserted is False
    assert report.source_candidates_preserved is True
    assert report.production_selection_changed is False


def test_same_lane_synthesis_is_rejected():
    portfolio = _portfolio(_rel("pc:r1"), _rel("pc:r2"), _ref())
    draft = _draft(source_candidate_ids=["pc:r1", "pc:r2"])
    with pytest.raises(ValueError, match="each source lane"):
        compile_cross_lane_synthesis(
            portfolio=portfolio,
            draft=CrossLaneSynthesisBatchDraft(hypotheses=[draft]),
            backend_name="fake",
            model_name="fake",
            max_syntheses=3,
            llm_calls_performed=1,
        )


def test_unknown_source_candidate_is_rejected():
    draft = _draft(source_candidate_ids=["pc:rel", "pc:missing"])
    with pytest.raises(ValueError, match="unknown source candidate IDs"):
        compile_cross_lane_synthesis(
            portfolio=_portfolio(),
            draft=CrossLaneSynthesisBatchDraft(hypotheses=[draft]),
            backend_name="fake",
            model_name="fake",
            max_syntheses=3,
            llm_calls_performed=1,
        )


def test_unsupported_statement_id_is_rejected():
    draft = _draft(premise_statement_ids=["p1", "p999"])
    with pytest.raises(ValueError, match="unsupported premise"):
        compile_cross_lane_synthesis(
            portfolio=_portfolio(),
            draft=CrossLaneSynthesisBatchDraft(hypotheses=[draft]),
            backend_name="fake",
            model_name="fake",
            max_syntheses=3,
            llm_calls_performed=1,
        )


def test_prediction_cannot_cite_unselected_candidate():
    portfolio = _portfolio(_rel(), _ref(), _ref("pc:ref2"))
    prediction = _draft().predictions[0].model_copy(
        update={"source_candidate_ids": ["pc:rel", "pc:ref2"]}
    )
    draft = _draft(predictions=[prediction])
    with pytest.raises(ValueError, match=r"prediction\[0\].*unknown"):
        compile_cross_lane_synthesis(
            portfolio=portfolio,
            draft=CrossLaneSynthesisBatchDraft(hypotheses=[draft]),
            backend_name="fake",
            model_name="fake",
            max_syntheses=3,
            llm_calls_performed=1,
        )


def test_source_equivalent_rewrite_is_rejected():
    draft = _draft(
        hypothesis_statement="Accessible hotspot fraction mediates the observed matrix-loading response."
    )
    with pytest.raises(ValueError, match="must differ"):
        compile_cross_lane_synthesis(
            portfolio=_portfolio(),
            draft=CrossLaneSynthesisBatchDraft(hypotheses=[draft]),
            backend_name="fake",
            model_name="fake",
            max_syntheses=3,
            llm_calls_performed=1,
        )


def test_abstention_preserves_sources_without_synthesis():
    draft = CrossLaneSynthesisBatchDraft(
        hypotheses=[], abstention_reason="No nontrivial cross-lane integration is justified."
    )
    report = compile_cross_lane_synthesis(
        portfolio=_portfolio(),
        draft=draft,
        backend_name="fake",
        model_name="fake",
        max_syntheses=3,
        llm_calls_performed=1,
    )
    assert report.synthesized_hypothesis_count == 0
    assert report.cross_lane_synthesis_performed is False
    assert report.source_candidate_count == 2


def test_missing_lane_abstains_without_llm_call():
    portfolio = _portfolio(_rel())
    backend = FakeBackend(CrossLaneSynthesisBatchDraft(hypotheses=[_draft()]))
    report, prompt = CrossLaneSynthesisRuntime(backend).run(portfolio=portfolio)
    assert prompt is None
    assert backend.calls == 0
    assert report.llm_calls_performed == 0
    assert report.abstention_reason is not None


def test_max_syntheses_is_fail_closed():
    draft1 = _draft()
    draft2 = _draft(
        local_id="s2",
        source_candidate_ids=["pc:rel", "pc:ref"],
        synthesis_kind="conditional_relation_refinement",
        title="second",
        hypothesis_statement="A second distinct synthesized hypothesis statement.",
    )
    with pytest.raises(ValueError, match="max_syntheses"):
        compile_cross_lane_synthesis(
            portfolio=_portfolio(),
            draft=CrossLaneSynthesisBatchDraft(hypotheses=[draft1, draft2]),
            backend_name="fake",
            model_name="fake",
            max_syntheses=1,
            llm_calls_performed=1,
        )


def test_compilation_is_deterministic():
    kwargs = dict(
        portfolio=_portfolio(),
        draft=CrossLaneSynthesisBatchDraft(hypotheses=[_draft()]),
        backend_name="fake",
        model_name="fake",
        max_syntheses=3,
        llm_calls_performed=1,
    )
    first = compile_cross_lane_synthesis(**kwargs)
    second = compile_cross_lane_synthesis(**kwargs)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
