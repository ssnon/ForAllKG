from __future__ import annotations

from pipeline_core.discovery.reframing.contradiction_resolution import (
    ContradictionResolutionRunReport,
    ScientificContradictionResolutionCandidate,
)
from pipeline_core.discovery.reframing.mode_contrast import (
    build_reasoning_mode_contrast,
)
from pipeline_core.discovery.reframing.proxy_challenge import (
    ProxyChallengeRunReport,
    ScientificProxyChallengeCandidate,
)
from pipeline_core.discovery.reframing.reframe_contracts import (
    DifferentialPredictionDraft,
    DiscriminatingTestDraft,
    ReframeFalsifierDraft,
    ReframeOperatorRunRecord,
    ScientificModelDraft,
    ScientificReframeCandidate,
    ScientificReframingShadowReport,
)


def _model(summary: str, ids: list[str]) -> ScientificModelDraft:
    return ScientificModelDraft(
        summary=summary,
        assumptions=["controlled comparison"],
        explained_statement_ids=ids,
        expected_observations=["spectral response changes"],
    )


def _prediction(observable: str) -> DifferentialPredictionDraft:
    return DifferentialPredictionDraft(
        local_id="dp1",
        observable=observable,
        baseline_expectation="far-field LSPR tracks SERS optimum",
        alternative_expectation="near-field-sensitive response can diverge from LSPR",
        discriminating_outcome="persistent offset distinguishes the models",
    )


def _test() -> DiscriminatingTestDraft:
    return DiscriminatingTestDraft(
        test_design="measure extinction, near field, and SERS excitation spectra",
        primary_observables=["far-field LSPR", "local near-field spectrum", "SERS optimum"],
        baseline_favoring_outcome="all maxima track together",
        alternative_favoring_outcome="near-field and SERS diverge from extinction",
    )


def _latent(task: str = "task:1") -> ScientificReframeCandidate:
    return ScientificReframeCandidate(
        reframe_id="latent:1",
        operator_id="LATENT_VARIABLE",
        source_task_id=task,
        source_context_id="ctx",
        source_context_sha256="sha",
        title="local spectral accessibility",
        premise_statement_ids=["p1", "p2", "p3"],
        gap_statement_ids=["g1"],
        baseline_model=_model("far-field LSPR predicts the SERS optimum", ["p1"]),
        alternative_model=_model(
            "latent local spectral accessibility governs the SERS optimum", ["p1", "p2"]
        ),
        challenged_assumption="far-field LSPR is a sufficient proxy for useful local near field",
        proposed_constructs=["environment-conditioned local spectral accessibility"],
        latent_constructs=["environment-conditioned local spectral accessibility"],
        boundary_variables=[],
        regime_change_kind=None,
        differential_predictions=[_prediction("relative shift of LSPR and SERS optimum")],
        falsifiers=[ReframeFalsifierDraft(local_id="f1", falsifying_outcome="LSPR predicts all cases")],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _proxy(task: str = "task:1", *, premises: list[str] | None = None) -> ScientificProxyChallengeCandidate:
    selected = premises or ["p1", "p2", "p4"]
    return ScientificProxyChallengeCandidate(
        candidate_id="proxy:1",
        source_task_id=task,
        source_context_id="ctx",
        source_context_sha256="sha",
        source_enrichment_report_id="enrich:1",
        source_annotation_sha256="a" * 64,
        title="LSPR may be an incomplete proxy",
        semantic_seed_annotation_ids=["ann:1"],
        premise_statement_ids=selected,
        gap_statement_ids=["g1"],
        baseline_model=_model("far-field LSPR predicts the SERS optimum", [selected[0]]),
        alternative_model=_model(
            "far-field LSPR may not represent the local response sampled by SERS", selected[:2]
        ),
        challenged_proxy_assumption="far-field LSPR is an interchangeable proxy for useful local near field",
        challenged_observable="far-field LSPR wavelength",
        target_construct="spectrally useful local near field",
        proxy_failure_mode="surrogate_breakdown",
        differential_predictions=[_prediction("relative shift of LSPR and SERS optimum")],
        falsifiers=[ReframeFalsifierDraft(local_id="f1", falsifying_outcome="LSPR predicts all cases")],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _reframe_report(candidate: ScientificReframeCandidate, task: str = "task:1") -> ScientificReframingShadowReport:
    return ScientificReframingShadowReport(
        report_id="shadow:1",
        source_task_id=task,
        source_context_id="ctx",
        source_context_sha256="sha",
        backend_name="fake",
        model_name="fake",
        runs=[ReframeOperatorRunRecord(operator_id=candidate.operator_id, readiness_status="ready_now", decision="generated", candidate_ids=[candidate.reframe_id])],
        candidates=[candidate],
        llm_calls_performed=1,
    )


def _proxy_report(candidate: ScientificProxyChallengeCandidate, task: str = "task:1") -> ProxyChallengeRunReport:
    return ProxyChallengeRunReport(
        report_id="proxy-shadow:1",
        source_task_id=task,
        source_context_id="ctx",
        source_context_sha256="sha",
        source_enrichment_report_id="enrich:1",
        source_annotation_sha256="a" * 64,
        backend_name="fake",
        model_name="fake",
        decision="generated",
        candidate_ids=[candidate.candidate_id],
        candidates=[candidate],
        task_relevant_seed_ids=["ann:1"],
        llm_calls_performed=1,
    )


def _contradiction(
    task: str = "task:1",
) -> ScientificContradictionResolutionCandidate:
    return ScientificContradictionResolutionCandidate(
        candidate_id="contradiction:1",
        source_task_id=task,
        source_context_id="ctx",
        source_context_sha256="sha",
        source_tension_report_id="tension-report:1",
        source_tension_report_sha256="b" * 64,
        title="matching and offset observations can coexist",
        tension_witness_ids=["w1"],
        premise_statement_ids=["p1", "p2", "p5"],
        gap_statement_ids=["g1"],
        side_a_statement_ids=["p1", "p2"],
        side_b_statement_ids=["p5"],
        baseline_model=_model("one global LSPR matching rule explains SERS", ["p1", "p2"]),
        resolution_model=_model(
            "matching helps when extinction and near-field maxima align, while offsets arise when their balance changes",
            ["p1", "p2", "p5"],
        ),
        apparent_contradiction=(
            "some observations associate stronger SERS with LSPR matching while another shows an offset maximum"
        ),
        resolution_principle=(
            "LSPR proximity is conditionally beneficial when extinction and near-field enhancement are spectrally aligned"
        ),
        resolution_kind="competing_mechanism_balance",
        distinguishing_context_variables=[
            "extinction-near-field spectral separation",
            "nanostructure coupling",
        ],
        proposed_resolution_constructs=[
            "wavelength-dependent extinction-near-field balance"
        ],
        differential_predictions=[
            _prediction("offset between far-field LSPR and SERS maximum")
        ],
        falsifiers=[
            ReframeFalsifierDraft(
                local_id="f1",
                falsifying_outcome="all SERS maxima track LSPR despite changing extinction and coupling",
            )
        ],
        discriminating_test=_test(),
        unresolved_questions=[],
    )


def _contradiction_report(
    candidate: ScientificContradictionResolutionCandidate,
    task: str = "task:1",
) -> ContradictionResolutionRunReport:
    return ContradictionResolutionRunReport(
        report_id="contradiction-shadow:1",
        source_task_id=task,
        source_context_id="ctx",
        source_context_sha256="sha",
        source_tension_report_id="tension-report:1",
        source_tension_report_sha256="b" * 64,
        backend_name="fake",
        model_name="fake",
        decision="generated",
        candidate_ids=[candidate.candidate_id],
        candidates=[candidate],
        eligible_witness_ids=["w1"],
        llm_calls_performed=1,
    )


def test_latent_and_proxy_have_distinct_representation_transforms():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(_proxy()),
    )
    pair = report.pairwise_contrasts[0]
    assert pair.representation_transform_distinct is True
    assert pair.transform_a == "introduces_hidden_construct"
    assert pair.transform_b == "challenges_measurement_equivalence"


def test_k03_like_pair_is_shared_scientific_neighborhood():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(_proxy()),
    )
    pair = report.pairwise_contrasts[0]
    assert pair.relation == "distinct_mode_shared_scientific_neighborhood"
    assert "shared_grounded_premise_family" in pair.shared_scientific_neighborhood_signals
    assert "overlapping_prediction_observables" in pair.shared_scientific_neighborhood_signals


def test_low_overlap_pair_remains_distinct_mode_low_overlap():
    proxy = _proxy(premises=["q1", "q2"])
    proxy = proxy.model_copy(update={
        "challenged_proxy_assumption": "cell viability is an interchangeable proxy for long-term organism safety",
        "challenged_observable": "cell viability",
        "target_construct": "long-term organism safety",
        "differential_predictions": [_prediction("long-term survival after chronic exposure")],
        "discriminating_test": DiscriminatingTestDraft(
            test_design="compare acute viability with chronic outcomes",
            primary_observables=["acute viability", "chronic survival"],
            baseline_favoring_outcome="acute and chronic outcomes track",
            alternative_favoring_outcome="they diverge",
        ),
        "baseline_model": _model("acute viability predicts long-term safety", ["q1"]),
        "alternative_model": _model("acute viability can diverge from long-term safety", ["q1", "q2"]),
    })
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(proxy),
    )
    assert report.pairwise_contrasts[0].relation == "distinct_mode_low_observed_overlap"


def test_mismatched_task_is_rejected():
    try:
        build_reasoning_mode_contrast(
            reframe_shadow=_reframe_report(_latent("task:1"), "task:1"),
            proxy_shadow=_proxy_report(_proxy("task:2"), "task:2"),
        )
    except ValueError as exc:
        assert "shared source task" in str(exc)
    else:
        raise AssertionError("expected task mismatch rejection")


def test_report_has_no_ranking_or_selection_authority():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(_proxy()),
    )
    assert report.llm_calls_performed == 0
    assert report.scientific_quality_ranking_performed is False
    assert report.candidate_winner_selected is False
    assert report.portfolio_selection_changed is False
    assert report.production_selection_changed is False
    assert report.canonical_graph_mutated is False


def test_relation_counts_are_derived_from_pairs():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(_proxy()),
    )
    assert report.relation_counts == {"distinct_mode_shared_scientific_neighborhood": 1}


def test_empty_proxy_candidates_produce_no_pairs():
    proxy = ProxyChallengeRunReport(
        report_id="proxy-shadow:empty",
        source_task_id="task:1",
        source_context_id="ctx",
        source_context_sha256="sha",
        source_enrichment_report_id="enrich:1",
        source_annotation_sha256="a" * 64,
        backend_name="fake",
        model_name="fake",
        decision="abstained",
        candidate_ids=[],
        candidates=[],
        task_relevant_seed_ids=["ann:1"],
        abstention_reason="no valid challenge",
        llm_calls_performed=1,
    )
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=proxy,
    )
    assert report.pairwise_contrasts == []
    assert report.relation_counts == {}


def test_pair_reports_premise_overlap_without_converting_it_to_rank():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(_proxy()),
    )
    pair = report.pairwise_contrasts[0]
    assert pair.shared_premise_count == 2
    assert 0.0 < pair.premise_jaccard < 1.0
    assert pair.scientific_quality_ranking_performed is False
    assert pair.portfolio_selection_authority is False


def test_contradiction_has_distinct_representation_transform():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(_proxy()),
        contradiction_shadow=_contradiction_report(_contradiction()),
    )
    contradiction_profile = next(
        row
        for row in report.candidate_profiles
        if row.operator_id == "CONTRADICTION_RESOLUTION"
    )
    assert (
        contradiction_profile.representation_transform
        == "reconciles_apparently_incompatible_evidence"
    )
    assert "offset maximum" in contradiction_profile.challenge_text
    assert "extinction-near-field" in " ".join(contradiction_profile.construct_texts)


def test_contradiction_is_compared_pairwise_without_ranking():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(_proxy()),
        contradiction_shadow=_contradiction_report(_contradiction()),
    )
    assert len(report.candidate_profiles) == 3
    assert len(report.pairwise_contrasts) == 3
    pairs = {
        frozenset((row.operator_a, row.operator_b)): row
        for row in report.pairwise_contrasts
    }
    contradiction_pair = pairs[
        frozenset(("LATENT_VARIABLE", "CONTRADICTION_RESOLUTION"))
    ]
    assert contradiction_pair.representation_transform_distinct is True
    assert contradiction_pair.scientific_quality_ranking_performed is False
    assert contradiction_pair.portfolio_selection_authority is False


def test_contradiction_task_mismatch_is_rejected():
    try:
        build_reasoning_mode_contrast(
            reframe_shadow=_reframe_report(_latent("task:1"), "task:1"),
            proxy_shadow=_proxy_report(_proxy("task:1"), "task:1"),
            contradiction_shadow=_contradiction_report(
                _contradiction("task:2"), "task:2"
            ),
        )
    except ValueError as exc:
        assert "shared source task" in str(exc)
    else:
        raise AssertionError("expected contradiction task mismatch rejection")


def test_existing_two_input_mode_contrast_remains_backward_compatible():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=_proxy_report(_proxy()),
    )
    assert len(report.candidate_profiles) == 2
    assert len(report.pairwise_contrasts) == 1
    assert all(
        row.operator_id != "CONTRADICTION_RESOLUTION"
        for row in report.candidate_profiles
    )


def test_mode_contrast_allows_proxy_and_contradiction_to_be_absent():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=None,
        contradiction_shadow=None,
    )
    assert [row.operator_id for row in report.candidate_profiles] == [
        "LATENT_VARIABLE"
    ]
    assert report.pairwise_contrasts == []
    assert report.relation_counts == {}


def test_mode_contrast_accepts_contradiction_without_proxy():
    report = build_reasoning_mode_contrast(
        reframe_shadow=_reframe_report(_latent()),
        proxy_shadow=None,
        contradiction_shadow=_contradiction_report(_contradiction()),
    )
    assert {row.operator_id for row in report.candidate_profiles} == {
        "LATENT_VARIABLE",
        "CONTRADICTION_RESOLUTION",
    }
    assert len(report.pairwise_contrasts) == 1
