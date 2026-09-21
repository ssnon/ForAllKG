from __future__ import annotations

from pipeline_core.discovery.reframing.mode_contrast import (
    PairwiseReasoningModeContrast,
    ReasoningModeCandidateProfile,
    ReasoningModeContrastReport,
)
from pipeline_core.discovery.reframing.reasoning_portfolio import (
    build_unified_reasoning_portfolio,
)


def _profile(
    candidate_id: str,
    operator_id: str,
    transform: str,
    premise_ids: list[str] | None = None,
) -> ReasoningModeCandidateProfile:
    return ReasoningModeCandidateProfile(
        candidate_id=candidate_id,
        operator_id=operator_id,
        representation_transform=transform,
        premise_statement_ids=premise_ids or ["p1", "p2"],
        gap_statement_ids=["g1"],
        challenge_text=f"challenge {operator_id}",
        baseline_summary=f"baseline {operator_id}",
        alternative_summary=f"alternative {operator_id}",
        construct_texts=[f"construct {operator_id}"],
        prediction_observables=["spectral response"],
        test_observables=["extinction", "SERS"],
    )


def _pair(
    left: ReasoningModeCandidateProfile,
    right: ReasoningModeCandidateProfile,
    relation: str = "distinct_mode_shared_scientific_neighborhood",
) -> PairwiseReasoningModeContrast:
    return PairwiseReasoningModeContrast(
        candidate_a_id=left.candidate_id,
        operator_a=left.operator_id,
        transform_a=left.representation_transform,
        candidate_b_id=right.candidate_id,
        operator_b=right.operator_id,
        transform_b=right.representation_transform,
        representation_transform_distinct=True,
        shared_premise_count=2,
        premise_jaccard=0.5,
        gap_jaccard=1.0,
        challenge_token_jaccard=0.1,
        baseline_summary_token_jaccard=0.2,
        alternative_summary_token_jaccard=0.2,
        prediction_observable_token_jaccard=0.3,
        test_observable_token_jaccard=0.4,
        construct_token_jaccard=0.1,
        shared_scientific_neighborhood_signals=(
            ["shared_grounded_premise_family", "overlapping_discriminating_observables"]
            if relation == "distinct_mode_shared_scientific_neighborhood"
            else []
        ),
        surface_near_duplicate_signals=(
            ["high_premise_overlap"]
            if relation == "distinct_operator_surface_near_duplicate"
            else []
        ),
        relation=relation,
    )


def _four_profiles():
    return [
        _profile("latent:1", "LATENT_VARIABLE", "introduces_hidden_construct"),
        _profile("regime:1", "REGIME_BOUNDARY", "partitions_response_law"),
        _profile("proxy:1", "PROXY_CHALLENGE", "challenges_measurement_equivalence"),
        _profile(
            "contradiction:1",
            "CONTRADICTION_RESOLUTION",
            "reconciles_apparently_incompatible_evidence",
        ),
    ]


def _report(profiles, pairs):
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.relation] = counts.get(pair.relation, 0) + 1
    return ReasoningModeContrastReport(
        report_id="contrast:1",
        source_task_id="task:1",
        candidate_profiles=profiles,
        pairwise_contrasts=pairs,
        relation_counts=counts,
    )


def test_four_reasoning_modes_are_preserved_as_distinct_slots():
    profiles = _four_profiles()
    pairs = [
        _pair(profiles[i], profiles[j])
        for i in range(len(profiles))
        for j in range(i + 1, len(profiles))
    ]
    portfolio = build_unified_reasoning_portfolio(_report(profiles, pairs))
    assert portfolio.candidate_count == 4
    assert portfolio.occupied_reasoning_mode_count == 4
    assert portfolio.empty_reasoning_mode_count == 0
    assert [row.slot_id for row in portfolio.slots] == [
        "LATENT_VARIABLE",
        "REGIME_BOUNDARY",
        "PROXY_CHALLENGE",
        "CONTRADICTION_RESOLUTION",
    ]
    assert all(len(row.candidate_ids) == 1 for row in portfolio.slots)


def test_k03_like_all_shared_neighborhood_forms_one_cluster_without_dropping():
    profiles = _four_profiles()
    pairs = [
        _pair(profiles[i], profiles[j])
        for i in range(len(profiles))
        for j in range(i + 1, len(profiles))
    ]
    portfolio = build_unified_reasoning_portfolio(_report(profiles, pairs))
    assert portfolio.scientific_neighborhood_cluster_count == 1
    cluster = portfolio.scientific_neighborhood_clusters[0]
    assert set(cluster.candidate_ids) == {row.candidate_id for row in profiles}
    assert len(cluster.representation_transforms) == 4
    assert portfolio.redundancy_based_candidate_drop_performed is False
    assert all(row.shadow_candidate_preserved for row in portfolio.entries)


def test_low_overlap_pair_separates_scientific_neighborhood_clusters():
    latent, regime, proxy, contradiction = _four_profiles()
    pairs = [
        _pair(latent, regime),
        _pair(proxy, contradiction),
        _pair(latent, proxy, "distinct_mode_low_observed_overlap"),
        _pair(latent, contradiction, "distinct_mode_low_observed_overlap"),
        _pair(regime, proxy, "distinct_mode_low_observed_overlap"),
        _pair(regime, contradiction, "distinct_mode_low_observed_overlap"),
    ]
    portfolio = build_unified_reasoning_portfolio(
        _report([latent, regime, proxy, contradiction], pairs)
    )
    assert portfolio.scientific_neighborhood_cluster_count == 2
    clusters = [set(row.candidate_ids) for row in portfolio.scientific_neighborhood_clusters]
    assert {"latent:1", "regime:1"} in clusters
    assert {"proxy:1", "contradiction:1"} in clusters


def test_empty_reasoning_mode_slot_is_explicit_and_not_fabricated():
    latent = _four_profiles()[0]
    portfolio = build_unified_reasoning_portfolio(_report([latent], []))
    assert portfolio.occupied_reasoning_mode_count == 1
    assert portfolio.empty_reasoning_mode_count == 3
    proxy_slot = next(row for row in portfolio.slots if row.slot_id == "PROXY_CHALLENGE")
    assert proxy_slot.candidate_ids == []
    assert "No compiled shadow candidate" in proxy_slot.empty_reason


def test_surface_near_duplicate_relation_does_not_remove_candidate():
    latent, regime = _four_profiles()[:2]
    pair = _pair(latent, regime, "distinct_operator_surface_near_duplicate")
    portfolio = build_unified_reasoning_portfolio(_report([latent, regime], [pair]))
    assert portfolio.candidate_count == 2
    assert portfolio.redundancy_based_candidate_drop_performed is False
    cluster = portfolio.scientific_neighborhood_clusters[0]
    assert cluster.pair_relations == {"distinct_operator_surface_near_duplicate": 1}
    assert "high_premise_overlap" in cluster.surface_near_duplicate_signals


def test_portfolio_has_no_ranking_selection_or_authority():
    profiles = _four_profiles()
    pairs = [_pair(profiles[0], profiles[1])]
    # Other candidates remain isolated clusters; that is valid diagnostic structure.
    portfolio = build_unified_reasoning_portfolio(_report(profiles, pairs))
    assert portfolio.llm_calls_performed == 0
    assert portfolio.scientific_quality_ranking_performed is False
    assert portfolio.overall_score_computed is False
    assert portfolio.candidate_winner_selected is False
    assert portfolio.portfolio_selection_authority is False
    assert portfolio.scientific_topic_diversity_established is False
    assert portfolio.production_selection_changed is False
    assert portfolio.canonical_graph_mutated is False
    assert portfolio.relational_lane_integrated is False


def test_portfolio_id_is_deterministic():
    profiles = _four_profiles()
    pairs = [_pair(profiles[0], profiles[1])]
    report = _report(profiles, pairs)
    first = build_unified_reasoning_portfolio(report)
    second = build_unified_reasoning_portfolio(report)
    assert first.portfolio_id == second.portfolio_id
    assert [row.cluster_id for row in first.scientific_neighborhood_clusters] == [
        row.cluster_id for row in second.scientific_neighborhood_clusters
    ]


def test_operator_transform_mismatch_is_rejected():
    bad = _profile(
        "latent:bad",
        "LATENT_VARIABLE",
        "partitions_response_law",
    )
    try:
        build_unified_reasoning_portfolio(_report([bad], []))
    except ValueError as exc:
        assert "unexpected representation transform" in str(exc)
    else:
        raise AssertionError("expected operator/transform mismatch rejection")


def test_cluster_relation_counts_are_local_to_component():
    latent, regime, proxy, contradiction = _four_profiles()
    pairs = [
        _pair(latent, regime),
        _pair(proxy, contradiction, "distinct_operator_surface_near_duplicate"),
        _pair(latent, proxy, "distinct_mode_low_observed_overlap"),
        _pair(latent, contradiction, "distinct_mode_low_observed_overlap"),
        _pair(regime, proxy, "distinct_mode_low_observed_overlap"),
        _pair(regime, contradiction, "distinct_mode_low_observed_overlap"),
    ]
    portfolio = build_unified_reasoning_portfolio(
        _report([latent, regime, proxy, contradiction], pairs)
    )
    relation_maps = [row.pair_relations for row in portfolio.scientific_neighborhood_clusters]
    assert {"distinct_mode_shared_scientific_neighborhood": 1} in relation_maps
    assert {"distinct_operator_surface_near_duplicate": 1} in relation_maps
