from __future__ import annotations

from dac_her.corpus_acquisition.candidate_selection import (
    assess_catalog,
    assess_candidate,
    select_candidates,
)
from dac_her.corpus_acquisition.contracts import (
    AcquisitionAxis,
    AcquisitionProfile,
    DiscoveryPolicy,
    ScopePolicy,
    ScorePolicy,
    SelectionPolicy,
    SignalRule,
)
from pipeline_core.literature.catalog_contracts import (
    CatalogWork,
    LiteratureCatalogPacket,
)


def _profile() -> AcquisitionProfile:
    return AcquisitionProfile(
        profile_id="test_profile",
        domain_profile_id="test_domain",
        discovery=DiscoveryPolicy(results_per_query=10),
        scope=ScopePolicy(
            required_term_groups=[
                ["sers"],
                ["gold", "silver"],
            ],
            excluded_title_terms=["review"],
            manual_review_if_no_abstract=True,
        ),
        scoring=ScorePolicy(
            open_access_bonus=2.0,
            abstract_available_bonus=1.0,
            retrieval_axis_bonus=1.0,
            axis_indicator_bonus=1.0,
            signals=[
                SignalRule(
                    signal_id="sweep",
                    terms=["systematic", "varying"],
                    weight=3.0,
                )
            ],
        ),
        selection=SelectionPolicy(target_total=3),
        axes=[
            AcquisitionAxis(
                axis_id="gap",
                target_selected=1,
                queries=["SERS nanogap"],
                indicators=["nanogap"],
            ),
            AcquisitionAxis(
                axis_id="shell",
                target_selected=1,
                queries=["SERS shell"],
                indicators=["shell thickness"],
            ),
        ],
    )


def _work(
    work_id: str,
    title: str,
    *,
    abstract: str | None,
    axis: str,
    oa: bool = False,
    publication_types=None,
) -> CatalogWork:
    return CatalogWork(
        work_id=work_id,
        title=title,
        abstract=abstract,
        open_access_url=(
            "https://example.test/paper.pdf"
            if oa
            else None
        ),
        publication_types=publication_types or [],
        retrieval_query_ids=[f"q:{axis}"],
        retrieval_axis_ids=[axis],
        providers=["test"],
    )


def _packet(works):
    return LiteratureCatalogPacket(
        catalog_id="catalog:test",
        catalog_sha256="x",
        acquisition_profile_id="test_profile",
        searched_at_utc="2026-08-14T00:00:00+00:00",
        providers_requested=["test"],
        works=works,
    )


def test_scope_excludes_review_and_never_infers_result_direction():
    profile = _profile()
    work = _work(
        "w1",
        "Review of gold SERS nanogap systems",
        abstract="systematic silver SERS nanogap overview",
        axis="gap",
    )
    row = assess_candidate(work, profile)
    assert row.eligibility_status == "excluded"
    assert any(
        reason.startswith("excluded_title_term:")
        for reason in row.exclusion_reasons
    )
    assert row.scientific_result_direction_inferred is False
    assert not hasattr(row, "result_direction")


def test_missing_abstract_routes_to_manual_review_not_positive_selection():
    profile = _profile()
    work = _work(
        "w1",
        "Gold SERS nanogap controlled experiment",
        abstract=None,
        axis="gap",
    )
    row = assess_candidate(work, profile)
    assert row.eligibility_status == "manual_review"


def test_score_prefers_oa_and_controlled_sweep_metadata():
    profile = _profile()
    plain = _work(
        "plain",
        "Gold SERS nanogap experiment",
        abstract="silver SERS nanogap",
        axis="gap",
    )
    strong = _work(
        "strong",
        "Gold SERS nanogap systematic experiment",
        abstract="silver SERS nanogap with varying gap size",
        axis="gap",
        oa=True,
    )
    assert (
        assess_candidate(strong, profile).total_score
        > assess_candidate(plain, profile).total_score
    )


def test_quota_selection_charges_each_work_to_one_primary_axis():
    profile = _profile()
    works = [
        _work(
            "gap",
            "Gold SERS nanogap systematic study",
            abstract="silver SERS nanogap varying gap size",
            axis="gap",
            oa=True,
        ),
        _work(
            "shell",
            "Gold SERS shell thickness systematic study",
            abstract="silver SERS shell thickness varying shell",
            axis="shell",
            oa=True,
        ),
        CatalogWork(
            work_id="both",
            title="Gold SERS nanogap and shell thickness experiment",
            abstract="silver SERS nanogap shell thickness",
            retrieval_query_ids=["q:gap", "q:shell"],
            retrieval_axis_ids=["gap", "shell"],
            providers=["test"],
        ),
    ]
    packet = _packet(works)
    assessments = assess_catalog(packet, profile)
    selected, report = select_candidates(
        packet=packet,
        profile=profile,
        assessments=assessments,
    )
    assert len(selected) == 3
    assert report.axis_primary_selected_counts["gap"] == 1
    assert report.axis_primary_selected_counts["shell"] == 1
    charged = [
        row.primary_quota_axis
        for row in selected
        if row.primary_quota_axis is not None
    ]
    assert len(charged) == len(set(charged))
    assert report.positive_evidence_promotion_performed is False


def test_retrieval_axis_does_not_satisfy_required_axis_match_without_indicator():
    profile = _profile()
    work = _work(
        "retrieval-only",
        "Gold SERS substrate performance study",
        abstract="silver SERS substrate performance and reproducibility",
        axis="gap",
    )

    row = assess_candidate(work, profile)

    assert row.eligibility_status == "excluded"
    assert row.matched_axes == []
    assert row.matched_terms_by_axis == {}
    assert "no_acquisition_axis_match" in row.exclusion_reasons
    # Retrieval provenance is still allowed to affect ranking metadata.
    assert row.score_components["retrieval_axis"] == 1.0
    assert "axis_indicators" not in row.score_components


def test_retrieval_axis_does_not_claim_quota_when_axis_match_is_optional():
    profile = _profile().model_copy(
        update={
            "scope": _profile().scope.model_copy(
                update={"require_axis_match": False}
            ),
            "selection": SelectionPolicy(target_total=1),
            "axes": [
                AcquisitionAxis(
                    axis_id="gap",
                    target_selected=1,
                    queries=["SERS nanogap"],
                    indicators=["nanogap"],
                )
            ],
        }
    )
    work = _work(
        "retrieval-only",
        "Gold SERS substrate performance study",
        abstract="silver SERS substrate performance and reproducibility",
        axis="gap",
    )
    packet = _packet([work])

    assessments = assess_catalog(packet, profile)
    selected, report = select_candidates(
        packet=packet,
        profile=profile,
        assessments=assessments,
    )

    assert assessments[0].eligibility_status == "eligible"
    assert assessments[0].matched_axes == []
    assert len(selected) == 1
    assert selected[0].matched_axes == []
    assert selected[0].primary_quota_axis is None
    assert report.axis_candidate_counts["gap"] == 0
    assert report.axis_primary_selected_counts["gap"] == 0
    assert report.unfilled_axis_quotas == {"gap": 1}


def test_indicator_match_counts_as_evidence_axis_even_from_other_retrieval_axis():
    profile = _profile()
    work = _work(
        "cross-axis",
        "Gold SERS nanogap experiment",
        abstract="silver SERS nanogap response",
        axis="shell",
    )

    row = assess_candidate(work, profile)

    assert row.eligibility_status == "eligible"
    assert row.matched_axes == ["gap"]
    assert row.matched_terms_by_axis == {"gap": ["nanogap"]}
    assert row.score_components["retrieval_axis"] == 1.0
    assert row.score_components["axis_indicators"] == 1.0
