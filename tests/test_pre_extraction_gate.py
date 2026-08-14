from __future__ import annotations

from dac_her.corpus_acquisition.contracts import (
    AcquisitionAxis,
    AcquisitionProfile,
    ScopePolicy,
    ScorePolicy,
    SelectionPolicy,
    SelectedCorpusWork,
)
from dac_her.corpus_acquisition.pre_extraction_gate import (
    IdentityGatePolicy,
    PreExtractionGatePolicy,
    SuitabilityGatePolicy,
    assess_bibliographic_identity,
    assess_fulltext_suitability,
    assess_pre_extraction_gate,
    build_pre_extraction_gate_report,
)
from dac_her.literature_catalog_contracts import CatalogWork


def _profile() -> AcquisitionProfile:
    return AcquisitionProfile(
        profile_id="sers_test",
        domain_profile_id="sers_au_ag",
        scope=ScopePolicy(require_axis_match=True),
        scoring=ScorePolicy(),
        selection=SelectionPolicy(target_total=2),
        axes=[
            AcquisitionAxis(
                axis_id="nanogap",
                target_selected=1,
                queries=["SERS nanogap"],
                indicators=["nanogap", "gap size", "hotspot"],
            ),
            AcquisitionAxis(
                axis_id="shell_thickness",
                target_selected=1,
                queries=["SERS shell thickness"],
                indicators=["shell thickness", "core shell"],
            ),
        ],
    )


def _policy() -> PreExtractionGatePolicy:
    return PreExtractionGatePolicy(
        policy_id="gate_test",
        identity=IdentityGatePolicy(
            front_matter_chars=5000,
            doi_scan_chars=5000,
            verified_title_token_f1=0.75,
            weak_title_token_f1=0.5,
            min_title_token_count=3,
        ),
        suitability=SuitabilityGatePolicy(
            min_main_markdown_chars=20,
            min_suitable_axes=1,
            min_axis_indicator_hits_per_axis=1,
            min_relation_context_blocks_per_axis=1,
            relation_signal_terms=[
                "dependence",
                "enhancement",
                "measured",
                "effect of",
            ],
        ),
    )


def _work() -> CatalogWork:
    return CatalogWork(
        work_id="w1",
        title="Nanogap Dependence of Gold Silver SERS Enhancement",
        doi="10.1234/example.1",
    )


def _selected(axis="nanogap") -> SelectedCorpusWork:
    return SelectedCorpusWork(
        work_id="w1",
        title=_work().title,
        doi=_work().doi,
        matched_axes=[axis],
        primary_quota_axis=axis,
        total_score=1.0,
    )


def test_exact_doi_and_local_axis_relation_context_allows_auto_extraction():
    text = """# Nanogap Dependence of Gold Silver SERS Enhancement

DOI: 10.1234/example.1

The measured nanogap dependence controls the SERS enhancement.
"""
    assessment = assess_pre_extraction_gate(
        paper_id="SERS_API_x",
        work=_work(),
        selected_work=_selected(),
        acquisition_profile=_profile(),
        main_markdown=text,
        policy=_policy(),
    )
    assert assessment.identity.status == "verified"
    assert assessment.identity.method == "doi_exact"
    assert assessment.suitability.status == "suitable"
    assert assessment.suitability.suitable_axes == ["nanogap"]
    assert assessment.auto_extraction_allowed is True
    assert assessment.scientific_result_direction_inferred is False


def test_wrong_doi_and_dissimilar_title_is_fail_closed_mismatch():
    text = """# Unrelated Raman calibration paper

DOI: 10.9999/wrong.paper

This document discusses calibration standards only.
"""
    identity = assess_bibliographic_identity(
        work=_work(),
        main_markdown=text,
        policy=_policy().identity,
    )
    assert identity.status == "mismatch"
    assert identity.method == "doi_conflict"
    assert "10.9999/wrong.paper" in identity.observed_dois


def test_missing_doi_can_be_verified_by_strong_front_matter_title():
    text = """# Nanogap Dependence of Gold Silver SERS Enhancement

Authors and affiliations

The measured nanogap dependence changes enhancement.
"""
    identity = assess_bibliographic_identity(
        work=_work(),
        main_markdown=text,
        policy=_policy().identity,
    )
    assert identity.status == "verified"
    assert identity.method == "title_verified"


def test_axis_mention_without_local_relation_context_requires_manual_review():
    text = """# Nanogap Dependence of Gold Silver SERS Enhancement

The substrate contains a nanogap architecture.

A separate section reports enhancement values without discussing geometry.
"""
    suitability = assess_fulltext_suitability(
        selected_work=_selected(),
        acquisition_profile=_profile(),
        main_markdown=text,
        policy=_policy().suitability,
    )
    assert suitability.status == "manual_review"
    assert suitability.axis_indicator_hits_by_axis["nanogap"] == ["nanogap"]
    assert suitability.relation_context_blocks_by_axis["nanogap"] == 0


def test_selected_axis_absent_from_fulltext_is_unsuitable():
    text = """# Nanogap Dependence of Gold Silver SERS Enhancement

The paper reports Raman calibration and sample preparation in detail.
"""
    suitability = assess_fulltext_suitability(
        selected_work=_selected(),
        acquisition_profile=_profile(),
        main_markdown=text,
        policy=_policy().suitability,
    )
    assert suitability.status == "unsuitable"
    assert suitability.suitable_axes == []


def test_gate_report_tracks_blocked_papers_without_promoting_evidence():
    allowed = assess_pre_extraction_gate(
        paper_id="p1",
        work=_work(),
        selected_work=_selected(),
        acquisition_profile=_profile(),
        main_markdown=(
            "# Nanogap Dependence of Gold Silver SERS Enhancement\n\n"
            "DOI: 10.1234/example.1\n\n"
            "Measured nanogap dependence produces enhancement."
        ),
        policy=_policy(),
    )
    blocked = allowed.model_copy(
        update={
            "paper_id": "p2",
            "auto_extraction_allowed": False,
            "identity": allowed.identity.model_copy(
                update={"status": "weak_match", "method": "title_weak"}
            ),
        }
    )
    report = build_pre_extraction_gate_report(
        assessments=[allowed, blocked],
        policy=_policy(),
        acquisition_profile=_profile(),
    )
    assert report.evaluated_paper_count == 2
    assert report.auto_extraction_ready_count == 1
    assert report.blocked_paper_ids == ["p2"]
    assert report.llm_calls_performed is False
    assert report.positive_evidence_promotion_performed is False


def test_repository_sers_gate_policy_loads():
    from pathlib import Path
    from dac_her.corpus_acquisition.pre_extraction_gate import (
        load_pre_extraction_gate_policy,
    )

    path = Path("configs/acquisition/sers_au_ag_pre_extraction_gate_v1.yaml")
    policy = load_pre_extraction_gate_policy(path)
    assert policy.policy_id == "sers_au_ag_strict_bridge_pre_extraction_v1"
    assert policy.allowed_identity_statuses == ["verified"]
    assert policy.allowed_suitability_statuses == ["suitable"]
