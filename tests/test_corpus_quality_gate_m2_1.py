from __future__ import annotations

from dac_her.corpus_acquisition.contracts import (
    AcquisitionAxis,
    AcquisitionProfile,
    CandidateAssessment,
    SelectionPolicy,
)
from dac_her.corpus_acquisition.corpus_quality import (
    assess_corpus_quality,
)
from dac_her.corpus_acquisition.quality_contracts import (
    CorpusQualityPolicy,
)
from pipeline_core.literature.catalog_contracts import CatalogWork


def _upstream():
    return CandidateAssessment(
        work_id="w",
        title="x",
        eligibility_status="eligible",
    )


def _policy():
    return CorpusQualityPolicy(
        policy_id="q",
        hard_exclude_title_patterns=[
            r"^\s*retracted\b",
            r"\breview\b",
        ],
        manual_review_title_patterns=[
            r"\brecent advances\b",
        ],
        manual_review_doi_prefixes=[
            "10.2139/ssrn.",
        ],
        primary_topic_terms=["SERS"],
        title_context_terms=[
            "gold",
            "silver",
            "nanowire",
            "substrate",
        ],
        min_title_context_matches_without_primary_topic=2,
    )


def test_retracted_is_hard_excluded():
    work = CatalogWork(
        work_id="w",
        title="Retracted: Zeptomolar detection by SERS using silver",
        abstract="SERS",
    )
    row = assess_corpus_quality(
        work=work,
        upstream=_upstream(),
        policy=_policy(),
        originally_selected=True,
    )
    assert row.status == "exclude"
    assert any(
        reason.startswith("hard_exclude_title_pattern:")
        for reason in row.reasons
    )


def test_recent_advances_is_manual_review():
    work = CatalogWork(
        work_id="w",
        title="Recent advances of Au Ag SERS biosensors",
        abstract="SERS gold silver",
    )
    row = assess_corpus_quality(
        work=work,
        upstream=_upstream(),
        policy=_policy(),
        originally_selected=True,
    )
    assert row.status == "manual_review"


def test_weak_offtopic_title_is_manual_review():
    work = CatalogWork(
        work_id="w",
        title="Biodistribution and toxicity analysis of nanoplastics in mice",
        abstract="A silver SERS assay was used.",
    )
    row = assess_corpus_quality(
        work=work,
        upstream=_upstream(),
        policy=_policy(),
        originally_selected=True,
    )
    assert row.status == "manual_review"
    assert (
        "manual_review_weak_primary_topic_title_grounding"
        in row.reasons
    )


def test_non_sers_work_is_excluded():
    work = CatalogWork(
        work_id="w",
        title="Gold silver nanowire morphology",
        abstract="No Raman method here.",
    )
    row = assess_corpus_quality(
        work=work,
        upstream=_upstream(),
        policy=_policy(),
        originally_selected=False,
    )
    assert row.status == "exclude"
    assert "missing_primary_topic_signal" in row.reasons


def test_abstract_sers_plus_grounded_title_can_pass():
    work = CatalogWork(
        work_id="w",
        title="Site-selective dealloying of gold silver nanowire arrays",
        abstract="The arrays are evaluated as SERS substrates.",
    )
    row = assess_corpus_quality(
        work=work,
        upstream=_upstream(),
        policy=_policy(),
        originally_selected=False,
    )
    assert row.status == "pass"
