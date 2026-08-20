from __future__ import annotations

from pathlib import Path

from dac_her.corpus_acquisition.access_contracts import (
    AccessLocation,
    AccessResolution,
    SourceArtifact,
)
from dac_her.corpus_acquisition.backfill_contracts import (
    AcquisitionAwareBackfillPolicy,
)
from dac_her.corpus_acquisition.backfill_engine import (
    candidate_rank_key,
    choose_most_constrained_axis,
    run_acquisition_aware_backfill,
)
from dac_her.corpus_acquisition.contracts import (
    AcquisitionAxis,
    AcquisitionProfile,
    CandidateAssessment,
    SelectionPolicy,
    SelectedCorpusWork,
)
from dac_her.corpus_acquisition.quality_contracts import (
    CorpusQualityAssessment,
)
from pipeline_core.literature.catalog_contracts import CatalogWork


def _profile():
    return AcquisitionProfile(
        profile_id="p",
        domain_profile_id="d",
        selection=SelectionPolicy(target_total=2),
        axes=[
            AcquisitionAxis(
                axis_id="a",
                target_selected=1,
                queries=["a"],
            ),
            AcquisitionAxis(
                axis_id="b",
                target_selected=1,
                queries=["b"],
            ),
        ],
    )


def _assessment(work_id, axes, score, oa=False):
    return CandidateAssessment(
        work_id=work_id,
        title=work_id,
        eligibility_status="eligible",
        matched_axes=axes,
        total_score=score,
        open_access_available=oa,
    )


def _quality(work_id):
    return CorpusQualityAssessment(
        work_id=work_id,
        title=work_id,
        status="pass",
        original_m2_eligibility_status="eligible",
    )


def _download(tmp_path, work_id):
    path = tmp_path / f"{work_id}.pdf"
    path.write_bytes(b"%PDF-1.7\nx")
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    resolution = AccessResolution(
        work_id=work_id,
        status="resolved_direct_pdf",
        locations=[
            AccessLocation(
                location_id=f"loc:{work_id}",
                resolver="test",
                url=f"https://example.org/{work_id}.pdf",
                url_for_pdf=f"https://example.org/{work_id}.pdf",
                automatic_download_eligible=True,
            )
        ],
        selected_location_id=f"loc:{work_id}",
        selected_download_url=f"https://example.org/{work_id}.pdf",
    )
    artifact = SourceArtifact(
        artifact_id=f"artifact:{work_id}",
        work_id=work_id,
        role="main",
        status="downloaded",
        local_path=str(path),
        sha256=sha,
    )
    return resolution, artifact


def test_oa_is_tiebreak_only_after_scientific_score():
    policy = AcquisitionAwareBackfillPolicy(policy_id="x")
    high = CatalogWork(work_id="high", title="high")
    low = CatalogWork(
        work_id="low",
        title="low",
        open_access_url="https://example.org/low.pdf",
    )
    high_a = _assessment("high", ["a"], 10.0, False)
    low_a = _assessment("low", ["a"], 9.0, True)
    assert (
        candidate_rank_key(
            work=high,
            assessment=high_a,
            policy=policy,
        )
        <
        candidate_rank_key(
            work=low,
            assessment=low_a,
            policy=policy,
        )
    )


def test_most_constrained_axis_prefers_scarce_pool():
    assessment_map = {
        "w1": _assessment("w1", ["a", "b"], 5),
        "w2": _assessment("w2", ["a"], 4),
        "w3": _assessment("w3", ["a"], 3),
    }
    assert choose_most_constrained_axis(
        deficits={"a": 1, "b": 1},
        available_work_ids=set(assessment_map),
        assessment_map=assessment_map,
        axis_order=["a", "b"],
    ) == "b"


def test_backfill_replaces_failed_slot_with_quality_pass_candidate(tmp_path):
    profile = _profile()
    policy = AcquisitionAwareBackfillPolicy(policy_id="x")
    works = {
        wid: CatalogWork(work_id=wid, title=wid)
        for wid in ["old_a", "old_b", "new_b"]
    }
    assessments = {
        "old_a": _assessment("old_a", ["a"], 10),
        "old_b": _assessment("old_b", ["b"], 10),
        "new_b": _assessment("new_b", ["b"], 9),
    }
    quality = {
        wid: _quality(wid)
        for wid in works
    }
    selected = [
        SelectedCorpusWork(
            work_id="old_a",
            title="old_a",
            matched_axes=["a"],
            primary_quota_axis="a",
            total_score=10,
        ),
        SelectedCorpusWork(
            work_id="old_b",
            title="old_b",
            matched_axes=["b"],
            primary_quota_axis="b",
            total_score=10,
        ),
    ]
    old_a_res, old_a_art = _download(tmp_path, "old_a")
    old_b_res = AccessResolution(
        work_id="old_b",
        status="unresolved",
    )
    old_b_art = SourceArtifact(
        artifact_id="artifact:old_b",
        work_id="old_b",
        role="main",
        status="not_attempted",
    )

    def acquire(work):
        assert work.work_id == "new_b"
        resolution, artifact = _download(
            tmp_path,
            work.work_id,
        )
        return resolution, artifact, False

    (
        final,
        attempts,
        _,
        _,
        initial_counts,
        final_counts,
        _,
    ) = run_acquisition_aware_backfill(
        profile=profile,
        policy=policy,
        work_map=works,
        assessment_map=assessments,
        quality_map=quality,
        starting_selected=selected,
        starting_resolution_map={
            "old_a": old_a_res,
            "old_b": old_b_res,
        },
        starting_artifact_map={
            "old_a": old_a_art,
            "old_b": old_b_art,
        },
        acquire_fn=acquire,
        project_root=tmp_path,
    )
    assert {row.work_id for row in final} == {
        "old_a",
        "new_b",
    }
    assert initial_counts == {"a": 1, "b": 0}
    assert final_counts == {"a": 1, "b": 1}
    assert len(attempts) == 1
    assert attempts[0].requested_axis == "b"
