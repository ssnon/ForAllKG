from pathlib import Path

from dac_her.corpus_acquisition.access_contracts import (
    AccessResolution,
    CorpusSourceAcquisitionReport,
    SourceArtifact,
)
from dac_her.corpus_acquisition.contracts import (
    AcquisitionAxis,
    AcquisitionProfile,
    CandidateAssessment,
    ScorePolicy,
    ScopePolicy,
    SelectionPolicy,
    SelectedCorpusWork,
)
from dac_her.corpus_acquisition.m3_rebase import rebase_downloaded_m3_snapshot
from dac_her.corpus_acquisition.quality_contracts import CorpusQualityAssessment
from dac_her.corpus_acquisition.source_state import atomic_write_json, write_jsonl
from pipeline_core.literature.catalog_contracts import CatalogWork, LiteratureCatalogPacket


def _profile():
    return AcquisitionProfile(
        profile_id="profile",
        domain_profile_id="sers",
        scope=ScopePolicy(),
        scoring=ScorePolicy(),
        selection=SelectionPolicy(target_total=2),
        axes=[
            AcquisitionAxis(
                axis_id="nanogap",
                target_selected=1,
                queries=["sers nanogap"],
            )
        ],
    )


def test_rebase_copies_verified_pdf_without_network(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    pdf = source / "main.pdf"
    pdf.write_bytes(b"%PDF-1.7\nsynthetic\n")

    work = CatalogWork(
        work_id="catalog_work:1",
        title="Gold silver nanogap SERS mechanism paper",
        doi="10.1000/1",
        retrieval_axis_ids=["nanogap"],
    )
    packet = LiteratureCatalogPacket(
        catalog_id="expanded",
        catalog_sha256="sha",
        acquisition_profile_id="profile",
        searched_at_utc="2026-08-14T00:00:00+00:00",
        works=[work],
        raw_work_count=1,
        canonical_work_count=1,
    )
    selected = SelectedCorpusWork(
        work_id=work.work_id,
        title=work.title,
        doi=work.doi,
        matched_axes=["nanogap"],
        primary_quota_axis="nanogap",
        total_score=10.0,
    )
    resolution = AccessResolution(
        work_id=work.work_id,
        doi=work.doi,
        status="resolved_direct_pdf",
        selected_download_url="https://example.org/a.pdf",
    )
    import hashlib

    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    artifact = SourceArtifact(
        artifact_id="artifact:1",
        work_id=work.work_id,
        role="main",
        status="downloaded",
        local_path=str(pdf),
        sha256=digest,
        byte_count=pdf.stat().st_size,
        content_type="application/pdf",
    )
    write_jsonl(source / "selected_works.jsonl", [selected])
    write_jsonl(source / "access_resolutions.jsonl", [resolution])
    write_jsonl(source / "artifacts.jsonl", [artifact])
    atomic_write_json(
        source / "acquisition_report.json",
        CorpusSourceAcquisitionReport(
            acquisition_id="old",
            source_profile_id="profile",
            source_catalog_id="base",
            source_selection_report_path="old.json",
            policy_id="policy",
            selected_work_count=1,
            artifact_downloaded_count=1,
            output_root=str(source),
        ),
    )

    assessment = CandidateAssessment(
        work_id=work.work_id,
        title=work.title,
        doi=work.doi,
        eligibility_status="eligible",
        matched_axes=["nanogap"],
        total_score=10.0,
    )
    quality = CorpusQualityAssessment(
        work_id=work.work_id,
        title=work.title,
        doi=work.doi,
        status="pass",
        original_m2_eligibility_status="eligible",
        originally_selected=True,
    )

    output = tmp_path / "rebased"
    report = rebase_downloaded_m3_snapshot(
        profile=_profile(),
        packet=packet,
        assessments=[assessment],
        quality_assessments=[quality],
        source_m3_dir=source,
        output_dir=output,
        rebase_id="rebase",
    )

    assert report["retained_downloaded_count"] == 1
    assert report["network_acquisition_performed"] is False
    copied = list((output / "artifacts").glob("*/main.pdf"))
    assert len(copied) == 1
    assert copied[0].read_bytes().startswith(b"%PDF-")
    new_report = CorpusSourceAcquisitionReport.model_validate_json(
        (output / "acquisition_report.json").read_text(encoding="utf-8")
    )
    assert new_report.source_catalog_id == "expanded"
