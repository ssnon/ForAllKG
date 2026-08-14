from __future__ import annotations

import hashlib
import json

from dac_her.corpus_acquisition.access_contracts import SourceArtifact
from dac_her.corpus_acquisition.supplementary_acquisition import (
    SupplementaryArtifactDownloader,
)
from dac_her.corpus_acquisition.supplementary_contracts import (
    SupplementaryCandidate,
    SupplementaryDiscoveryPolicy,
)


def test_existing_supplement_marker_resumes_without_duplicate_keyword(tmp_path):
    candidate = SupplementaryCandidate(
        candidate_id="candidate:1",
        work_id="work:1",
        kind="direct_file",
        resolver="public_landing_html",
        url="https://example.org/supplement.pdf",
        confidence="high",
        automatic_download_eligible=True,
    )
    policy = SupplementaryDiscoveryPolicy(
        policy_id="p",
        retries=0,
    )
    downloader = SupplementaryArtifactDownloader(policy)

    import dac_her.corpus_acquisition.supplementary_acquisition as module

    work_dir = (
        tmp_path
        / "artifacts"
        / module._safe_work_dir(candidate.work_id)
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(
        candidate.candidate_id.encode("utf-8")
    ).hexdigest()[:14]
    local_path = work_dir / f"supplementary_{stem}.pdf"
    local_path.write_bytes(b"%PDF-1.7\\nexisting")
    sha = hashlib.sha256(local_path.read_bytes()).hexdigest()

    artifact = SourceArtifact(
        artifact_id="artifact:1",
        work_id=candidate.work_id,
        role="supporting_information",
        status="downloaded",
        source_url=candidate.url,
        resolved_url=candidate.url,
        local_path=str(local_path),
        sha256=sha,
        byte_count=local_path.stat().st_size,
        content_type="application/pdf",
        acquired_at_utc="2026-08-14T00:00:00+00:00",
        acquisition_method="public_supplementary_direct_http",
        positive_evidence_promotion_performed=False,
    )
    marker_path = work_dir / f"supplementary_{stem}.artifact.json"
    marker_path.write_text(
        json.dumps(
            {
                "candidate_id": candidate.candidate_id,
                "local_path": str(local_path),
                "sha256": sha,
                "artifact": artifact.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    resumed = downloader.acquire(
        candidate=candidate,
        output_root=tmp_path,
    )
    assert resumed.status == "downloaded"
    assert resumed.sha256 == sha
    assert (
        resumed.acquisition_method
        == "resume_existing_verified_supplement"
    )
