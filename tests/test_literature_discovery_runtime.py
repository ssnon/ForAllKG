from __future__ import annotations

import json
from pathlib import Path

from pipeline_core.literature.discovery import (
    LiteratureRecord,
    LiteratureRegistry,
    load_query_plan,
    run_discovery,
    select_pilot_requests,
)


CONFIG = Path("configs/literature/broad_catalysis_v1.yaml")


class FakeProvider:
    provider_name = "fake"

    def search(self, request):
        shared = LiteratureRecord.from_provider_result(
            provider="fake",
            provider_id=f"shared-{request.mechanism_bucket}",
            title="Shared paper",
            abstract="Shared mechanism abstract",
            doi="10.1234/shared",
            year=2025,
            discovery_query=request.query,
            mechanism_bucket=request.mechanism_bucket,
        )
        unique = LiteratureRecord.from_provider_result(
            provider="fake",
            provider_id=f"unique-{request.mechanism_bucket}",
            title=f"Unique {request.mechanism_bucket}",
            abstract="Unique abstract",
            doi=f"10.1234/{request.mechanism_bucket}",
            year=2025,
            discovery_query=request.query,
            mechanism_bucket=request.mechanism_bucket,
        )
        return [shared, unique]


def test_pilot_selection_round_robins_mechanism_buckets():
    plan = load_query_plan(CONFIG)
    requests = select_pilot_requests(plan, query_count=5, per_query_limit=100)

    assert [item.mechanism_bucket for item in requests] == [
        "working_state_reconstruction",
        "elementary_step_kinetics",
        "interfacial_environment",
        "structural_landscape",
        "active_site_attribution",
    ]
    assert all(item.limit == 100 for item in requests)


def test_run_discovery_deduplicates_doi_and_writes_artifacts(tmp_path: Path):
    plan = load_query_plan(CONFIG)
    requests = select_pilot_requests(plan, query_count=2, per_query_limit=100)
    registry = LiteratureRegistry(tmp_path / "registry.json")

    artifacts = run_discovery(
        provider=FakeProvider(),
        plan=plan,
        requests=requests,
        registry=registry,
        output_dir=tmp_path / "run",
    )

    assert artifacts.raw_candidates == 4
    assert artifacts.unique_candidates == 3
    candidate_rows = [
        json.loads(line)
        for line in artifacts.candidates_path.read_text(encoding="utf-8").splitlines()
    ]
    shared = next(row for row in candidate_rows if row["doi"] == "10.1234/shared")
    assert set(shared["mechanism_buckets"]) == {
        "working_state_reconstruction",
        "elementary_step_kinetics",
    }
    assert len(shared["discovery_queries"]) == 2

    run = json.loads(artifacts.run_path.read_text(encoding="utf-8"))
    assert run["status"] == "complete"
    assert run["query_count"] == 2
    assert run["raw_candidate_count"] == 4
    assert run["unique_candidate_count"] == 3
    assert run["duplicate_or_repeat_count"] == 1
    assert len(registry.entries) == 3
