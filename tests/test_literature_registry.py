from __future__ import annotations

from pathlib import Path

from dac_her.literature_discovery.contracts import LiteratureRecord
from dac_her.literature_discovery.registry import LiteratureRegistry


def test_registry_roundtrip_and_upsert_merge(tmp_path: Path):
    path = tmp_path / "literature.json"
    registry = LiteratureRegistry(path)

    first = LiteratureRecord.from_provider_result(
        provider="openalex",
        provider_id="W1",
        title="Catalyst reconstruction",
        doi="10.1000/example",
        discovery_query="electrocatalyst reconstruction",
        mechanism_bucket="working_state_reconstruction",
    )
    second = LiteratureRecord.from_provider_result(
        provider="openalex",
        provider_id="W1",
        title="Catalyst reconstruction",
        abstract="The working catalyst reconstructs under applied potential.",
        doi="10.1000/example",
        discovery_query="potential induced reconstruction",
        mechanism_bucket="working_state_reconstruction",
    )

    registry.upsert(first)
    registry.upsert(second)
    registry.save()

    reloaded = LiteratureRegistry(path)
    record = reloaded.get(first.paper_id)
    assert record is not None
    assert record.source_depth == "abstract"
    assert record.discovery_queries == (
        "electrocatalyst reconstruction",
        "potential induced reconstruction",
    )
