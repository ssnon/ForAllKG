import sys
from types import SimpleNamespace

import pytest

import scripts.literature.expand_literature_catalog as module
from pipeline_core.literature.catalog_contracts import (
    LiteratureCatalogPacket,
)


def _packet(
    *,
    catalog_id: str,
) -> LiteratureCatalogPacket:
    return LiteratureCatalogPacket.model_validate(
        {
            "schema_version":
                "literature-catalog-packet-v1",
            "catalog_id":
                catalog_id,
            "acquisition_profile_id":
                "test-profile",
            "searched_at_utc":
                "2026-08-20T00:00:00+00:00",
            "providers_requested":
                ["openalex"],
            "queries":
                [],
            "works":
                [],
            "executions":
                [],
            "raw_work_count":
                0,
            "canonical_work_count":
                0,
            "deduplicated_work_count":
                0,
            "supplementary_records_collapsed":
                0,
            "epistemic_usage":
                "candidate_source_only_not_positive_premise",
            "catalog_sha256":
                "test-sha",
        }
    )


def test_incoming_packet_is_checkpointed_before_append_failure(
    tmp_path,
    monkeypatch,
) -> None:
    base = _packet(
        catalog_id="base",
    )

    incoming = _packet(
        catalog_id="incoming",
    )

    base_path = tmp_path / "base.json"

    base_path.write_text(
        base.model_dump_json(),
        encoding="utf-8",
    )

    output = tmp_path / "out"

    monkeypatch.setattr(
        module,
        "load_acquisition_profile",
        lambda _: SimpleNamespace(
            profile_id="test-profile"
        ),
    )

    monkeypatch.setattr(
        module,
        "build_catalog_queries",
        lambda _: [],
    )

    monkeypatch.setattr(
        module,
        "_providers",
        lambda _: [object()],
    )

    class _Retriever:
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            pass

        def retrieve(
            self,
            **kwargs,
        ):
            return SimpleNamespace(
                packet=incoming
            )

    monkeypatch.setattr(
        module,
        "LiteratureCatalogRetriever",
        _Retriever,
    )

    def _fail_append(**kwargs):
        raise RuntimeError(
            "synthetic append failure"
        )

    monkeypatch.setattr(
        module,
        "append_catalog_expansion",
        _fail_append,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expand_literature_catalog",
            "--profile",
            str(tmp_path / "profile.yaml"),
            "--base-catalog",
            str(base_path),
            "--output-dir",
            str(output),
            "--expansion-id",
            "checkpoint-test",
            "--providers",
            "openalex",
            "--results-per-query",
            "100",
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic append failure",
    ):
        module.main()

    checkpoint = (
        output
        / "incoming_catalog.json"
    )

    assert checkpoint.is_file()

    observed = (
        LiteratureCatalogPacket
        .model_validate_json(
            checkpoint.read_text(
                encoding="utf-8"
            )
        )
    )

    assert observed.catalog_id == "incoming"
