from __future__ import annotations

import pipeline_core.corpus.graph.graph_validation as graph_validation
from pipeline_core.corpus.extraction.draft_schema import KnowledgeGraphDraft


def _empty_draft() -> KnowledgeGraphDraft:
    return KnowledgeGraphDraft.model_validate(
        {
            "paper_id": "paper",
            "chunk_id": "chunk",
            "section": "Results",
            "document_id": "main",
            "document_role": "main",
            "page_ids": [],
            "asset_ids": [],
            "entities": [],
            "experiments": [],
            "calculations": [],
            "measurements": [],
            "measurement_groups": [],
            "observation_claims": [],
            "mechanism_claims": [],
            "edges": [],
        }
    )


def test_no_contract_path_delegates_to_legacy_compat(
    monkeypatch,
):
    calls = []

    def fake_legacy_compat(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        graph_validation,
        "append_legacy_dac_relation_compat_issues",
        fake_legacy_compat,
    )

    report = graph_validation.collect_graph_issues(
        _empty_draft()
    )

    assert report.valid
    assert len(calls) == 1
    assert calls[0]["graph"].paper_id == "paper"


def test_explicit_empty_contract_bypasses_legacy_compat(
    monkeypatch,
):
    def forbidden_legacy_compat(**kwargs):
        raise AssertionError(
            "legacy DAC fallback must not run when an "
            "explicit relation contract is supplied"
        )

    monkeypatch.setattr(
        graph_validation,
        "append_legacy_dac_relation_compat_issues",
        forbidden_legacy_compat,
    )

    report = graph_validation.collect_graph_issues(
        _empty_draft(),
        relation_constraints=(),
    )

    assert report.valid
