from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import domains.catalysis_mechanism.prompt_builders as broad_builders
from domains.dac_her.micro_reextract_prompts import (
    build_domain_gate_recovery_prompt as legacy_domain_gate,
)
from domains.dac_her.micro_reextract_prompts import (
    build_micro_reextract_prompt as legacy_micro,
)
from domains.dac_her.prompts import (
    build_extraction_prompt as legacy_generation,
)
from domains.dac_her.semantic_patch_prompts import (
    build_patch_rejection_feedback as legacy_patch_feedback,
)
from domains.dac_her.semantic_patch_prompts import (
    build_semantic_patch_prompt as legacy_patch,
)
from domains.extraction_registry import get_extraction_adapter


ROOT = Path(__file__).resolve().parents[2]


def _empty_report():
    return SimpleNamespace(issues=[])


def test_broad_adapter_owns_native_builder_callables_and_provenance() -> None:
    adapter = get_extraction_adapter(
        "catalysis_mechanism"
    )

    expected = {
        "generation_prompt_builder":
            broad_builders.build_extraction_prompt,
        "semantic_patch_prompt_builder":
            broad_builders.build_semantic_patch_prompt,
        "patch_rejection_feedback_builder":
            broad_builders.build_patch_rejection_feedback,
        "micro_reextract_prompt_builder":
            broad_builders.build_micro_reextract_prompt,
        "domain_gate_recovery_prompt_builder":
            broad_builders.build_domain_gate_recovery_prompt,
    }

    legacy = {
        "generation_prompt_builder":
            legacy_generation,
        "semantic_patch_prompt_builder":
            legacy_patch,
        "patch_rejection_feedback_builder":
            legacy_patch_feedback,
        "micro_reextract_prompt_builder":
            legacy_micro,
        "domain_gate_recovery_prompt_builder":
            legacy_domain_gate,
    }

    for name, builder in expected.items():
        assert getattr(adapter, name) is builder

        assert (
            builder.__module__
            == "domains.catalysis_mechanism.prompt_builders"
        )

        assert (
            inspect.signature(builder)
            == inspect.signature(legacy[name])
        )

    assert {
        Path(path).resolve()
        for path
        in adapter.prompt_builder_implementation_paths()
    } == {
        Path(broad_builders.__file__).resolve()
    }

    extraction_source = (
        ROOT
        / "domains"
        / "catalysis_mechanism"
        / "extraction.py"
    ).read_text(encoding="utf-8")

    builder_source = (
        ROOT
        / "domains"
        / "catalysis_mechanism"
        / "prompt_builders.py"
    ).read_text(encoding="utf-8")

    assert (
        "extraction_prompt_compat"
        not in extraction_source
    )

    assert (
        "domains.dac_her"
        not in builder_source
    )

    assert not (
        ROOT
        / "domains"
        / "extraction_prompt_compat.py"
    ).exists()


def test_broad_generation_user_prompt_exactly_matches_pre_h1b() -> None:
    kwargs = dict(
        paper_id="BROAD_TEST",
        chunk_id="BROAD_TEST:abstract:0",
        document_id="abstract",
        document_role="abstract",
        section="Abstract",
        page_ids=(1,),
        asset_ids=(),
        asset_context="",
        vocabulary_context=(
            "REGISTERED EXPERIMENT METHODS:\n"
            "- broad_method"
        ),
        left_context="",
        core_text="Broad catalyst mechanism source text.",
        right_context="",
        validation_feedback=(
            "Synthetic validation failure."
        ),
    )

    assert (
        broad_builders.build_extraction_prompt(**kwargs)
        == legacy_generation(**kwargs)
    )


def test_broad_semantic_patch_user_prompt_exactly_matches_pre_h1b() -> None:
    kwargs = dict(
        paper_id="BROAD_TEST",
        chunk_id="BROAD_TEST:abstract:0",
        document_id="abstract",
        document_role="abstract",
        page_ids=(1,),
        asset_ids=(),
        core_text="Broad catalyst mechanism source text.",
        asset_context="",
        graph_payload={
            "paper_id": "BROAD_TEST",
            "edges": [],
        },
        report=_empty_report(),
        previous_patch_feedback=(
            "Synthetic rejected patch."
        ),
    )

    assert (
        broad_builders.build_semantic_patch_prompt(**kwargs)
        == legacy_patch(**kwargs)
    )


def test_broad_patch_rejection_feedback_exactly_matches_pre_h1b() -> None:
    error = ValueError(
        "Operation 'add_edge' requires non-null fields: [edge]"
    )

    assert (
        broad_builders.build_patch_rejection_feedback(error)
        == legacy_patch_feedback(error)
    )


def test_broad_domain_gate_user_prompt_exactly_matches_pre_h1b() -> None:
    kwargs = dict(
        paper_id="BROAD_TEST",
        chunk_id="BROAD_TEST:abstract:0",
        document_id="abstract",
        document_role="abstract",
        section="Abstract",
        page_ids=(1,),
        asset_ids=(),
        core_text="Broad catalyst mechanism source text.",
        left_context="",
        right_context="",
        asset_context="",
        rejected_graph_payload={
            "paper_id": "BROAD_TEST",
            "entities": [],
        },
        domain_error=(
            "Synthetic domain gate error."
        ),
    )

    assert (
        broad_builders.build_domain_gate_recovery_prompt(
            **kwargs
        )
        == legacy_domain_gate(**kwargs)
    )


def test_broad_micro_reextract_user_prompt_exactly_matches_pre_h1b() -> None:
    kwargs = dict(
        paper_id="BROAD_TEST",
        chunk_id="BROAD_TEST:abstract:0",
        document_id="abstract",
        document_role="abstract",
        section="Abstract",
        page_ids=(1,),
        asset_ids=(),
        core_text="Broad catalyst mechanism source text.",
        left_context="",
        right_context="",
        asset_context="",
        graph_payload={
            "paper_id": "BROAD_TEST",
            "edges": [],
        },
        report=_empty_report(),
    )

    assert (
        broad_builders.build_micro_reextract_prompt(
            **kwargs
        )
        == legacy_micro(**kwargs)
    )
