from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import domains.sers.prompt_builders as sers_builders
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


def test_sers_adapter_owns_native_builder_callables_and_provenance() -> None:
    adapter = get_extraction_adapter("sers_au_ag")

    expected = {
        "generation_prompt_builder":
            sers_builders.build_extraction_prompt,
        "semantic_patch_prompt_builder":
            sers_builders.build_semantic_patch_prompt,
        "patch_rejection_feedback_builder":
            sers_builders.build_patch_rejection_feedback,
        "micro_reextract_prompt_builder":
            sers_builders.build_micro_reextract_prompt,
        "domain_gate_recovery_prompt_builder":
            sers_builders.build_domain_gate_recovery_prompt,
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
        assert builder.__module__ == "domains.sers.prompt_builders"
        assert (
            inspect.signature(builder)
            == inspect.signature(legacy[name])
        )

    assert {
        Path(path).resolve()
        for path
        in adapter.prompt_builder_implementation_paths()
    } == {
        Path(sers_builders.__file__).resolve()
    }

    extraction_source = (
        ROOT / "domains" / "sers" / "extraction.py"
    ).read_text(encoding="utf-8")

    builder_source = (
        ROOT / "domains" / "sers" / "prompt_builders.py"
    ).read_text(encoding="utf-8")

    assert "extraction_prompt_compat" not in extraction_source
    assert "domains.dac_her" not in builder_source


def test_sers_generation_user_prompt_exactly_matches_pre_h1b() -> None:
    kwargs = dict(
        paper_id="SERS_TEST",
        chunk_id="SERS_TEST:main:0",
        document_id="main",
        document_role="main",
        section="Results",
        page_ids=(1, 2),
        asset_ids=("fig_1",),
        asset_context="Figure 1 caption.",
        vocabulary_context="REGISTERED EXPERIMENT METHODS:\n- sers_test",
        left_context="Left context.",
        core_text="Core scientific source text.",
        right_context="Right context.",
        validation_feedback="Synthetic validation failure.",
    )

    assert (
        sers_builders.build_extraction_prompt(**kwargs)
        == legacy_generation(**kwargs)
    )


def test_sers_semantic_patch_user_prompt_exactly_matches_pre_h1b() -> None:
    kwargs = dict(
        paper_id="SERS_TEST",
        chunk_id="SERS_TEST:main:0",
        document_id="main",
        document_role="main",
        page_ids=(1,),
        asset_ids=(),
        core_text="Core scientific source text.",
        asset_context="",
        graph_payload={
            "paper_id": "SERS_TEST",
            "edges": [],
        },
        report=_empty_report(),
        previous_patch_feedback="Synthetic rejected patch.",
    )

    assert (
        sers_builders.build_semantic_patch_prompt(**kwargs)
        == legacy_patch(**kwargs)
    )


def test_sers_patch_rejection_feedback_exactly_matches_pre_h1b() -> None:
    error = ValueError(
        "Operation 'add_edge' requires non-null fields: [edge]"
    )

    assert (
        sers_builders.build_patch_rejection_feedback(error)
        == legacy_patch_feedback(error)
    )


def test_sers_domain_gate_user_prompt_exactly_matches_pre_h1b() -> None:
    kwargs = dict(
        paper_id="SERS_TEST",
        chunk_id="SERS_TEST:main:0",
        document_id="main",
        document_role="main",
        section="Methods",
        page_ids=(2,),
        asset_ids=(),
        core_text="Core scientific source text.",
        left_context="Left.",
        right_context="Right.",
        asset_context="",
        rejected_graph_payload={
            "paper_id": "SERS_TEST",
            "entities": [],
        },
        domain_error="Synthetic domain gate error.",
    )

    assert (
        sers_builders.build_domain_gate_recovery_prompt(**kwargs)
        == legacy_domain_gate(**kwargs)
    )


def test_sers_micro_reextract_user_prompt_exactly_matches_pre_h1b() -> None:
    kwargs = dict(
        paper_id="SERS_TEST",
        chunk_id="SERS_TEST:main:0",
        document_id="main",
        document_role="main",
        section="Results",
        page_ids=(3,),
        asset_ids=("fig_3",),
        core_text="Core scientific source text.",
        left_context="Left.",
        right_context="Right.",
        asset_context="Figure 3.",
        graph_payload={
            "paper_id": "SERS_TEST",
            "edges": [],
        },
        report=_empty_report(),
    )

    assert (
        sers_builders.build_micro_reextract_prompt(**kwargs)
        == legacy_micro(**kwargs)
    )
