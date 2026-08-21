from __future__ import annotations

from pathlib import Path

from domains.dac_her.micro_reextract_prompts import (
    build_domain_gate_recovery_prompt,
    build_micro_reextract_prompt,
)
from domains.dac_her.prompts import build_extraction_prompt
from domains.dac_her.semantic_patch_prompts import (
    build_patch_rejection_feedback,
    build_semantic_patch_prompt,
)
from domains.extraction_registry import get_extraction_adapter


def test_dac_adapter_preserves_native_prompt_builder_callables() -> None:
    adapter = get_extraction_adapter("dac_her")

    assert (
        adapter.generation_prompt_builder
        is build_extraction_prompt
    )
    assert (
        adapter.semantic_patch_prompt_builder
        is build_semantic_patch_prompt
    )
    assert (
        adapter.patch_rejection_feedback_builder
        is build_patch_rejection_feedback
    )
    assert (
        adapter.micro_reextract_prompt_builder
        is build_micro_reextract_prompt
    )
    assert (
        adapter.domain_gate_recovery_prompt_builder
        is build_domain_gate_recovery_prompt
    )


def test_dac_prompt_builder_provenance_tracks_native_source_files() -> None:
    adapter = get_extraction_adapter("dac_her")

    expected = {
        Path(build_extraction_prompt.__code__.co_filename).resolve(),
        Path(build_semantic_patch_prompt.__code__.co_filename).resolve(),
        Path(build_micro_reextract_prompt.__code__.co_filename).resolve(),
    }

    actual = {
        Path(path).resolve()
        for path
        in adapter.prompt_builder_implementation_paths()
    }

    assert actual == expected
