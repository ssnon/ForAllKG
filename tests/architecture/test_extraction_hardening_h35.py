from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from domains.extraction_registry import (
    get_extraction_adapter,
)
from pipeline_core.domain.extraction_domain import (
    ExtractionDomainAdapter,
)


ROOT = Path(__file__).resolve().parents[2]


class DummyCompactModel(BaseModel):
    value: str = "ok"


def _text_builder(**_kwargs) -> str:
    return "prompt"


def _feedback_builder(error: Exception) -> str:
    return str(error)


def _policy_transform(policy):
    return policy


def _vocabulary_builder(_registry) -> str:
    return "vocabulary"


def _semantic_collector(_draft):
    return []


def _adapter(**overrides) -> ExtractionDomainAdapter:
    values = {
        "adapter_id": "test",
        "domain_profile_id": "test",
        "prompt_version": "test-v1",
        "system_prompt": "system",
        "patch_system_prompt": "patch",
        "micro_reextract_system_prompt": "micro",
        "generation_prompt_builder": _text_builder,
        "semantic_patch_prompt_builder": _text_builder,
        "patch_rejection_feedback_builder": _feedback_builder,
        "micro_reextract_prompt_builder": _text_builder,
        "domain_gate_recovery_prompt_builder": _text_builder,
        "default_data_root": "data_test",
        "allowed_entity_types": frozenset({"Entity"}),
        "allowed_relation_types": frozenset({"RELATED_TO"}),
    }

    values.update(overrides)

    return ExtractionDomainAdapter(
        **values
    )


def test_runtime_llm_identity_and_debug_fallback_are_domain_neutral():
    path = (
        ROOT
        / "scripts"
        / "corpus"
        / "strict_extraction_runtime.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )

    assert "GraphAgents DAC-HER" not in source

    assert (
        "data_dac/debug/"
        "last_invalid_structured_response.json"
        not in source
    )

    assert (
        'application_title="ForAllKG"'
        in source
    )

    assert (
        "default_debug_path="
        "default_llm_debug_path"
        in source
    )

    assert (
        'f"{safe_chunk_id}'
        '__last_invalid_structured_response.json"'
        in source
    )


def test_valid_registered_adapters_satisfy_constructor_invariants():
    for profile_id in (
        "dac_her",
        "sers_au_ag",
        "catalysis_mechanism",
    ):
        adapter = get_extraction_adapter(
            profile_id
        )

        assert isinstance(
            adapter,
            ExtractionDomainAdapter,
        )


def test_compact_generation_capability_requires_model_and_id():
    with pytest.raises(
        ValueError,
        match="compact generation",
    ):
        _adapter(
            compact_generation_response_model=(
                DummyCompactModel
            ),
        )

    with pytest.raises(
        ValueError,
        match="compact generation",
    ):
        _adapter(
            compact_generation_schema_id=(
                "compact-v1"
            ),
        )


def test_compact_recovery_capability_requires_model_and_id():
    with pytest.raises(
        ValueError,
        match="compact domain-gate recovery",
    ):
        _adapter(
            compact_domain_gate_recovery_response_model=(
                DummyCompactModel
            ),
        )

    with pytest.raises(
        ValueError,
        match="compact domain-gate recovery",
    ):
        _adapter(
            compact_domain_gate_recovery_schema_id=(
                "recovery-v1"
            ),
        )


def test_extraction_policy_capability_requires_transform_and_id():
    with pytest.raises(
        ValueError,
        match="extraction policy",
    ):
        _adapter(
            extraction_policy_transform=(
                _policy_transform
            ),
        )

    with pytest.raises(
        ValueError,
        match="extraction policy",
    ):
        _adapter(
            extraction_policy_id="policy-v1",
        )


def test_reduced_vocabulary_capability_requires_builder_and_id():
    with pytest.raises(
        ValueError,
        match="reduced vocabulary context",
    ):
        _adapter(
            reduced_vocabulary_context_builder=(
                _vocabulary_builder
            ),
        )

    with pytest.raises(
        ValueError,
        match="reduced vocabulary context",
    ):
        _adapter(
            reduced_vocabulary_context_id=(
                "vocabulary-v1"
            ),
        )


def test_strict_semantic_capability_requires_collector_and_contract_id():
    with pytest.raises(
        ValueError,
        match="strict semantic validation",
    ):
        _adapter(
            strict_semantic_issue_collector=(
                _semantic_collector
            ),
        )

    with pytest.raises(
        ValueError,
        match="strict semantic validation",
    ):
        _adapter(
            strict_semantic_contract_id=(
                "semantic-v1"
            ),
        )


def test_semantic_rules_cannot_exist_without_semantic_capability():
    with pytest.raises(
        ValueError,
        match="strict semantic contract rules",
    ):
        _adapter(
            strict_semantic_contract_rules=(
                "rule-1",
            ),
        )


def test_compact_fingerprint_includes_all_active_model_implementation_paths():
    path = (
        ROOT
        / "scripts"
        / "corpus"
        / "extract_paper.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )

    assert (
        "compact_response_model_implementation_paths()[0]"
        not in source
    )

    assert (
        ".compact_response_model_implementation_paths()"
        in source
    )

    assert (
        "args.compact_generation_schema"
        in source
    )

    assert (
        "args.compact_domain_gate_recovery"
        in source
    )
