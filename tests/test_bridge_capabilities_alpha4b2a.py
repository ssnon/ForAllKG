from pathlib import Path

from dac_her.bridge_policy import BRIDGE_POLICY_VERSION, partition_bridge_result
from dac_her.bridge_validation import bridge_validation_issues, validate_bridge_chunk
from dac_her.domains.bridge_registry import get_bridge_adapter
from dac_her.scientific_signatures import (
    strict_node_catalog,
    strong_anchor_context_issues,
)


def _names(paths: tuple[str, ...]) -> set[str]:
    return {Path(path).name for path in paths}


def test_alpha4b2a_her_capability_facade_preserves_legacy_callbacks():
    adapter = get_bridge_adapter("dac_her")
    assert adapter.signatures.catalog_builder is strict_node_catalog
    assert adapter.signatures.anchor_context_issues is strong_anchor_context_issues
    assert adapter.validation.issues is bridge_validation_issues
    assert adapter.validation.validate is validate_bridge_chunk
    assert adapter.policy.version == BRIDGE_POLICY_VERSION
    assert adapter.policy.partition is partition_bridge_result


def test_alpha4b2a_her_domain_owned_fingerprint_files_preserve_legacy_set():
    adapter = get_bridge_adapter("dac_her")
    assert _names(adapter.implementation_files.extraction) == {
        "dac_her_bridge.py",
        "bridge_prompts.py",
        "bridge_validation.py",
        "scientific_signatures.py",
    }
    assert _names(adapter.implementation_files.policy) == {
        "dac_her_bridge.py",
        "bridge_policy.py",
        "scientific_signatures.py",
    }


def test_alpha4b2a_implementation_stage_accessor():
    adapter = get_bridge_adapter("dac_her")
    assert adapter.implementation_files.for_stage("extraction") == adapter.implementation_files.extraction
    assert adapter.implementation_files.for_stage("policy") == adapter.implementation_files.policy
