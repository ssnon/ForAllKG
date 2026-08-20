from __future__ import annotations

import pytest

from domains.dac_her.bridge_policy import BRIDGE_POLICY_VERSION, partition_bridge_result
from domains.dac_her.bridge_prompts import (
    BRIDGE_PROMPT_VERSION,
    BRIDGE_SYSTEM_PROMPT,
    build_bridge_prompt,
)
from domains.dac_her.bridge_recovery_prompts import (
    BRIDGE_RECOVERY_PROMPT_VERSION,
    BRIDGE_RECOVERY_SYSTEM_PROMPT,
    build_bridge_candidate_repair_prompt,
)
from dac_her.bridge_validation import (
    bridge_validation_issues,
    validate_bridge_chunk,
)
from dac_her.domains.bridge_registry import (
    available_bridge_adapters,
    get_bridge_adapter,
)
from domains.dac_her.scientific_signatures import strict_node_catalog


def test_alpha4b1_dac_her_adapter_is_exact_legacy_wrapper():
    adapter = get_bridge_adapter("dac_her")

    assert adapter.adapter_id == "dac_her"
    assert adapter.domain_profile_id == "dac_her"
    assert adapter.prompt_version == BRIDGE_PROMPT_VERSION
    assert adapter.system_prompt == BRIDGE_SYSTEM_PROMPT
    assert adapter.build_prompt is build_bridge_prompt
    assert adapter.recovery_prompt_version == BRIDGE_RECOVERY_PROMPT_VERSION
    assert adapter.recovery_system_prompt == BRIDGE_RECOVERY_SYSTEM_PROMPT
    assert adapter.build_candidate_repair_prompt is build_bridge_candidate_repair_prompt
    assert adapter.policy_version == BRIDGE_POLICY_VERSION
    assert adapter.strict_node_catalog_builder is strict_node_catalog
    assert adapter.validation_issues is bridge_validation_issues
    assert adapter.validate_chunk is validate_bridge_chunk
    assert adapter.partition_result is partition_bridge_result


def test_alpha4b1_dac_her_bridge_adapter_remains_registered():
    # alpha4b.1 established DAC-HER registration. Later phases may register
    # additional domain adapters, so preserve the invariant rather than
    # freezing the registry cardinality.
    assert "dac_her" in available_bridge_adapters()


@pytest.mark.parametrize("profile_id", ["catalysis_mechanism"])
def test_alpha4b1_still_unimplemented_domains_fail_closed(profile_id: str):
    # SERS is intentionally implemented in alpha4b.2b. Keep fail-closed
    # coverage for domains that still have no Bridge adapter.
    with pytest.raises(ValueError, match="has no Bridge adapter"):
        get_bridge_adapter(profile_id)
