from __future__ import annotations

from dac_her.bridge_domain import BridgeDomainAdapter
from dac_her.bridge_policy import (
    BRIDGE_POLICY_VERSION,
    partition_bridge_result,
)
from dac_her.bridge_prompts import (
    BRIDGE_PROMPT_VERSION,
    BRIDGE_SYSTEM_PROMPT,
    build_bridge_prompt,
)
from dac_her.bridge_recovery_prompts import (
    BRIDGE_RECOVERY_PROMPT_VERSION,
    BRIDGE_RECOVERY_SYSTEM_PROMPT,
    build_bridge_candidate_repair_prompt,
)
from dac_her.bridge_validation import (
    bridge_validation_issues,
    validate_bridge_chunk,
)
from dac_her.scientific_signatures import strict_node_catalog


DAC_HER_BRIDGE_ADAPTER = BridgeDomainAdapter(
    adapter_id="dac_her",
    domain_profile_id="dac_her",
    prompt_version=BRIDGE_PROMPT_VERSION,
    system_prompt=BRIDGE_SYSTEM_PROMPT,
    build_prompt=build_bridge_prompt,
    recovery_prompt_version=BRIDGE_RECOVERY_PROMPT_VERSION,
    recovery_system_prompt=BRIDGE_RECOVERY_SYSTEM_PROMPT,
    build_candidate_repair_prompt=build_bridge_candidate_repair_prompt,
    policy_version=BRIDGE_POLICY_VERSION,
    strict_node_catalog_builder=strict_node_catalog,
    validation_issues=bridge_validation_issues,
    validate_chunk=validate_bridge_chunk,
    partition_result=partition_bridge_result,
)
