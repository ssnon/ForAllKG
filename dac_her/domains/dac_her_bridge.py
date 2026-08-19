from __future__ import annotations

import dac_her.bridge_policy as bridge_policy_module
import domains.dac_her.bridge_prompts as bridge_prompts_module
import dac_her.bridge_validation as bridge_validation_module
import dac_her.scientific_signatures as scientific_signatures_module

from dac_her.bridge_domain import (
    BridgeDomainAdapter,
    BridgeImplementationFiles,
)
from dac_her.bridge_policy import (
    BRIDGE_POLICY_VERSION,
    partition_bridge_result,
)
from domains.dac_her.bridge_prompts import (
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
from dac_her.scientific_signatures import (
    strict_node_catalog,
    strong_anchor_context_issues,
)


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
    anchor_context_issues=strong_anchor_context_issues,
    implementation_files=BridgeImplementationFiles(
        extraction=(
            __file__,
            bridge_prompts_module.__file__,
            bridge_validation_module.__file__,
            scientific_signatures_module.__file__,
        ),
        policy=(
            __file__,
            bridge_policy_module.__file__,
            scientific_signatures_module.__file__,
        ),
    ),
)
