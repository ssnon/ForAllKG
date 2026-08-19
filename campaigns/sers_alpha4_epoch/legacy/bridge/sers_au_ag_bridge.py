from __future__ import annotations

import pipeline_core.bridge_policy_runtime as bridge_policy_runtime_module
import pipeline_core.bridge_validation as bridge_validation_module
import campaigns.sers_alpha4_epoch.legacy.bridge.sers_bridge_policy as sers_bridge_policy_module
import campaigns.sers_alpha4_epoch.legacy.bridge.sers_bridge_prompts as sers_bridge_prompts_module
import campaigns.sers_alpha4_epoch.legacy.bridge.sers_bridge_recovery_prompts as sers_bridge_recovery_prompts_module
import campaigns.sers_alpha4_epoch.legacy.bridge.sers_bridge_signatures as sers_bridge_signatures_module

from dac_her.bridge_domain import (
    BridgeDomainAdapter,
    BridgeImplementationFiles,
)
from pipeline_core.bridge_validation import bind_bridge_validation
from campaigns.sers_alpha4_epoch.legacy.bridge.sers_bridge_policy import (
    SERS_BRIDGE_POLICY_VERSION,
    partition_sers_bridge_result,
)
from campaigns.sers_alpha4_epoch.legacy.bridge.sers_bridge_prompts import (
    SERS_BRIDGE_PROMPT_VERSION,
    SERS_BRIDGE_SYSTEM_PROMPT,
    build_sers_bridge_prompt,
)
from campaigns.sers_alpha4_epoch.legacy.bridge.sers_bridge_recovery_prompts import (
    SERS_BRIDGE_RECOVERY_PROMPT_VERSION,
    SERS_BRIDGE_RECOVERY_SYSTEM_PROMPT,
    build_sers_bridge_candidate_repair_prompt,
)
from campaigns.sers_alpha4_epoch.legacy.bridge.sers_bridge_signatures import (
    strict_node_catalog,
    strong_anchor_context_issues,
)


(
    SERS_BRIDGE_VALIDATION_ISSUES,
    VALIDATE_SERS_BRIDGE_CHUNK,
) = bind_bridge_validation(strong_anchor_context_issues)


SERS_AU_AG_BRIDGE_ADAPTER = BridgeDomainAdapter(
    adapter_id='sers_au_ag',
    domain_profile_id='sers_au_ag',
    prompt_version=SERS_BRIDGE_PROMPT_VERSION,
    system_prompt=SERS_BRIDGE_SYSTEM_PROMPT,
    build_prompt=build_sers_bridge_prompt,
    recovery_prompt_version=SERS_BRIDGE_RECOVERY_PROMPT_VERSION,
    recovery_system_prompt=SERS_BRIDGE_RECOVERY_SYSTEM_PROMPT,
    build_candidate_repair_prompt=build_sers_bridge_candidate_repair_prompt,
    policy_version=SERS_BRIDGE_POLICY_VERSION,
    strict_node_catalog_builder=strict_node_catalog,
    validation_issues=SERS_BRIDGE_VALIDATION_ISSUES,
    validate_chunk=VALIDATE_SERS_BRIDGE_CHUNK,
    partition_result=partition_sers_bridge_result,
    anchor_context_issues=strong_anchor_context_issues,
    implementation_files=BridgeImplementationFiles(
        extraction=(
            __file__,
            sers_bridge_prompts_module.__file__,
            sers_bridge_recovery_prompts_module.__file__,
            sers_bridge_signatures_module.__file__,
            bridge_validation_module.__file__,
        ),
        policy=(
            __file__,
            sers_bridge_policy_module.__file__,
            bridge_policy_runtime_module.__file__,
            sers_bridge_signatures_module.__file__,
        ),
    ),
)
