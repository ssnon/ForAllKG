from dac_her.llm_openrouter import OpenRouterLLM
from dac_her.bridge_schemas import BridgeChunkGraph
from dac_her.bridge_policy import BridgeRejection
from pathlib import Path
from typing import Any

REPAIRABLE_BRIDGE_CODES = {
    "SUBJECT_EVIDENCE_NOT_IN_SPAN",
    "RELATION_EVIDENCE_NOT_IN_SPAN",
    "OBJECT_EVIDENCE_NOT_IN_SPAN",
    "RELATION_CUE_MISMATCH",
}

def repair_rejected_bridge_candidates(
    *,
    llm: OpenRouterLLM,
    accepted_result: BridgeChunkGraph,
    rejections: list[BridgeRejection],
    strict_nodes: list[
        dict[str, Any]
    ],
    source_payload: dict[
        str,
        Any,
    ],
    max_tokens: int,
    debug_path: Path,
) -> tuple[
    BridgeChunkGraph,
    list[BridgeRejection],
]:
    ...