from __future__ import annotations

import json
from typing import Any


SERS_BRIDGE_RECOVERY_PROMPT_VERSION = (
    'sers-au-ag-bridge-candidate-recovery-v1-alpha4b2b'
)

SERS_BRIDGE_RECOVERY_SYSTEM_PROMPT = r"""
You repair exactly ONE previously extracted Bridge candidate for an Au-Ag SERS
knowledge-graph pipeline.

The original candidate already expresses the intended scientific content. Only
repair hard grounding/representation failures. Do not broaden, strengthen,
reinterpret, or replace the scientific relation.

ALLOWED
- replace grounding/evidence phrases with verbatim CORE_TEXT substrings that
  express the SAME content;
- replace anchor_id with a more compatible supplied STRICT_GRAPH_NODE;
- accepted_pattern must use EXPRESSES_PATTERN and RelationPattern;
- correct EvidencePointers using only supplied document/pages/assets.

FORBIDDEN
- changing pattern_subject/relation/object/relation_strength;
- changing retention_lane or concept ID;
- inventing a new SERS relation, mechanism, condition, or concept;
- using text outside CORE_TEXT;
- guessing an anchor when no compatible strict node exists.

If repair is impossible, return repairable=false, concept=null, links=[] and
explain why. Missing detail in an anchor label is not itself a contradiction;
explicit incompatible metal identity is.
""".strip()


def build_sers_bridge_candidate_repair_prompt(
    *,
    original_concept: dict[str, Any],
    original_links: list[dict[str, Any]],
    validation_issues: list[str],
    strict_nodes: list[dict[str, Any]],
    core_text: str,
    source_metadata: dict[str, Any],
) -> str:
    return f"""
SOURCE METADATA:
{json.dumps(source_metadata, ensure_ascii=False, indent=2)}

STRICT_GRAPH_NODES:
{json.dumps(strict_nodes, ensure_ascii=False, indent=2)}

ORIGINAL CONCEPT:
{json.dumps(original_concept, ensure_ascii=False, indent=2)}

ORIGINAL LINKS:
{json.dumps(original_links, ensure_ascii=False, indent=2)}

HARD VALIDATION ISSUES:
{json.dumps(validation_issues, ensure_ascii=False, indent=2)}

CORE_TEXT:
{core_text}

Repair this candidate only. Preserve its scientific subject, relation, object,
relation strength, retention lane, and concept ID exactly. If that cannot be
done without unsupported evidence or an unsupported anchor, return
repairable=false.
""".strip()
