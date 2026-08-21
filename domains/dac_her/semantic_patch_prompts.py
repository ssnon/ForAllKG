from __future__ import annotations

import json
import re
from typing import Any

from pipeline_core.runtime.validation_issues import ValidationReport


PATCH_PROMPT_VERSION = "dac-her-semantic-patch-v1.4-capability-routing"

PATCH_SYSTEM_PROMPT = """
You repair a provenance-preserving scientific knowledge-graph draft.
Return only a KnowledgeGraphPatch. Never return a complete graph.

The patch must be minimal and source-grounded:
1. Modify only objects directly implicated by the supplied issue IDs.
2. Do not add scientific facts, entities, measurements, or claims.
3. An add_edge operation is permitted only when CORE_TEXT explicitly states
   the relationship and both endpoint nodes already exist in the draft.
4. Preserve supplied paper_id, chunk_id, document metadata, page IDs, and
   asset IDs. Every added edge must use only supplied provenance locators.
5. Do not guess a claim target, catalyst/model identity, or edge endpoint.
6. When evidence is insufficient, put the issue ID in unresolved_issue_ids.
7. Do not remove an edge merely to make validation pass unless the source
   explicitly shows that the edge is erroneous.
8. Do not rename a node unless the intended existing ID correspondence is
   explicit in CORE_TEXT and the draft.
9. Keep operations independent and concise.
10. Use replace_edge when correcting the relation, direction,
    source, and target requires replacing the whole edge.
11. Do not use multiple endpoint edits when one atomic
    replace_edge operation expresses the correction.
12. Replacing Entity --SUPPORTS_CLAIM--> Claim with
    Claim --APPLIES_TO--> Entity is permitted only when
    CORE_TEXT explicitly supports the application target.

PATCH OPERATION SHAPE:
Every operation uses the same flat object schema. Populate the fields needed
for the selected op and set every unrelated operation-specific field to null.

- add_edge: edge is non-null; all other operation-specific fields are null.
- remove_edge: edge_index, expected_source, expected_relation, and
  expected_target are non-null; other operation-specific fields are null.
- change_entity_type: node_id, old_type, and new_type are non-null; other
  operation-specific fields are null.
- replace_edge: edge, edge_index, expected_source,
  expected_relation, and expected_target are non-null.
  The expected_* fields describe the complete current edge.
  edge contains the complete replacement KGEdge.
  Every other operation-specific field is null.
- replace_edge_endpoint: edge_index, expected_source, expected_relation,
  expected_target, endpoint, old_id, and new_id are non-null. The expected_*
  fields must describe the complete current edge before modification; other
  operation-specific fields are null.
- rename_node_id: old_id and new_id are non-null; all other
  operation-specific fields are null.

For CLAIM_MISSING_APPLICATION_TARGET:

- APPLIES_TO should identify the primary scientific subject
  of the claim, not merely any entity mentioned in the sentence.
- For catalyst performance claims, prefer the Catalyst or
  CatalystModel being evaluated as an APPLIES_TO target.
- If the claim is explicitly scoped to a reaction, the Reaction
  may be added as an additional APPLIES_TO target, but it should
  not replace an explicitly stated catalyst/model subject.
- For structural characterization claims, prefer the Catalyst,
  CatalystModel, CoordinationMotif, or other structure-bearing
  entity that was characterized.
- Add only targets explicitly supported by CORE_TEXT.
""".strip()


_OPERATION_SHAPE_GUIDANCE = {
    "add_edge": (
        "NON-NULL: edge. The edge must be one complete source-grounded KGEdge. "
        "All other operation-specific fields must be null."
    ),
    "remove_edge": (
        "NON-NULL: edge_index, expected_source, expected_relation, "
        "expected_target. All other operation-specific fields must be null."
    ),
    "replace_edge": (
        "NON-NULL: edge, edge_index, expected_source, expected_relation, "
        "expected_target. expected_* describes the complete CURRENT edge; "
        "edge is the complete replacement KGEdge. All unrelated "
        "operation-specific fields must be null."
    ),
    "change_entity_type": (
        "NON-NULL: node_id, old_type, new_type. All other "
        "operation-specific fields must be null."
    ),
    "replace_edge_endpoint": (
        "NON-NULL: edge_index, expected_source, expected_relation, "
        "expected_target, endpoint, old_id, new_id. All unrelated "
        "operation-specific fields must be null."
    ),
    "rename_node_id": (
        "NON-NULL: old_id, new_id. All other operation-specific fields "
        "must be null."
    ),
}


def build_patch_rejection_feedback(error: Exception) -> str:
    """Turn provider/Pydantic operation-shape failures into actionable retry text."""
    message = str(error)
    match = re.search(
        r"Operation '([^']+)' requires non-null fields: \[([^\]]+)\]",
        message,
    )
    if match:
        op = match.group(1)
        missing = match.group(2)
        guidance = _OPERATION_SHAPE_GUIDANCE.get(
            op,
            "Use the exact PATCH OPERATION SHAPE rules.",
        )
        return (
            "The previous patch failed operation-shape validation. "
            f"Selected op: {op!r}. Missing required field(s): [{missing}].\n"
            f"Required shape for {op}: {guidance}\n"
            "Before returning, self-check every operation: every field required "
            "by the selected op is non-null, and every unrelated "
            "operation-specific field is null. Do not change scientific "
            "content merely to satisfy the schema."
        )

    return f"{type(error).__name__}: {error}"


def build_semantic_patch_prompt(
    *,
    paper_id: str,
    chunk_id: str,
    document_id: str,
    document_role: str,
    page_ids: list[int] | tuple[int, ...],
    asset_ids: list[str] | tuple[str, ...],
    core_text: str,
    asset_context: str,
    graph_payload: dict[str, Any],
    report: ValidationReport,
    previous_patch_feedback: str | None = None,
) -> str:
    prompt = f"""
PAPER_ID:
{paper_id}

CHUNK_ID:
{chunk_id}

DOCUMENT_ID:
{document_id}

DOCUMENT_ROLE:
{document_role}

PAGE_IDS:
{list(page_ids)}

ASSET_IDS:
{list(asset_ids)}

CORE_TEXT:
{core_text}

ASSET_CONTEXT:
{asset_context or 'No linked assets.'}

CURRENT_GRAPH_DRAFT_JSON:
{json.dumps(graph_payload, ensure_ascii=False, indent=2)}

STRUCTURED_VALIDATION_ISSUES_JSON:
{json.dumps([item.model_dump(mode='json') for item in report.issues], ensure_ascii=False, indent=2)}

PATCH_OUTPUT_SELF_CHECK:
For every operation, verify the selected op against PATCH OPERATION SHAPE before
returning JSON. Required fields for that op must be non-null. Every unrelated
operation-specific field must be null. A replace_edge must contain both the
complete current-edge guards (edge_index + expected_source/relation/target) and
one complete replacement KGEdge.
""".strip()

    if previous_patch_feedback:
        prompt += f"""

PREVIOUS_PATCH_REJECTION:
{previous_patch_feedback}

Return a new minimal patch. Do not repeat the rejected operation unless the
rejection can be resolved with explicit CORE_TEXT evidence.
""".rstrip()

    return prompt
