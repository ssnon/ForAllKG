"""Catalysis-mechanism-owned extraction user-prompt builders.

H1b-B transfers ownership of the effective extraction user-prompt envelope
away from DAC-HER without changing generated prompt text. Scientific broad
catalysis-mechanism semantics remain owned by
domains.catalysis_mechanism.prompts system/patch/micro prompts.

The initial implementations intentionally preserve the pre-H1b callable
behavior exactly. Future broad-domain changes must be reviewed as explicit
scientific-behavior changes rather than architecture refactors.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from pipeline_core.runtime.validation_issues import (
    IssueCode,
    ValidationReport,
)


def build_extraction_prompt(
    *,
    paper_id: str,
    chunk_id: str,
    document_id: str,
    document_role: str,
    section: str,
    page_ids: tuple[int, ...] | list[int],
    asset_ids: tuple[str, ...] | list[str],
    asset_context: str,
    vocabulary_context: str,
    left_context: str,
    core_text: str,
    right_context: str,
    validation_feedback: str | None = None,
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

SECTION:
{section}

PAGE_IDS:
{list(page_ids)}

ASSET_IDS:
{list(asset_ids)}

ASSET_CONTEXT:
{asset_context or 'No linked assets.'}

VOCABULARY_CONTEXT:
{vocabulary_context or 'No registry context supplied.'}

LEFT_CONTEXT:
{left_context}

CORE_TEXT:
{core_text}

RIGHT_CONTEXT:
{right_context}
""".strip()

    if validation_feedback:
        prompt += f"""

PREVIOUS VALIDATION ERROR:
{validation_feedback}

The previous graph failed validation. Return a complete
new KnowledgeGraph, not a partial patch.

STRICT REPAIR RULES:

1. HAS_MEASUREMENT must be Experiment/Calculation -> Measurement.
2. SUPPORTS_CLAIM must start from Measurement, Experiment, or Calculation.
3. ObservationClaim -> INTERPRETED_AS -> MechanismClaim.
4. Every claim requires APPLIES_TO.
5. Every node must participate in at least one edge.
6. Rebuild the entire edge list, not only the first error.
7. Preserve PAPER_ID, CHUNK_ID, DOCUMENT_ID, DOCUMENT_ROLE,
   PAGE_IDS, and ASSET_IDS exactly.
8. Every edge requires evidence_pointers. Each pointer must use
   the supplied DOCUMENT_ID/ROLE and only supplied PAGE_IDS/ASSET_IDS.
9. Use asset_ids=[] for text-only evidence. Marker alt text alone
   is never sufficient evidence.
10. Re-check scalar measurements: one subject, one result, one condition
    set, one MEASURED_FOR edge, and matching MeasurementGroups.
11. Use only valid registry IDs from VOCABULARY_CONTEXT, or an explicit
    unregistered_<slug> when no method/metric fits.

Do not invent evidence, experiments, measurements, assets, pages,
or relationships unsupported by CORE_TEXT and ASSET_CONTEXT.
""".strip()

    return prompt


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


def _common_undefined_endpoint_hints(
    report: ValidationReport,
) -> str:
    """Aggregate undefined endpoints by node ID, independent of direction."""
    endpoint_counts: Counter[str] = Counter()
    incident_edges: dict[str, list[str]] = {}

    for item in report.issues:
        endpoint_id: str | None = None
        if (
            item.code == IssueCode.UNDEFINED_EDGE_SOURCE
            and item.source_id
        ):
            endpoint_id = item.source_id
        elif (
            item.code == IssueCode.UNDEFINED_EDGE_TARGET
            and item.target_id
        ):
            endpoint_id = item.target_id

        if endpoint_id is None:
            continue

        endpoint_counts[endpoint_id] += 1
        incident_edges.setdefault(endpoint_id, []).append(
            f"edge_index={item.edge_index}: "
            f"{item.source_id!r} --{item.relation}--> {item.target_id!r}"
        )

    hints: list[str] = []
    for endpoint_id, count in endpoint_counts.most_common():
        if count < 2:
            continue
        edges = "\n    ".join(incident_edges.get(endpoint_id, []))
        hints.append(
            f"- Missing node ID {endpoint_id!r} participates in "
            f"{count} invalid incident edge(s):\n    {edges}\n"
            "  Treat these as one missing-object problem even when the same "
            "ID appears as a source in some edges and a target in others. "
            "If CORE_TEXT explicitly supports that scientific object, emit "
            "it once in the correct top-level collection and connect only "
            "source-supported relations. Otherwise omit the unsupported "
            "dangling relations."
        )

    if not hints:
        return "No common undefined-endpoint cluster was detected."

    return (
        "COMMON_UNDEFINED_ENDPOINT_RECOVERY_HINTS:\n"
        + "\n".join(hints)
        + "\nDo not preserve dangling edges merely to resemble the previous "
        "graph. Do not invent a node solely because an ID appeared in an "
        "invalid edge."
    )


def _coupled_claim_subgraph_hints(
    report: ValidationReport,
) -> str:
    claim_codes = {
        IssueCode.CLAIM_MISSING_APPLICATION_TARGET,
        IssueCode.OBSERVATION_MISSING_SUPPORT,
        IssueCode.MECHANISM_MISSING_SUPPORT,
        IssueCode.CLAIM_LIKE_ENTITY,
    }
    claim_relations = {
        "SUPPORTS_CLAIM",
        "INTERPRETED_AS",
        "APPLIES_TO",
    }

    claim_issues = [
        item
        for item in report.issues
        if item.code in claim_codes
    ]
    relation_issues = [
        item
        for item in report.issues
        if (
            item.code
            in {
                IssueCode.RELATION_SOURCE_TYPE_MISMATCH,
                IssueCode.RELATION_TARGET_TYPE_MISMATCH,
            }
            and item.relation in claim_relations
        )
    ]

    if not claim_issues or not relation_issues:
        return "No coupled claim-subgraph residual was detected."

    lines = [
        "COUPLED_CLAIM_SUBGRAPH_RECOVERY_HINTS:",
        (
            "- Multiple claim and claim-relation validation errors appear "
            "coupled. Rebuild the affected claim subgraph as one "
            "source-grounded unit instead of preserving invalid edges."
        ),
        (
            "- SUPPORTS_CLAIM must originate only from a validator-allowed "
            "evidence-producing object and point to a supported claim."
        ),
        (
            "- INTERPRETED_AS must follow the validator-required direction "
            "between an observation and a mechanism claim; do not reverse it."
        ),
        (
            "- APPLIES_TO must point from the claim to an explicit scientific "
            "subject supported by CORE_TEXT. Never guess a target merely to "
            "satisfy validation."
        ),
        (
            "- If CORE_TEXT does not support a complete valid claim subgraph, "
            "omit unsupported claims/relations rather than inventing evidence."
        ),
        "- Current coupled issues:",
    ]

    for item in [*claim_issues, *relation_issues]:
        lines.append(
            f"  * {item.code.value}: node={item.node_id!r}, "
            f"edge_index={item.edge_index}, relation={item.relation!r}, "
            f"source={item.source_id!r}, target={item.target_id!r}"
        )

    return "\n".join(lines)


def build_domain_gate_recovery_prompt(
    *,
    paper_id: str,
    chunk_id: str,
    document_id: str,
    document_role: str,
    section: str,
    page_ids: list[int] | tuple[int, ...],
    asset_ids: list[str] | tuple[str, ...],
    core_text: str,
    left_context: str,
    right_context: str,
    asset_context: str,
    rejected_graph_payload: dict[str, Any],
    domain_error: str,
) -> str:
    """Targeted full-draft repair after reserved-collection domain rejection."""
    return f"""
PAPER_ID:
{paper_id}

CHUNK_ID:
{chunk_id}

DOCUMENT_ID:
{document_id}

DOCUMENT_ROLE:
{document_role}

SECTION:
{section}

PAGE_IDS:
{list(page_ids)}

ASSET_IDS:
{list(asset_ids)}

LEFT_CONTEXT:
{left_context}

CORE_TEXT:
{core_text}

RIGHT_CONTEXT:
{right_context}

ASSET_CONTEXT:
{asset_context or 'No linked assets.'}

DOMAIN_GATE_ERROR:
{domain_error}

PREVIOUS_DOMAIN_REJECTED_GRAPH_JSON:
{json.dumps(
    rejected_graph_payload,
    ensure_ascii=False,
    indent=2,
)}

The previous object parsed as a KnowledgeGraphDraft but was rejected by the
active scientific-domain gate. Return one complete corrected draft.

Repair the collection/domain placement error explicitly identified above.
Reserved structured objects such as Experiment, Calculation, Measurement,
MeasurementGroup, ObservationClaim, and MechanismClaim belong in their
dedicated top-level collections rather than entities[].

Preserve only source-grounded facts. Do not invent new science to make the
graph connected or valid. If an invalidly placed object cannot be represented
in the correct collection using source-supported fields, omit that object and
any dependent unsupported edges instead of fabricating fields.

Every emitted edge endpoint must exist exactly once in the corrected draft.
Preserve authoritative paper/chunk/document/page/asset metadata.
""".strip()


def build_micro_reextract_prompt(
    *,
    paper_id: str,
    chunk_id: str,
    document_id: str,
    document_role: str,
    section: str,
    page_ids: list[int] | tuple[int, ...],
    asset_ids: list[str] | tuple[str, ...],
    core_text: str,
    left_context: str,
    right_context: str,
    asset_context: str,
    graph_payload: dict[str, Any],
    report: ValidationReport,
) -> str:
    return f"""
PAPER_ID:
{paper_id}

CHUNK_ID:
{chunk_id}

DOCUMENT_ID:
{document_id}

DOCUMENT_ROLE:
{document_role}

SECTION:
{section}

PAGE_IDS:
{list(page_ids)}

ASSET_IDS:
{list(asset_ids)}

LEFT_CONTEXT:
{left_context}

CORE_TEXT:
{core_text}

RIGHT_CONTEXT:
{right_context}

ASSET_CONTEXT:
{asset_context or 'No linked assets.'}

PREVIOUS_INVALID_GRAPH_JSON:
{json.dumps(
    graph_payload,
    ensure_ascii=False,
    indent=2,
)}

STRUCTURED_VALIDATION_ISSUES_JSON:
{json.dumps(
    [
        issue.model_dump(mode="json")
        for issue in report.issues
    ],
    ensure_ascii=False,
    indent=2,
)}

{_common_undefined_endpoint_hints(report)}

{_coupled_claim_subgraph_hints(report)}

Re-extract CORE_TEXT as one complete corrected
KnowledgeGraphDraft.
""".strip()
