from __future__ import annotations

import json
from collections import Counter
from typing import Any

from pipeline_core.validation_issues import (
    IssueCode,
    ValidationReport,
)


MICRO_REEXTRACT_PROMPT_VERSION = (
    "dac-her-micro-reextract-v1-alpha4a2-fix5"
)


MICRO_REEXTRACT_SYSTEM_PROMPT = """
You re-extract one very small scientific source chunk into a
complete provenance-preserving KnowledgeGraphDraft.

The previous graph is diagnostic evidence only. Do not preserve
an invalid node or edge merely to resemble it.

Rules:

1. Extract facts only from CORE_TEXT and supplied ASSET_CONTEXT.
2. Return one complete KnowledgeGraphDraft, not a patch.
3. Correct every supplied structured validation issue.
3a. If a CLAIM_LIKE_ENTITY issue is present, do not keep the claim-like
    statement as an Entity merely to preserve its ID. If CORE_TEXT supports
    it as a scientific claim, place it in observation_claims[] or
    mechanism_claims[] as appropriate and rebuild only source-supported
    claim relations. Otherwise omit it.
4. SUPPORTS_CLAIM may start only from a Measurement,
   Experiment, or Calculation.
5. A scientific Entity may be an APPLIES_TO target, but may not
   directly be the source of SUPPORTS_CLAIM.
6. MODELED_BY must start from a CatalystModel.
7. Distinguish a physical Catalyst from a computational
   CatalystModel.
8. When both roles are explicit, connect:
   CatalystModel --MODEL_OF--> Catalyst.
9. Omit unsupported objects rather than returning isolated nodes.
10. Preserve paper_id, chunk_id, section, document metadata,
    page_ids, and asset_ids exactly.
11. Every edge must contain valid evidence_pointers within the
    supplied source scope.
""".strip()


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