from __future__ import annotations

import json
from typing import Any

from dac_her.validation_issues import (
    ValidationReport,
)


MICRO_REEXTRACT_PROMPT_VERSION = (
    "dac-her-micro-reextract-v1"
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

Re-extract CORE_TEXT as one complete corrected
KnowledgeGraphDraft.
""".strip()