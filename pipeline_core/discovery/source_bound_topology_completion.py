from __future__ import annotations

import re
from typing import Any

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltyClaimDecompositionDraft,
)


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[‐‑‒–—−_/]+", " ", text)
    text = re.sub(r"[^\w\s+*.]", " ", text)
    return " ".join(text.split())


def complete_explicit_source_bound_topology_shadow(
    draft: NoveltyClaimDecompositionDraft,
) -> tuple[NoveltyClaimDecompositionDraft, list[dict[str, Any]]]:
    """Complete only empty composite topology from exact source containment.

    No scientific relation is inferred. A non-composite claim is eligible only
    when its exact proposition_basis is textually contained in one of the
    composite claim's explicit higher_order_relation_basis spans after surface
    normalization. Existing explicit topology is preserved unchanged.
    """
    rows = list(draft.claims)
    output = []
    records: list[dict[str, Any]] = []

    for row in rows:
        if row.kind != "composite":
            output.append(row)
            continue

        existing = [
            str(x).strip()
            for x in row.higher_order_component_local_ids
            if str(x).strip()
        ]
        if existing:
            output.append(row)
            records.append(
                {
                    "composite_local_id": row.local_id,
                    "status": "EXPLICIT_TOPOLOGY_PRESERVED",
                    "component_local_ids": existing,
                    "source_bound_recovery_performed": False,
                    "production_authority": False,
                }
            )
            continue

        bases = [
            _norm(x)
            for x in row.higher_order_relation_basis
            if _norm(x)
        ]
        recovered: list[str] = []
        evidence: list[dict[str, str]] = []

        if bases:
            for candidate in rows:
                if candidate.local_id == row.local_id:
                    continue
                if candidate.kind == "composite":
                    continue

                proposition = _norm(
                    candidate.semantic_fidelity_binding.proposition_basis
                )
                if not proposition:
                    continue
                if not any(proposition in basis for basis in bases):
                    continue

                recovered.append(candidate.local_id)
                evidence.append(
                    {
                        "component_local_id": candidate.local_id,
                        "proposition_basis": (
                            candidate.semantic_fidelity_binding.proposition_basis
                        ),
                    }
                )

        if recovered:
            updated = row.model_copy(
                update={
                    "higher_order_component_local_ids": recovered,
                }
            )
            output.append(updated)
            records.append(
                {
                    "composite_local_id": row.local_id,
                    "status": "SOURCE_BOUND_TOPOLOGY_COMPLETED_SHADOW",
                    "component_local_ids": recovered,
                    "evidence": evidence,
                    "source_bound_recovery_performed": True,
                    "production_authority": False,
                }
            )
        else:
            output.append(row)
            records.append(
                {
                    "composite_local_id": row.local_id,
                    "status": "NO_SOURCE_BOUND_COMPONENT_TOPOLOGY",
                    "component_local_ids": [],
                    "source_bound_recovery_performed": False,
                    "production_authority": False,
                }
            )

    payload = draft.model_dump(mode="python")
    payload["claims"] = [
        row.model_dump(mode="python")
        for row in output
    ]
    return (
        NoveltyClaimDecompositionDraft.model_validate(payload),
        records,
    )
