from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping

from dac_her.bridge_schemas import BridgeChunkGraph
from dac_her.scientific_signatures import strong_anchor_context_issues


_CAUSAL_RELATIONS = {
    "MODULATES",
    "MEDIATES",
    "PROMOTES",
    "SUPPRESSES",
}
_CORRELATIONAL_RELATIONS = {
    "CORRELATES_WITH",
    "VARIES_WITH",
    "COMPETES_WITH",
    "COMPETES_FOR",
    "CONTRASTS_WITH",
}


def _normalized_source(value: str) -> str:
    value = value.casefold().replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _is_subspan(child: str | None, parent: str) -> bool:
    if child is None:
        return False
    return _normalized_source(child) in _normalized_source(parent)


def _strict_catalog(
    strict_nodes: Iterable[dict[str, Any] | str],
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    ids: set[str] = set()
    catalog: dict[str, dict[str, Any]] = {}
    for item in strict_nodes:
        if isinstance(item, Mapping):
            node_id = str(item.get("id", ""))
            if not node_id:
                continue
            ids.add(node_id)
            catalog[node_id] = dict(item)
        else:
            node_id = str(item)
            ids.add(node_id)
            catalog[node_id] = {"id": node_id}
    return ids, catalog


def bridge_validation_issues(
    result: BridgeChunkGraph,
    *,
    paper_id: str,
    chunk_id: str,
    document_id: str,
    document_role: str,
    page_ids: Iterable[int],
    asset_ids: Iterable[str],
    core_text: str,
    strict_nodes: Iterable[dict[str, Any] | str] | None = None,
    strict_node_ids: Iterable[str] | None = None,
    anchor_context_issues_fn: Callable[..., list[str]] = (
        strong_anchor_context_issues
    ),
) -> list[str]:
    """Return hard validation failures.

    This function verifies provenance, verbatim grounding, schema-level pattern
    support, and anchor compatibility. Softer semantic decisions such as
    relation-cue sufficiency and competition argument roles are handled by the
    deterministic bridge policy so one bad candidate can be rejected without
    failing the whole chunk.
    """
    issues: list[str] = []

    expected = {
        "paper_id": paper_id,
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_role": document_role,
    }
    for field, expected_value in expected.items():
        actual_value = getattr(result, field)
        if actual_value != expected_value:
            issues.append(
                f"{field} mismatch: expected {expected_value!r}, got {actual_value!r}."
            )

    allowed_pages = {int(value) for value in page_ids}
    allowed_assets = {str(value) for value in asset_ids}
    supplied_nodes: Iterable[dict[str, Any] | str]
    if strict_nodes is not None:
        supplied_nodes = strict_nodes
    elif strict_node_ids is not None:
        supplied_nodes = strict_node_ids
    else:
        supplied_nodes = []
    strict_ids, catalog = _strict_catalog(supplied_nodes)
    normalized_core = _normalized_source(core_text)

    if set(result.page_ids) != allowed_pages:
        issues.append(
            f"page_ids mismatch: expected {sorted(allowed_pages)!r}, "
            f"got {sorted(result.page_ids)!r}."
        )
    if set(result.asset_ids) != allowed_assets:
        issues.append(
            f"asset_ids mismatch: expected {sorted(allowed_assets)!r}, "
            f"got {sorted(result.asset_ids)!r}."
        )

    concept_by_id = {concept.id: concept for concept in result.concepts}
    for concept in result.concepts:
        normalized_phrase = _normalized_source(concept.source_phrase)
        if normalized_phrase not in normalized_core:
            issues.append(
                f"Concept {concept.id!r} source_phrase is not present in CORE_TEXT: "
                f"{concept.source_phrase!r}."
            )

        for phrase in concept.supporting_phrases:
            if _normalized_source(phrase) not in normalized_core:
                issues.append(
                    f"Concept {concept.id!r} supporting phrase is not present in "
                    f"CORE_TEXT: {phrase!r}."
                )

        for item in concept.comparison_items:
            if _normalized_source(item.source_phrase) not in normalized_core:
                issues.append(
                    f"Concept {concept.id!r} comparison item source phrase is not "
                    f"present in CORE_TEXT: {item.source_phrase!r}."
                )

        if concept.pattern_support_mode == "explicit_single_span":
            span = concept.supporting_phrases[0]
            evidence_fields = {
                "subject_evidence_phrase": concept.subject_evidence_phrase,
                "relation_evidence_phrase": concept.relation_evidence_phrase,
                "object_evidence_phrase": concept.object_evidence_phrase,
            }
            for field, phrase in evidence_fields.items():
                if not _is_subspan(phrase, span):
                    issues.append(
                        f"Concept {concept.id!r} {field} is not a verbatim "
                        f"substring of its explicit supporting span: {phrase!r}."
                    )

        relation = concept.pattern_relation
        strength = concept.relation_strength
        if relation in _CAUSAL_RELATIONS and strength != "causal_interpretive":
            issues.append(
                f"Concept {concept.id!r} uses causal relation {relation!r} "
                "without relation_strength='causal_interpretive'."
            )
        if relation in _CORRELATIONAL_RELATIONS and strength == "causal_interpretive":
            issues.append(
                f"Concept {concept.id!r} overstates non-causal relation "
                f"{relation!r} as causal_interpretive."
            )

    for link in result.links:
        if link.anchor_id not in strict_ids:
            issues.append(
                f"Bridge link uses unknown strict anchor: {link.anchor_id!r}."
            )
            continue

        concept = concept_by_id[link.concept_id]
        if (
            concept.pattern_support_mode == "derived_multi_span"
            and link.evidence_strength != "indirect"
        ):
            issues.append(
                f"Derived multi-span pattern {concept.id!r} must use "
                "evidence_strength='indirect'."
            )

        anchor = catalog.get(link.anchor_id, {})
        context_text = " ".join(
            filter(
                None,
                (
                    concept.label,
                    concept.source_phrase,
                    concept.pattern_subject or "",
                    concept.pattern_object or "",
                ),
            )
        )
        for context_issue in anchor_context_issues_fn(
            concept_text=context_text,
            anchor=anchor,
            pattern_relation=concept.pattern_relation,
            pattern_support_mode=concept.pattern_support_mode,
            pattern_subject=concept.pattern_subject,
            pattern_object=concept.pattern_object,
        ):
            issues.append(
                f"Bridge anchor context mismatch for {link.anchor_id!r} -> "
                f"{link.concept_id!r}: {context_issue}."
            )

        for pointer in link.evidence_pointers:
            if pointer.document_id != document_id:
                issues.append(
                    f"Pointer document_id mismatch on {link.anchor_id!r}: "
                    f"{pointer.document_id!r}."
                )
            if pointer.document_role != document_role:
                issues.append(
                    f"Pointer document_role mismatch on {link.anchor_id!r}: "
                    f"{pointer.document_role!r}."
                )
            if pointer.page_id is not None and pointer.page_id not in allowed_pages:
                issues.append(
                    f"Pointer page_id {pointer.page_id!r} is not supplied for chunk."
                )
            invalid_assets = set(pointer.asset_ids) - allowed_assets
            if invalid_assets:
                issues.append(
                    "Pointer contains assets not supplied for chunk: "
                    f"{sorted(invalid_assets)!r}."
                )

    return issues


def validate_bridge_chunk(result: BridgeChunkGraph, **kwargs: Any) -> None:
    issues = bridge_validation_issues(result, **kwargs)
    if issues:
        raise ValueError("Bridge graph validation failed:\n- " + "\n- ".join(issues))


def bind_bridge_validation(
    anchor_context_issues_fn: Callable[..., list[str]],
) -> tuple[Callable[..., list[str]], Callable[..., None]]:
    # Bind generic validation to one domain's anchor-context semantics.

    def issues(
        result: BridgeChunkGraph,
        **kwargs: Any,
    ) -> list[str]:
        return bridge_validation_issues(
            result,
            anchor_context_issues_fn=anchor_context_issues_fn,
            **kwargs,
        )

    def validate(
        result: BridgeChunkGraph,
        **kwargs: Any,
    ) -> None:
        found = issues(result, **kwargs)
        if found:
            raise ValueError(
                "Bridge graph validation failed:\n- "
                + "\n- ".join(found)
            )

    return issues, validate
