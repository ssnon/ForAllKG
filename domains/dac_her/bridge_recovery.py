from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from pipeline_core.corpus.bridge.bridge_draft_schema import (
    BridgeCandidateRepair,
    BridgeChunkDraft,
)
from pipeline_core.corpus.bridge.bridge_schemas import (
    BridgeChunkGraph,
    BridgeConcept,
    BridgeLink,
)
from pipeline_core.corpus.bridge.bridge_source_reconciliation import (
    BRIDGE_SOURCE_RECONCILIATION_VERSION,
    reconcile_concept_payload,
)
from domains.dac_her.bridge_validation import (
    bridge_validation_issues,
    validate_bridge_chunk,
)


BRIDGE_RECOVERY_VERSION = "dac-her-bridge-recovery-v2.4.1"

_FRONTIER_ANCHOR_RELATION_BY_TYPE: dict[str, str] = {
    "Phenomenon": "INVOLVES_PHENOMENON",
    "InterfacialEffect": "DESCRIBES_INTERFACE",
    "DynamicState": "EXHIBITS_DYNAMIC_STATE",
    "DesignPrinciple": "SUGGESTS_DESIGN_PRINCIPLE",
    "FailureMode": "HAS_FAILURE_MODE",
    "MechanisticAnalogy": "USES_MECHANISTIC_ANALOGY",
    "OpenQuestion": "RAISES_OPEN_QUESTION",
}


CandidateRepairCallback = Callable[
    [
        dict[str, Any],
        list[dict[str, Any]],
        list[str],
        int,
    ],
    BridgeCandidateRepair | None,
]


@dataclass(frozen=True)
class BridgeRecoveryResult:
    graph: BridgeChunkGraph
    normalization_operations: list[dict[str, Any]]
    quarantined_items: list[dict[str, Any]]
    candidate_repair_attempts: int
    repaired_candidate_count: int
    candidate_repairs: list[dict[str, Any]]
    generated_concept_count: int
    generated_link_count: int

    @property
    def quarantined_candidate_count(self) -> int:
        return sum(
            item.get("kind") == "concept"
            for item in self.quarantined_items
        )

    @property
    def quarantined_link_count(self) -> int:
        return sum(
            item.get("kind") in {
                "link",
                "orphan_link",
            }
            for item in self.quarantined_items
        )


class BridgeRecoveryError(RuntimeError):
    """Internal recovery failure that should remain a technical failure."""


def _source_metadata(
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "paper_id": str(source_payload["paper_id"]),
        "chunk_id": str(source_payload["chunk_id"]),
        "section": str(source_payload["section"]),
        "document_id": str(source_payload["document_id"]),
        "document_role": str(source_payload["document_role"]),
        "page_ids": [
            int(value)
            for value in source_payload.get("page_ids", [])
        ],
        "asset_ids": [
            str(value)
            for value in source_payload.get("asset_ids", [])
        ],
    }


def _record_change(
    operations: list[dict[str, Any]],
    *,
    operation: str,
    field: str,
    old_value: Any,
    new_value: Any,
    reason: str,
    concept_id: str | None = None,
    link_index: int | None = None,
) -> None:
    if old_value == new_value:
        return

    operations.append(
        {
            "operation": operation,
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
            "concept_id": concept_id,
            "link_index": link_index,
        }
    )


def _normalize_metadata(
    payload: dict[str, Any],
    *,
    expected: dict[str, Any],
    operations: list[dict[str, Any]],
) -> None:
    for field in (
        "paper_id",
        "chunk_id",
        "section",
        "document_id",
        "document_role",
        "page_ids",
        "asset_ids",
    ):
        old_value = payload.get(field)
        new_value = expected[field]
        if old_value != new_value:
            payload[field] = new_value
            _record_change(
                operations,
                operation="restore_source_metadata",
                field=field,
                old_value=old_value,
                new_value=new_value,
                reason=(
                    "Bridge chunk metadata is supplied by the frozen source "
                    "snapshot and is not a model decision."
                ),
            )


def _looks_like_complete_pattern(
    concept: dict[str, Any],
) -> bool:
    return all(
        concept.get(field) is not None
        for field in (
            "pattern_subject",
            "pattern_relation",
            "pattern_object",
            "relation_strength",
            "pattern_support_mode",
        )
    ) and bool(concept.get("supporting_phrases"))


def _normalize_concept_and_links(
    concept: dict[str, Any],
    links: list[dict[str, Any]],
    *,
    core_text: str,
    operations: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    concept = dict(concept)
    links = [dict(item) for item in links]
    concept_id = str(concept.get("id", ""))

    reconciled, phrase_operations = (
        reconcile_concept_payload(
            concept,
            core_text=core_text,
        )
    )
    concept = reconciled

    for item in phrase_operations:
        operations.append(
            {
                "operation": "reconcile_source_span",
                "concept_id": concept_id,
                "link_index": None,
                "reason": (
                    "Unique match under conservative Markdown/LaTeX "
                    "normalization; restored exact CORE_TEXT substring."
                ),
                **item,
            }
        )

    if (
        concept.get("retention_lane") == "accepted_pattern"
        and concept.get("concept_type") != "RelationPattern"
        and _looks_like_complete_pattern(concept)
    ):
        old_type = concept.get("concept_type")
        concept["concept_type"] = "RelationPattern"
        _record_change(
            operations,
            operation="normalize_lane_type",
            field="concept_type",
            old_value=old_type,
            new_value="RelationPattern",
            reason=(
                "accepted_pattern semantics deterministically require "
                "concept_type=RelationPattern."
            ),
            concept_id=concept_id,
        )

    if concept.get("retention_lane") == "accepted_pattern":
        for index, link in enumerate(links):
            if link.get("concept_id") != concept_id:
                continue
            if link.get("relation") != "EXPRESSES_PATTERN":
                old_relation = link.get("relation")
                link["relation"] = "EXPRESSES_PATTERN"
                _record_change(
                    operations,
                    operation="normalize_pattern_anchor_relation",
                    field="relation",
                    old_value=old_relation,
                    new_value="EXPRESSES_PATTERN",
                    reason=(
                        "accepted_pattern grounding links have a fixed "
                        "representation relation."
                    ),
                    concept_id=concept_id,
                    link_index=index,
                )
    elif (
        concept.get("retention_lane")
        == "paper_local_frontier"
    ):
        concept_type = str(
            concept.get("concept_type", "")
        )

        expected_relation = (
            _FRONTIER_ANCHOR_RELATION_BY_TYPE.get(
                concept_type
            )
        )

        # Only normalize concept types whose Bridge schema gives an
        # unambiguous one-to-one grounding relation.
        if expected_relation is not None:
            for index, link in enumerate(links):
                if link.get("concept_id") != concept_id:
                    continue

                old_relation = link.get("relation")

                if old_relation == expected_relation:
                    continue

                link["relation"] = expected_relation

                _record_change(
                    operations,
                    operation=(
                        "normalize_frontier_anchor_relation"
                    ),
                    field="relation",
                    old_value=old_relation,
                    new_value=expected_relation,
                    reason=(
                        "paper_local_frontier concept type "
                        f"{concept_type!r} has an unambiguous "
                        "schema-defined grounding relation."
                    ),
                    concept_id=concept_id,
                    link_index=index,
                )

    return concept, links


def _mini_validation_kwargs(
    *,
    metadata: dict[str, Any],
    core_text: str,
    strict_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "paper_id": metadata["paper_id"],
        "chunk_id": metadata["chunk_id"],
        "document_id": metadata["document_id"],
        "document_role": metadata["document_role"],
        "page_ids": metadata["page_ids"],
        "asset_ids": metadata["asset_ids"],
        "core_text": core_text,
        "strict_nodes": strict_nodes,
    }


def _validate_one_concept(
    concept_payload: dict[str, Any],
    link_payloads: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    core_text: str,
    strict_nodes: list[dict[str, Any]],
    validation_issues_fn: Callable[..., list[str]] = bridge_validation_issues,
) -> tuple[
    BridgeConcept | None,
    list[BridgeLink],
    list[dict[str, Any]],
    list[str],
]:
    """
    Validate one candidate independently of its siblings.

    Invalid extra links are localized and may be quarantined while a candidate
    remains usable if at least one grounding link is independently valid.
    """
    try:
        concept = BridgeConcept.model_validate(
            concept_payload
        )
    except (ValidationError, ValueError) as error:
        return None, [], [], [str(error)]

    if not link_payloads:
        return (
            concept,
            [],
            [],
            [
                "Every bridge concept must be linked to at least one "
                "strict-graph anchor."
            ],
        )

    valid_links: list[BridgeLink] = []
    invalid_links: list[dict[str, Any]] = []
    candidate_issues: list[str] = []

    for index, link_payload in enumerate(link_payloads):
        try:
            link = BridgeLink.model_validate(
                link_payload
            )
            mini = BridgeChunkGraph(
                **metadata,
                concepts=[concept],
                links=[link],
            )
            issues = validation_issues_fn(
                mini,
                **_mini_validation_kwargs(
                    metadata=metadata,
                    core_text=core_text,
                    strict_nodes=strict_nodes,
                ),
            )
        except (ValidationError, ValueError) as error:
            issues = [str(error)]
            link = None

        if issues:
            invalid_links.append(
                {
                    "kind": "link",
                    "concept_id": concept.id,
                    "link_index": index,
                    "link": link_payload,
                    "issues": issues,
                }
            )
            for issue in issues:
                if issue not in candidate_issues:
                    candidate_issues.append(issue)
            continue

        assert link is not None
        valid_links.append(link)

    if valid_links:
        # A source/shape issue would invalidate every mini graph. Therefore any
        # surviving link proves the concept itself passed the hard validators.
        return concept, valid_links, invalid_links, []

    return concept, [], invalid_links, candidate_issues


def _repair_preserves_science(
    original: dict[str, Any],
    repaired: dict[str, Any],
) -> tuple[bool, list[str]]:
    issues: list[str] = []

    fixed_fields = (
        "id",
        "retention_lane",
        "pattern_subject",
        "pattern_relation",
        "pattern_object",
        "relation_strength",
        "evidence_scope",
    )

    for field in fixed_fields:
        if repaired.get(field) != original.get(field):
            issues.append(
                f"Local repair changed forbidden scientific field {field!r}: "
                f"{original.get(field)!r} -> {repaired.get(field)!r}."
            )

    # Frontier concept type is itself scientific semantics and must be stable.
    if (
        original.get("retention_lane") == "paper_local_frontier"
        and repaired.get("concept_type") != original.get("concept_type")
    ):
        issues.append(
            "Local repair changed paper_local_frontier concept_type."
        )

    return not issues, issues


def recover_bridge_draft(
    draft: BridgeChunkDraft,
    *,
    source_payload: dict[str, Any],
    strict_nodes: list[dict[str, Any]],
    repair_callback: CandidateRepairCallback | None = None,
    max_candidate_repairs_per_chunk: int = 3,
    validation_issues_fn: Callable[..., list[str]] = bridge_validation_issues,
    validate_chunk_fn: Callable[..., None] = validate_bridge_chunk,
) -> BridgeRecoveryResult:
    """
    Convert a recoverable Bridge draft into a strict BridgeChunkGraph.

    Scientific safety rules:
    - source reconciliation is unique-match only;
    - accepted-pattern type/link normalization changes representation, not
      scientific semantics;
    - local LLM repair may change grounding only, never pattern semantics;
    - unresolved candidates are quarantined individually;
    - the returned raw graph is always strict-valid, possibly empty.
    """
    if max_candidate_repairs_per_chunk < 0:
        raise ValueError(
            "max_candidate_repairs_per_chunk must be non-negative."
        )

    payload = draft.model_dump(mode="json")
    metadata = _source_metadata(source_payload)
    core_text = str(source_payload["core_text"])

    operations: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []

    _normalize_metadata(
        payload,
        expected=metadata,
        operations=operations,
    )

    generated_concepts = [
        dict(item)
        for item in payload.get("concepts", [])
    ]
    generated_links = [
        dict(item)
        for item in payload.get("links", [])
    ]

    generated_concept_count = len(generated_concepts)
    generated_link_count = len(generated_links)

    duplicate_ids = {
        concept_id
        for concept_id, count in Counter(
            str(item.get("id", ""))
            for item in generated_concepts
        ).items()
        if count > 1
    }

    links_by_concept: dict[str, list[dict[str, Any]]] = {}
    known_ids = {
        str(item.get("id", ""))
        for item in generated_concepts
    }

    seen_link_signatures: set[tuple[str, str, str]] = set()
    for index, link in enumerate(generated_links):
        concept_id = str(link.get("concept_id", ""))
        signature = (
            str(link.get("anchor_id", "")),
            str(link.get("relation", "")),
            concept_id,
        )

        if signature in seen_link_signatures:
            operations.append(
                {
                    "operation": "drop_exact_duplicate_link",
                    "field": "links",
                    "old_value": link,
                    "new_value": None,
                    "reason": (
                        "Exact duplicate grounding link carries no additional "
                        "scientific information."
                    ),
                    "concept_id": concept_id,
                    "link_index": index,
                }
            )
            continue
        seen_link_signatures.add(signature)

        if concept_id not in known_ids:
            quarantined.append(
                {
                    "kind": "orphan_link",
                    "concept_id": concept_id,
                    "link_index": index,
                    "link": link,
                    "issues": [
                        "Bridge link targets an unknown concept ID."
                    ],
                    "repair_attempted": False,
                }
            )
            continue

        links_by_concept.setdefault(
            concept_id,
            [],
        ).append(link)

    accepted_concepts: list[BridgeConcept] = []
    accepted_links: list[BridgeLink] = []
    candidate_repair_attempts = 0
    repaired_candidate_count = 0
    candidate_repairs: list[dict[str, Any]] = []

    for original_concept in generated_concepts:
        concept_id = str(
            original_concept.get("id", "")
        )
        original_links = [
            dict(item)
            for item in links_by_concept.get(
                concept_id,
                [],
            )
        ]

        if concept_id in duplicate_ids:
            quarantined.append(
                {
                    "kind": "concept",
                    "concept_id": concept_id,
                    "concept": original_concept,
                    "links": original_links,
                    "issues": [
                        "Duplicate Bridge concept ID makes grounding links "
                        "ambiguous within the chunk."
                    ],
                    "repair_attempted": False,
                }
            )
            continue

        concept_payload, link_payloads = (
            _normalize_concept_and_links(
                original_concept,
                original_links,
                core_text=core_text,
                operations=operations,
            )
        )

        (
            concept_model,
            valid_links,
            invalid_links,
            issues,
        ) = _validate_one_concept(
            concept_payload,
            link_payloads,
            metadata=metadata,
            core_text=core_text,
            strict_nodes=strict_nodes,
            validation_issues_fn=validation_issues_fn,
        )

        if concept_model is not None and valid_links:
            accepted_concepts.append(concept_model)
            accepted_links.extend(valid_links)
            quarantined.extend(
                {
                    **item,
                    "repair_attempted": False,
                    "reason": (
                        "Candidate remained valid through at least one "
                        "grounding link; only the invalid extra link was "
                        "quarantined."
                    ),
                }
                for item in invalid_links
            )
            continue

        repaired = None
        repair_errors: list[str] = []
        repair_record: dict[str, Any] | None = None

        if (
            repair_callback is not None
            and candidate_repair_attempts
            < max_candidate_repairs_per_chunk
        ):
            repair_index = candidate_repair_attempts
            candidate_repair_attempts += 1
            try:
                repaired = repair_callback(
                    concept_payload,
                    link_payloads,
                    issues,
                    repair_index,
                )
            except Exception as error:  # local failure must not kill the chunk
                repair_errors.append(
                    f"Candidate repair call failed: {type(error).__name__}: "
                    f"{error}"
                )

            repair_record = {
                "concept_id": concept_id,
                "repair_index": repair_index,
                "issues_before": list(issues),
                "response": (
                    repaired.model_dump(mode="json")
                    if repaired is not None
                    else None
                ),
                "accepted": False,
                "issues_after": [],
            }

        if (
            repaired is not None
            and repaired.repairable
            and repaired.concept is not None
        ):
            repaired_concept = (
                repaired.concept.model_dump(
                    mode="json"
                )
            )
            repaired_links = [
                item.model_dump(mode="json")
                for item in repaired.links
            ]

            preserves, preservation_issues = (
                _repair_preserves_science(
                    concept_payload,
                    repaired_concept,
                )
            )

            if preserves:
                (
                    repaired_concept,
                    repaired_links,
                ) = _normalize_concept_and_links(
                    repaired_concept,
                    repaired_links,
                    core_text=core_text,
                    operations=operations,
                )

                (
                    repaired_model,
                    repaired_valid_links,
                    repaired_invalid_links,
                    repaired_issues,
                ) = _validate_one_concept(
                    repaired_concept,
                    repaired_links,
                    metadata=metadata,
                    core_text=core_text,
                    strict_nodes=strict_nodes,
                    validation_issues_fn=validation_issues_fn,
                )

                if (
                    repaired_model is not None
                    and repaired_valid_links
                ):
                    accepted_concepts.append(
                        repaired_model
                    )
                    accepted_links.extend(
                        repaired_valid_links
                    )
                    repaired_candidate_count += 1
                    quarantined.extend(
                        {
                            **item,
                            "repair_attempted": True,
                            "reason": (
                                "Local repair recovered the concept, but this "
                                "extra grounding link remained invalid."
                            ),
                        }
                        for item in repaired_invalid_links
                    )
                    if repair_record is not None:
                        repair_record["accepted"] = True
                        repair_record["issues_after"] = []
                        candidate_repairs.append(repair_record)
                    continue

                repair_errors.extend(
                    repaired_issues
                )
                if repair_record is not None:
                    repair_record["issues_after"] = list(repaired_issues)
            else:
                repair_errors.extend(
                    preservation_issues
                )

        if (
            repaired is not None
            and not repaired.repairable
        ):
            repair_errors.append(
                f"Candidate repair declared unrepairable: {repaired.reason}"
            )

        if repair_record is not None:
            repair_record["issues_after"] = list(repair_errors)
            candidate_repairs.append(repair_record)

        quarantined.append(
            {
                "kind": "concept",
                "concept_id": concept_id,
                "concept": concept_payload,
                "links": link_payloads,
                "issues": issues + repair_errors,
                "repair_attempted": (
                    repaired is not None
                    or bool(repair_errors)
                ),
                "repair_response": (
                    repaired.model_dump(mode="json")
                    if repaired is not None
                    else None
                ),
            }
        )

    final_graph = BridgeChunkGraph(
        **metadata,
        concepts=accepted_concepts,
        links=accepted_links,
    )

    try:
        validate_chunk_fn(
            final_graph,
            **_mini_validation_kwargs(
                metadata=metadata,
                core_text=core_text,
                strict_nodes=strict_nodes,
            ),
        )
    except Exception as error:
        # Candidate-local recovery should have made this impossible. Treat a
        # residual failure as an implementation bug, not a semantic quarantine.
        raise BridgeRecoveryError(
            "Final recovered BridgeChunkGraph failed strict validation: "
            f"{error}"
        ) from error

    return BridgeRecoveryResult(
        graph=final_graph,
        normalization_operations=operations,
        quarantined_items=quarantined,
        candidate_repair_attempts=(
            candidate_repair_attempts
        ),
        repaired_candidate_count=(
            repaired_candidate_count
        ),
        candidate_repairs=candidate_repairs,
        generated_concept_count=(
            generated_concept_count
        ),
        generated_link_count=(
            generated_link_count
        ),
    )
