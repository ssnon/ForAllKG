from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pipeline_core.discovery.direct_task_relation_backbone import (
    DirectTaskRelationBackboneView,
)
from pipeline_core.discovery.higher_order_modifier_eligibility import (
    ASSOCIATIONAL_RELATIONS,
    DIRECTIONAL_RELATIONS,
    GENERIC_PHRASES,
    GENERIC_SINGLE_TOKENS,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentAuthority,
    RelationComponentView,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    lexical_tokens,
)


@dataclass(frozen=True)
class DirectModifierEligibilityScreenResult:
    eligible_records: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


def _norm(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _slot_text(
    component: RelationComponentView,
    slot: str,
) -> str:
    if slot == "subject":
        return component.subject
    if slot == "object":
        return component.object
    raise ValueError(f"unsupported relation slot: {slot!r}")


def _other_slot(slot: str) -> str:
    if slot == "subject":
        return "object"
    if slot == "object":
        return "subject"
    raise ValueError(f"unsupported relation slot: {slot!r}")


def _containment_overlap(
    left: Iterable[str],
    right: Iterable[str],
) -> float:
    a = set(left)
    b = set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _jaccard(
    left: Iterable[str],
    right: Iterable[str],
) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _compatible_role_text(
    value: str,
    reference: str,
) -> bool:
    if _norm(value) == _norm(reference):
        return True

    left = lexical_tokens(value)
    right = lexical_tokens(reference)

    return (
        bool(left)
        and bool(right)
        and _containment_overlap(
            left,
            right,
        )
        >= 0.80
    )


def _role_vocabulary(
    backbones: Sequence[DirectTaskRelationBackboneView],
) -> dict[str, tuple[str, ...]]:
    roles: dict[str, list[str]] = {
        "source": [],
        "target": [],
    }

    def add(role: str, text: str) -> None:
        text = " ".join(str(text).split())
        if not text:
            return

        normalized = {
            _norm(row)
            for row in roles[role]
        }
        if _norm(text) not in normalized:
            roles[role].append(text)

    for backbone in backbones:
        add(
            "source",
            backbone.role_projection.source_role_text,
        )
        add(
            "target",
            backbone.role_projection.target_role_text,
        )

    return {
        role: tuple(values)
        for role, values in roles.items()
    }


def _roles_for_text(
    text: str,
    role_vocabulary: dict[str, tuple[str, ...]],
) -> set[str]:
    return {
        role
        for role, references in role_vocabulary.items()
        if any(
            _compatible_role_text(
                text,
                reference,
            )
            for reference in references
        )
    }


def screen_direct_backbone_candidate_modifiers(
    *,
    backbones: Sequence[DirectTaskRelationBackboneView],
    components: Sequence[RelationComponentView],
    dedup_jaccard_threshold: float = 0.80,
) -> DirectModifierEligibilityScreenResult:
    """
    Frozen direct-backbone mirror of the existing higher-order modifier screen.

    The scientific role vocabulary is derived only from validated direct
    task-relation role projections. Candidate components may become modifier
    inspiration only when exactly one relation argument anchors to one task
    role and the opposite argument survives the same independence/genericity/
    direction/restatement guards used by the existing higher-order screen.

    Passing this screen creates modifier-eligibility authority only. It never
    creates interaction evidence, scientific identity, positive-premise,
    novelty, certification, or production-selection authority.
    """
    role_vocabulary = _role_vocabulary(
        backbones
    )

    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    skipped_no_role = 0
    skipped_non_candidate = 0

    for component in components:
        if (
            component.authority
            != RelationComponentAuthority.CANDIDATE_INSPIRATION
        ):
            skipped_non_candidate += 1
            continue

        subject_roles = _roles_for_text(
            component.subject,
            role_vocabulary,
        )
        object_roles = _roles_for_text(
            component.object,
            role_vocabulary,
        )

        if not subject_roles and not object_roles:
            skipped_no_role += 1
            continue

        reasons: list[str] = []

        if subject_roles and object_roles:
            reasons.append(
                "MODIFIER_ROLE_LEAKAGE"
            )
            anchor_slot = (
                "subject"
                if len(subject_roles)
                <= len(object_roles)
                else "object"
            )
        elif subject_roles:
            anchor_slot = "subject"
        else:
            anchor_slot = "object"

        anchor_roles = (
            subject_roles
            if anchor_slot == "subject"
            else object_roles
        )
        modifier_roles = (
            object_roles
            if anchor_slot == "subject"
            else subject_roles
        )
        modifier_slot = _other_slot(
            anchor_slot
        )

        if len(anchor_roles) != 1:
            reasons.append(
                "ANCHOR_ROLE_CONTAMINATION"
            )
            anchor_role = (
                sorted(anchor_roles)[0]
                if anchor_roles
                else ""
            )
        else:
            anchor_role = next(
                iter(anchor_roles)
            )

        anchor_text = _slot_text(
            component,
            anchor_slot,
        )
        modifier_text = _slot_text(
            component,
            modifier_slot,
        )

        modifier_tokens = lexical_tokens(
            modifier_text
        )
        anchor_tokens = lexical_tokens(
            anchor_text
        )

        if "source" in modifier_roles:
            reasons.append(
                "SOURCE_ALIAS_LEAKAGE"
            )
        if "target" in modifier_roles:
            reasons.append(
                "TARGET_ALIAS_LEAKAGE"
            )

        if not modifier_tokens:
            reasons.append(
                "EMPTY_MODIFIER"
            )

        if _norm(modifier_text) in {
            _norm(value)
            for value in GENERIC_PHRASES
        }:
            reasons.append(
                "GENERIC_MODIFIER_PHRASE"
            )

        if (
            len(modifier_tokens) == 1
            and next(
                iter(modifier_tokens),
                "",
            )
            in GENERIC_SINGLE_TOKENS
        ):
            reasons.append(
                "GENERIC_SINGLE_TOKEN"
            )

        predicate = (
            component.relation.upper()
        )

        if predicate in DIRECTIONAL_RELATIONS:
            if modifier_slot != "subject":
                reasons.append(
                    "DIRECTIONAL_ROLE_REVERSED_OR_UNRESOLVED"
                )
        elif (
            predicate
            not in ASSOCIATIONAL_RELATIONS
        ):
            reasons.append(
                "UNRECOGNIZED_RELATION_SEMANTICS"
            )

        if (
            modifier_tokens
            and anchor_tokens
            and _containment_overlap(
                modifier_tokens,
                anchor_tokens,
            )
            >= 0.80
        ):
            reasons.append(
                "MODIFIER_ANCHOR_RESTATEMENT"
            )

        record = {
            "component_id":
                component.component_id,
            "subject":
                component.subject,
            "relation":
                component.relation,
            "object":
                component.object,
            "anchor_role":
                anchor_role,
            "anchor_slot":
                anchor_slot,
            "modifier_slot":
                modifier_slot,
            "anchor_text":
                anchor_text,
            "modifier_text":
                modifier_text,
            "modifier_tokens":
                sorted(modifier_tokens),
            "predicate":
                predicate,
            "subject_roles":
                sorted(subject_roles),
            "object_roles":
                sorted(object_roles),
            "reasons":
                sorted(set(reasons)),
            "passes_frozen_screen_mirror":
                not reasons,
        }

        if reasons:
            rejected.append(record)
        else:
            passed.append(record)

    passed.sort(
        key=lambda row: (
            -len(row["modifier_tokens"]),
            _norm(row["modifier_text"]),
            row["component_id"],
            row["anchor_role"],
        )
    )

    dedup: list[dict[str, Any]] = []
    signatures: list[set[str]] = []
    dedup_dropped: list[dict[str, Any]] = []

    for row in passed:
        token_set = set(
            row["modifier_tokens"]
        )

        if any(
            _jaccard(
                token_set,
                previous,
            )
            >= dedup_jaccard_threshold
            for previous in signatures
        ):
            dropped = dict(row)
            dropped[
                "passes_frozen_screen_mirror"
            ] = False
            dropped["reasons"] = [
                "STRICT_NEAR_DUPLICATE_SUPPRESSED"
            ]
            dedup_dropped.append(
                dropped
            )
            continue

        signatures.append(
            token_set
        )
        dedup.append(
            row
        )

    reason_counts: dict[str, int] = {}
    for row in [
        *rejected,
        *dedup_dropped,
    ]:
        for reason in row["reasons"]:
            reason_counts[reason] = (
                reason_counts.get(
                    reason,
                    0,
                )
                + 1
            )

    audit = {
        "schema_version":
            "direct-backbone-modifier-eligibility-audit-v1",
        "frozen_reference_contract":
            "higher-order-modifier-eligibility-audit-v1",
        "input_component_count":
            len(components),
        "backbone_count":
            len(backbones),
        "raw_role_candidate_count":
            len(passed) + len(rejected),
        "skipped_no_backbone_role_count":
            skipped_no_role,
        "skipped_non_candidate_authority_count":
            skipped_non_candidate,
        "independence_pass_component_count":
            len(passed),
        "independence_reject_component_count":
            len(rejected),
        "dedup_modifier_count":
            len(dedup),
        "dedup_dropped_count":
            len(dedup_dropped),
        "role_vocabulary": {
            role: list(values)
            for role, values
            in role_vocabulary.items()
        },
        "rejection_reason_counts":
            dict(sorted(
                reason_counts.items()
            )),
        "eligible_modifier_records":
            dedup,
        "records": [
            *rejected,
            *dedup_dropped,
            *dedup,
        ],
        "role_vocabulary_source":
            "direct_task_relation_role_projection_only",
        "domain_synonym_ontology_used":
            False,
        "scientific_quality_ranking_performed":
            False,
        "interaction_evidence_created":
            False,
        "scientific_identity_asserted":
            False,
        "positive_premise_authority_created":
            False,
        "novelty_authority_created":
            False,
        "production_selection_authority":
            False,
        "shadow_only":
            True,
    }

    return DirectModifierEligibilityScreenResult(
        eligible_records=tuple(
            dedup
        ),
        audit=audit,
    )


__all__ = [
    "DirectModifierEligibilityScreenResult",
    "screen_direct_backbone_candidate_modifiers",
]
