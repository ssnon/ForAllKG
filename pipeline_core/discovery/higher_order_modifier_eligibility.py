from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pipeline_core.discovery.higher_order_topology_composition import (
    EligibleModifierComponent,
    ModifierEligibilityWitness,
)
from pipeline_core.discovery.relation_component_composition import (
    RelationComponentView,
    RelationalTopologyView,
)
from pipeline_core.discovery.task_bridge_candidate_composition import lexical_tokens


DIRECTIONAL_RELATIONS = frozenset({
    "PROMOTES",
    "SUPPRESSES",
    "MODULATES",
    "MEDIATES",
    "SUGGESTS_DESIGN_RULE",
})
ASSOCIATIONAL_RELATIONS = frozenset({
    "VARIES_WITH",
    "CORRELATES_WITH",
    "CONTRASTS_WITH",
    "COMPETES_WITH",
})

# Genericity guards only. Scientific role vocabulary is never hard-coded:
# source / mediator / target are derived from validated task backbones.
GENERIC_SINGLE_TOKENS = frozenset({
    "structure",
    "geometry",
    "morphology",
    "material",
    "property",
    "properties",
    "nanostructure",
    "nanostructures",
})
GENERIC_PHRASES = frozenset({
    "geometric structure",
    "composite nanostructure geometry",
    "enhancing-substrate morphology",
    "material optical properties",
})


@dataclass(frozen=True)
class ModifierEligibilityScreenResult:
    eligible: tuple[EligibleModifierComponent, ...]
    audit: dict[str, Any]


def _norm(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _slot_text(component: RelationComponentView, slot: str) -> str:
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


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _containment_overlap(a: Iterable[str], b: Iterable[str]) -> float:
    left = set(a)
    right = set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _compatible_role_text(value: str, reference: str) -> bool:
    if _norm(value) == _norm(reference):
        return True
    left = lexical_tokens(value)
    right = lexical_tokens(reference)
    return (
        bool(left)
        and bool(right)
        and _containment_overlap(left, right) >= 0.80
    )


def _role_vocabulary(
    backbones: Sequence[RelationalTopologyView],
) -> dict[str, tuple[str, ...]]:
    roles: dict[str, list[str]] = {
        "source": [],
        "mediator": [],
        "target": [],
    }

    def add(role: str, text: str) -> None:
        text = " ".join(text.split())
        if text and _norm(text) not in {
            _norm(row) for row in roles[role]
        }:
            roles[role].append(text)

    for backbone in backbones:
        source_mediator_slot = backbone.source_binding.mediator_slot
        target_mediator_slot = backbone.target_binding.mediator_slot

        add(
            "source",
            _slot_text(
                backbone.source_component,
                _other_slot(source_mediator_slot),
            ),
        )
        add(
            "mediator",
            _slot_text(
                backbone.source_component,
                source_mediator_slot,
            ),
        )
        add(
            "mediator",
            _slot_text(
                backbone.target_component,
                target_mediator_slot,
            ),
        )
        add(
            "target",
            _slot_text(
                backbone.target_component,
                _other_slot(target_mediator_slot),
            ),
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
            _compatible_role_text(text, reference)
            for reference in references
        )
    }


def _stable_witness_id(
    component: RelationComponentView,
    *,
    anchor_role: str,
    anchor_slot: str,
    modifier_slot: str,
    anchor_text: str,
    modifier_text: str,
) -> str:
    raw = "|".join([
        component.component_id,
        anchor_role,
        anchor_slot,
        modifier_slot,
        _norm(anchor_text),
        _norm(modifier_text),
    ])
    return (
        "modifier_eligibility:"
        + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    )


def screen_confirmed_known_modifiers(
    *,
    backbones: Sequence[RelationalTopologyView],
    components: Sequence[RelationComponentView],
    dedup_jaccard_threshold: float = 0.80,
) -> ModifierEligibilityScreenResult:
    """
    Task-scoped upstream authority for higher-order modifiers.

    The role vocabulary comes only from already endpoint-faithful backbones.
    The screen preserves the S21L/S23c fail-closed dimensions without
    embedding a SERS/DAC synonym ontology:
      * anchor purity / role contamination
      * source, target, mediator leakage
      * genericity
      * directional-role validity
      * modifier/anchor restatement
      * strict near-duplicate suppression

    Passing creates modifier-eligibility authority only, never novelty
    authority or positive-premise authority.
    """
    if not backbones:
        return ModifierEligibilityScreenResult(
            eligible=(),
            audit={
                "schema_version":
                    "higher-order-modifier-eligibility-audit-v1",
                "input_component_count": len(components),
                "backbone_count": 0,
                "raw_role_candidate_count": 0,
                "independence_pass_component_count": 0,
                "independence_reject_component_count": 0,
                "dedup_modifier_count": 0,
                "dedup_dropped_count": 0,
                "role_vocabulary": {
                    "source": [],
                    "mediator": [],
                    "target": [],
                },
                "rejection_reason_counts": {},
                "records": [],
                "domain_synonym_ontology_used": False,
                "scientific_quality_ranking_performed": False,
                "novelty_authority_created": False,
            },
        )

    role_vocabulary = _role_vocabulary(backbones)
    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    skipped_no_role = 0

    for component in components:
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

        # Exactly one argument must be a single backbone-role anchor.
        # The opposite argument is the candidate modifier C.
        if subject_roles and object_roles:
            reasons.append("MODIFIER_ROLE_LEAKAGE")
            anchor_slot = (
                "subject"
                if len(subject_roles) <= len(object_roles)
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
        modifier_slot = _other_slot(anchor_slot)

        if len(anchor_roles) != 1:
            reasons.append("ANCHOR_ROLE_CONTAMINATION")
            anchor_role = (
                sorted(anchor_roles)[0]
                if anchor_roles
                else ""
            )
        else:
            anchor_role = next(iter(anchor_roles))

        anchor_text = _slot_text(component, anchor_slot)
        modifier_text = _slot_text(component, modifier_slot)
        modifier_tokens = lexical_tokens(modifier_text)
        anchor_tokens = lexical_tokens(anchor_text)

        if "source" in modifier_roles:
            reasons.append("SOURCE_ALIAS_LEAKAGE")
        if "target" in modifier_roles:
            reasons.append("TARGET_ALIAS_LEAKAGE")
        if "mediator" in modifier_roles:
            reasons.append("MEDIATOR_ALIAS_LEAKAGE")

        if not modifier_tokens:
            reasons.append("EMPTY_MODIFIER")

        if _norm(modifier_text) in GENERIC_PHRASES:
            reasons.append("GENERIC_MODIFIER_PHRASE")

        if (
            len(modifier_tokens) == 1
            and next(iter(modifier_tokens), "")
            in GENERIC_SINGLE_TOKENS
        ):
            reasons.append("GENERIC_SINGLE_TOKEN")

        predicate = component.relation.upper()

        if predicate in DIRECTIONAL_RELATIONS:
            if modifier_slot != "subject":
                reasons.append(
                    "DIRECTIONAL_ROLE_REVERSED_OR_UNRESOLVED"
                )
        elif predicate not in ASSOCIATIONAL_RELATIONS:
            reasons.append("UNRECOGNIZED_RELATION_SEMANTICS")

        if (
            modifier_tokens
            and anchor_tokens
            and _containment_overlap(
                modifier_tokens,
                anchor_tokens,
            ) >= 0.80
        ):
            reasons.append("MODIFIER_ANCHOR_RESTATEMENT")

        record = {
            "component": component,
            "component_id": component.component_id,
            "accepted_pattern_id": component.accepted_pattern_id,
            "anchor_role": anchor_role,
            "anchor_slot": anchor_slot,
            "modifier_slot": modifier_slot,
            "anchor_text": anchor_text,
            "modifier_text": modifier_text,
            "modifier_tokens": sorted(modifier_tokens),
            "predicate": predicate,
            "subject_roles": sorted(subject_roles),
            "object_roles": sorted(object_roles),
            "reasons": sorted(set(reasons)),
        }

        if reasons:
            rejected.append(record)
        else:
            passed.append(record)

    # Frozen S23c ordering + strict near-duplicate suppression.
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

    for row in passed:
        tokens = set(row["modifier_tokens"])
        if any(
            _jaccard(tokens, previous)
            >= dedup_jaccard_threshold
            for previous in signatures
        ):
            continue
        signatures.append(tokens)
        dedup.append(row)

    eligible: list[EligibleModifierComponent] = []

    for row in dedup:
        component = row["component"]
        provenance_ids = [
            str(component.provenance.source_id),
        ]
        if component.accepted_pattern_id is not None:
            provenance_ids.append(
                "accepted_pattern_id:"
                + str(component.accepted_pattern_id)
            )

        witness = ModifierEligibilityWitness(
            witness_id=_stable_witness_id(
                component,
                anchor_role=row["anchor_role"],
                anchor_slot=row["anchor_slot"],
                modifier_slot=row["modifier_slot"],
                anchor_text=row["anchor_text"],
                modifier_text=row["modifier_text"],
            ),
            modifier_component_id=component.component_id,
            validated_anchor_role=row["anchor_role"],
            validated_anchor_slot=row["anchor_slot"],
            modifier_slot=row["modifier_slot"],
            validated_anchor_text=row["anchor_text"],
            modifier_text=row["modifier_text"],
            anchor_purity_pass=True,
            source_leakage_pass=True,
            target_leakage_pass=True,
            mediator_leakage_pass=True,
            genericity_pass=True,
            directional_role_pass=True,
            provenance_ids=provenance_ids,
            reason_codes=[
                "dynamic_endpoint_fidelity_role_projection",
                "anchor_purity_pass",
                "source_target_mediator_leakage_pass",
                "genericity_pass",
                "directional_role_pass",
                "modifier_anchor_restatement_pass",
                "s21l_s23c_independence_contract",
            ],
        )
        eligible.append(
            EligibleModifierComponent(
                component=component,
                eligibility=witness,
            )
        )

    rejection_counts: dict[str, int] = {}
    for row in rejected:
        for reason in row["reasons"]:
            rejection_counts[reason] = (
                rejection_counts.get(reason, 0) + 1
            )

    serializable_records = [
        {
            key: value
            for key, value in row.items()
            if key != "component"
        }
        for row in [*rejected, *dedup]
    ]

    return ModifierEligibilityScreenResult(
        eligible=tuple(eligible),
        audit={
            "schema_version":
                "higher-order-modifier-eligibility-audit-v1",
            "input_component_count": len(components),
            "backbone_count": len(backbones),
            "raw_role_candidate_count":
                len(passed) + len(rejected),
            "skipped_no_backbone_role_count":
                skipped_no_role,
            "independence_pass_component_count":
                len(passed),
            "independence_reject_component_count":
                len(rejected),
            "dedup_modifier_count": len(dedup),
            "dedup_dropped_count":
                len(passed) - len(dedup),
            "role_vocabulary": {
                role: list(values)
                for role, values in role_vocabulary.items()
            },
            "rejection_reason_counts":
                dict(sorted(rejection_counts.items())),
            "records": serializable_records,
            "role_vocabulary_source":
                "endpoint_fidelity_backbones_only",
            "domain_synonym_ontology_used": False,
            "scientific_quality_ranking_performed": False,
            "novelty_authority_created": False,
        },
    )
