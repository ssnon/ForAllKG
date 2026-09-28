from __future__ import annotations

import hashlib
import re
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateRelationView(StrictModel):
    unit_id: str
    label: str = ""

    proposed_subject: str
    proposed_relation: str
    proposed_object: str

    occurrences: int = 0



class RelationRoleBindingView(StrictModel):
    # Diagnostic-only argument-slot binding for S18b-1.
    schema_version: str = (
        "relation-role-binding-v1"
    )

    task_subject_tokens: list[str] = Field(
        default_factory=list
    )
    task_predicate_tokens: list[str] = Field(
        default_factory=list
    )
    task_object_tokens: list[str] = Field(
        default_factory=list
    )

    mediator_subject_tokens: list[str] = Field(
        default_factory=list
    )
    mediator_predicate_tokens: list[str] = Field(
        default_factory=list
    )
    mediator_object_tokens: list[str] = Field(
        default_factory=list
    )


class TaskBridgeCompositeCandidate(StrictModel):
    schema_version: str = (
        "task-bridge-composite-candidate-v1"
    )

    composite_id: str

    source_unit_id: str
    target_unit_id: str

    source_overlap_tokens: list[str]
    target_overlap_tokens: list[str]

    source_mediator_tokens: list[str]
    target_mediator_tokens: list[str]
    shared_mediator_tokens: list[str]

    source_relation: CandidateRelationView
    target_relation: CandidateRelationView

    source_role_binding: RelationRoleBindingView = Field(
        default_factory=RelationRoleBindingView
    )
    target_role_binding: RelationRoleBindingView = Field(
        default_factory=RelationRoleBindingView
    )

    compatibility_score: float

    epistemic_status: str = (
        "inspiration_only"
    )

    requires_verification: bool = True

    reason_codes: list[str] = Field(
        default_factory=lambda: [
            "task_conditioned_candidate_composition",
            "source_relation_inspiration_only",
            "target_relation_inspiration_only",
            "shared_mediator_required",
            "composite_requires_verification",
        ]
    )

    # S22a relation-component bridge metadata.
    #
    # Defaults preserve the frozen lexical-candidate contract.  S22a
    # topologies populate these fields explicitly so known relation
    # components never masquerade as discovery candidates.
    source_component_id: str = ""
    target_component_id: str = ""

    source_component_authority: Literal[
        "candidate_inspiration",
        "confirmed_known",
    ] = "candidate_inspiration"

    target_component_authority: Literal[
        "candidate_inspiration",
        "confirmed_known",
    ] = "candidate_inspiration"

    provenance_candidate_unit_id: str | None = None
    topology_id: str | None = None

    composition_mode: Literal[
        "candidate_lexical_v1",
        "relation_component_topology_v1",
    ] = "candidate_lexical_v1"

    @model_validator(mode="after")
    def validate_relation_component_topology_shape(
        self,
    ) -> "TaskBridgeCompositeCandidate":
        if (
            self.composition_mode
            != "relation_component_topology_v1"
        ):
            return self

        if not self.topology_id:
            raise ValueError(
                "relation-component topology requires topology_id"
            )
        if not self.source_component_id:
            raise ValueError(
                "relation-component topology requires source_component_id"
            )
        if not self.target_component_id:
            raise ValueError(
                "relation-component topology requires target_component_id"
            )
        if not self.provenance_candidate_unit_id:
            raise ValueError(
                "relation-component topology requires a real candidate "
                "provenance anchor"
            )

        return self


_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+"
)

_TARGET_ATOM_SPLIT_RE = re.compile(
    r"\s*(?:,|\band\b|\bor\b)\s*",
    flags=re.IGNORECASE,
)

_STOP = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
    "within",

    # Generic relation-language tokens.
    "affect",
    "affects",
    "alter",
    "alters",
    "depend",
    "depends",
    "modulate",
    "modulates",
    "promote",
    "promotes",
    "relate",
    "relates",
    "relationship",
    "suggest",
    "suggests",
    "vary",
    "varies",
}


def _stem_token(
    token: str,
) -> str:
    token = token.lower().strip()

    if len(token) <= 3:
        return token

    if (
        token.endswith("ies")
        and len(token) > 4
    ):
        return (
            token[:-3]
            + "y"
        )

    if (
        token.endswith("ses")
        and len(token) > 4
    ):
        # e.g. "responses" is handled below;
        # avoid aggressive linguistic stemming.
        return token

    if (
        token.endswith("s")
        and not token.endswith("ss")
        and len(token) > 4
    ):
        return token[:-1]

    return token


def lexical_tokens(
    text: str,
) -> frozenset[str]:
    normalized = str(text).replace(
        "_",
        " ",
    ).replace(
        "-",
        " ",
    ).replace(
        "/",
        " ",
    )

    out = set()

    for token in _TOKEN_RE.findall(
        normalized
    ):
        token = _stem_token(
            token
        )

        if (
            len(token) >= 2
            and token not in _STOP
        ):
            out.add(token)

    return frozenset(
        out
    )


def relation_tokens(
    relation: CandidateRelationView,
) -> frozenset[str]:
    return lexical_tokens(
        " ".join(
            [
                relation.proposed_subject,
                relation.proposed_relation,
                relation.proposed_object,
            ]
        )
    )


def _target_endpoint_atoms(
    text: str,
) -> tuple[frozenset[str], ...]:
    atoms = []
    seen = set()

    for raw in _TARGET_ATOM_SPLIT_RE.split(
        str(text)
    ):
        tokens = lexical_tokens(
            raw
        )

        if not tokens:
            continue

        key = tuple(
            sorted(tokens)
        )

        if key in seen:
            continue

        seen.add(key)
        atoms.append(
            tokens
        )

    return tuple(
        atoms
    )


def _target_endpoint_atom_matches(
    *,
    relation: CandidateRelationView,
    target_atoms: tuple[
        frozenset[str],
        ...,
    ],
) -> bool:
    if len(target_atoms) < 2:
        return True

    subject_tokens = lexical_tokens(
        relation.proposed_subject
    )

    object_tokens = lexical_tokens(
        relation.proposed_object
    )

    return any(
        atom <= subject_tokens
        or atom <= object_tokens
        for atom in target_atoms
    )


def _relation_role_binding(
    *,
    relation: CandidateRelationView,
    task_tokens: frozenset[str],
    mediator_tokens: frozenset[str],
) -> RelationRoleBindingView:
    # Diagnostic-only in S18b-1. Existing eligibility/ranking remains frozen.
    subject_tokens = lexical_tokens(
        relation.proposed_subject
    )
    predicate_tokens = lexical_tokens(
        relation.proposed_relation
    )
    object_tokens = lexical_tokens(
        relation.proposed_object
    )

    return RelationRoleBindingView(
        task_subject_tokens=sorted(
            subject_tokens & task_tokens
        ),
        task_predicate_tokens=sorted(
            predicate_tokens & task_tokens
        ),
        task_object_tokens=sorted(
            object_tokens & task_tokens
        ),
        mediator_subject_tokens=sorted(
            subject_tokens & mediator_tokens
        ),
        mediator_predicate_tokens=sorted(
            predicate_tokens & mediator_tokens
        ),
        mediator_object_tokens=sorted(
            object_tokens & mediator_tokens
        ),
    )



def _endpoint_role_shadow(
    *,
    relation: CandidateRelationView,
    task_endpoint: str,
) -> dict[str, Any]:
    """
    Diagnose how one task endpoint binds to explicit relation argument slots.

    Shadow-only contract:
    - exact means normalized task tokens equal one complete subject/object slot;
    - partial means only a non-empty token overlap exists;
    - no semantic synonymy/equivalence is inferred;
    - ties between equally strong subject/object bindings fail closed as ambiguous.
    """

    task_tokens = lexical_tokens(
        task_endpoint
    )

    slot_tokens = {
        "subject": lexical_tokens(
            relation.proposed_subject
        ),
        "object": lexical_tokens(
            relation.proposed_object
        ),
    }

    rows = []

    for slot in (
        "subject",
        "object",
    ):
        tokens = slot_tokens[slot]
        overlap = (
            tokens
            & task_tokens
        )

        if (
            task_tokens
            and tokens
            and tokens == task_tokens
        ):
            authority = "exact"
            authority_rank = 2
        elif overlap:
            authority = "partial"
            authority_rank = 1
        else:
            authority = "none"
            authority_rank = 0

        coverage = (
            len(overlap)
            / len(task_tokens)
            if task_tokens
            else 0.0
        )

        rows.append(
            {
                "slot": slot,
                "binding_authority": authority,
                "authority_rank": authority_rank,
                "task_coverage": float(coverage),
                "matched_tokens": sorted(
                    overlap
                ),
                "slot_tokens": sorted(
                    tokens
                ),
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row["authority_rank"]),
            -float(row["task_coverage"]),
            -len(row["matched_tokens"]),
            str(row["slot"]),
        ),
    )

    best = ranked[0]

    if int(best["authority_rank"]) == 0:
        binding_slot = None
        binding_authority = "none"
        mediator_slot = None
        mediator_tokens: list[str] = []
        ambiguous = False
    else:
        tied = [
            row
            for row in ranked
            if (
                row["authority_rank"]
                == best["authority_rank"]
                and row["task_coverage"]
                == best["task_coverage"]
                and len(row["matched_tokens"])
                == len(best["matched_tokens"])
            )
        ]

        ambiguous = (
            len(tied) > 1
        )

        if ambiguous:
            binding_slot = None
            binding_authority = "ambiguous"
            mediator_slot = None
            mediator_tokens = []
        else:
            binding_slot = str(
                best["slot"]
            )
            binding_authority = str(
                best["binding_authority"]
            )
            mediator_slot = (
                "object"
                if binding_slot == "subject"
                else "subject"
            )
            mediator_tokens = sorted(
                slot_tokens[
                    mediator_slot
                ]
                - task_tokens
            )

    return {
        "task_endpoint": str(
            task_endpoint
        ),
        "task_tokens": sorted(
            task_tokens
        ),
        "binding_authority": (
            binding_authority
        ),
        "binding_slot": (
            binding_slot
        ),
        "ambiguous": bool(
            ambiguous
        ),
        "task_coverage": (
            float(best["task_coverage"])
            if not ambiguous
            else 0.0
        ),
        "matched_tokens": (
            list(best["matched_tokens"])
            if not ambiguous
            else []
        ),
        "mediator_slot": (
            mediator_slot
        ),
        "mediator_tokens": (
            mediator_tokens
        ),
        "slot_diagnostics": [
            {
                key: value
                for key, value in row.items()
                if key != "authority_rank"
            }
            for row in rows
        ],
    }


def diagnose_task_bridge_candidate(
    *,
    composite: TaskBridgeCompositeCandidate,
    requested_source: str,
    requested_target: str,
) -> dict[str, Any]:
    """
    Compare the frozen legacy lexical bridge with strict slot-aware endpoint
    and mediator fidelity without changing composition, ranking, or selection.
    """

    source = _endpoint_role_shadow(
        relation=composite.source_relation,
        task_endpoint=requested_source,
    )
    target = _endpoint_role_shadow(
        relation=composite.target_relation,
        task_endpoint=requested_target,
    )

    task_tokens = (
        lexical_tokens(
            requested_source
        )
        |
        lexical_tokens(
            requested_target
        )
    )

    source_mediator = (
        frozenset(
            source["mediator_tokens"]
        )
        - task_tokens
    )
    target_mediator = (
        frozenset(
            target["mediator_tokens"]
        )
        - task_tokens
    )

    role_shared = (
        source_mediator
        & target_mediator
    )

    exact_endpoint_fidelity = bool(
        source["binding_authority"]
        == "exact"
        and target["binding_authority"]
        == "exact"
    )

    strict_materializable = bool(
        exact_endpoint_fidelity
        and role_shared
    )

    return {
        "schema_version": (
            "legacy-task-bridge-endpoint-role-shadow-v1"
        ),
        "composite_id": (
            composite.composite_id
        ),
        "composition_mode": (
            composite.composition_mode
        ),
        "source_unit_id": (
            composite.source_unit_id
        ),
        "target_unit_id": (
            composite.target_unit_id
        ),
        "legacy_source_overlap_tokens": list(
            composite.source_overlap_tokens
        ),
        "legacy_target_overlap_tokens": list(
            composite.target_overlap_tokens
        ),
        "legacy_shared_mediator_tokens": list(
            composite.shared_mediator_tokens
        ),
        "source_endpoint_binding": source,
        "target_endpoint_binding": target,
        "role_aware_source_mediator_tokens": sorted(
            source_mediator
        ),
        "role_aware_target_mediator_tokens": sorted(
            target_mediator
        ),
        "role_aware_shared_mediator_tokens": sorted(
            role_shared
        ),
        "role_aware_mediator_compatible": bool(
            role_shared
        ),
        "exact_endpoint_fidelity": (
            exact_endpoint_fidelity
        ),
        "strict_materializable_without_equivalence_witness": (
            strict_materializable
        ),
        "semantic_equivalence_inferred": False,
        "shadow_only": True,
        "production_selection_changed": False,
    }


def _stable_id(
    prefix: str,
    *parts: object,
) -> str:
    raw = "|".join(
        str(x)
        for x in parts
    ).encode("utf-8")

    return (
        prefix
        + ":"
        + hashlib.sha256(
            raw
        ).hexdigest()[:20]
    )


def candidate_relation_from_mapping(
    row: Mapping[str, Any],
) -> CandidateRelationView:
    return CandidateRelationView(
        unit_id=str(
            row.get(
                "unit_id",
                "",
            )
        ),
        label=str(
            row.get(
                "label",
                "",
            )
            or ""
        ),
        proposed_subject=str(
            row.get(
                "proposed_subject",
                "",
            )
            or ""
        ),
        proposed_relation=str(
            row.get(
                "proposed_relation",
                "",
            )
            or ""
        ),
        proposed_object=str(
            row.get(
                "proposed_object",
                "",
            )
            or ""
        ),
        occurrences=int(
            row.get(
                "occurrences",
                0,
            )
            or 0
        ),
    )


def compose_task_bridge_candidates(
    *,
    candidates: Sequence[
        CandidateRelationView
    ],
    requested_source: str,
    requested_target: str,
    max_composites: int = 12,
) -> tuple[
    TaskBridgeCompositeCandidate,
    ...,
]:
    """
    Compose source-side and target-side unverified candidate relations
    through shared mediator vocabulary.

    This function does NOT infer that either component relation applies
    to the user's requested system. Component relations remain
    inspiration-only and every composite requires verification.

    Selection contract:
      1. source-side candidate must lexically overlap requested source;
      2. target-side candidate must lexically overlap requested target;
      3. for explicitly compound target endpoints, at least one complete
         target atom must be preserved in one target relation argument;
      4. after removing task-nucleus tokens, both candidate relations
         must share at least one mediator token;
      5. rank by mediator overlap/coverage, never by downstream novelty
         or semantic-distinctiveness outcomes.
    """

    if max_composites < 1:
        raise ValueError(
            "max_composites must be >= 1"
        )

    source_task_tokens = lexical_tokens(
        requested_source
    )

    target_task_tokens = lexical_tokens(
        requested_target
    )

    target_endpoint_atoms = (
        _target_endpoint_atoms(
            requested_target
        )
    )

    if not source_task_tokens:
        raise ValueError(
            "requested_source produced no tokens"
        )

    if not target_task_tokens:
        raise ValueError(
            "requested_target produced no tokens"
        )

    if not target_endpoint_atoms:
        raise ValueError(
            "requested_target produced no endpoint atoms"
        )

    rows = []

    source_candidates = []
    target_candidates = []

    for candidate in candidates:
        tokens = relation_tokens(
            candidate
        )

        source_overlap = (
            tokens
            & source_task_tokens
        )

        target_overlap = (
            tokens
            & target_task_tokens
        )

        if source_overlap:
            source_candidates.append(
                (
                    candidate,
                    tokens,
                    source_overlap,
                )
            )

        if (
            target_overlap
            and
            _target_endpoint_atom_matches(
                relation=candidate,
                target_atoms=(
                    target_endpoint_atoms
                ),
            )
        ):
            target_candidates.append(
                (
                    candidate,
                    tokens,
                    target_overlap,
                )
            )

    for (
        source_candidate,
        source_tokens,
        source_overlap,
    ) in source_candidates:

        source_mediator = (
            source_tokens
            - source_task_tokens
            - target_task_tokens
        )

        if not source_mediator:
            continue

        for (
            target_candidate,
            target_tokens,
            target_overlap,
        ) in target_candidates:

            if (
                source_candidate.unit_id
                == target_candidate.unit_id
            ):
                continue

            target_mediator = (
                target_tokens
                - source_task_tokens
                - target_task_tokens
            )

            if not target_mediator:
                continue

            shared = (
                source_mediator
                & target_mediator
            )

            if not shared:
                continue

            union = (
                source_mediator
                | target_mediator
            )

            jaccard = (
                len(shared)
                / len(union)
                if union
                else 0.0
            )

            source_coverage = (
                len(shared)
                / len(source_mediator)
            )

            target_coverage = (
                len(shared)
                / len(target_mediator)
            )

            # Shared-token count is primary so two independently
            # matching mediator concepts outrank a one-token accident.
            # Coverage then prefers tighter bridge relations.
            score = (
                2.0 * len(shared)
                + 0.75 * jaccard
                + 0.25 * min(
                    source_coverage,
                    target_coverage,
                )
            )

            composite_id = _stable_id(
                "task_bridge_composite",
                source_candidate.unit_id,
                target_candidate.unit_id,
                *sorted(shared),
            )

            rows.append(
                TaskBridgeCompositeCandidate(
                    composite_id=composite_id,

                    source_unit_id=(
                        source_candidate.unit_id
                    ),

                    target_unit_id=(
                        target_candidate.unit_id
                    ),

                    source_overlap_tokens=sorted(
                        source_overlap
                    ),

                    target_overlap_tokens=sorted(
                        target_overlap
                    ),

                    source_mediator_tokens=sorted(
                        source_mediator
                    ),

                    target_mediator_tokens=sorted(
                        target_mediator
                    ),

                    shared_mediator_tokens=sorted(
                        shared
                    ),

                    source_relation=(
                        source_candidate
                    ),

                    target_relation=(
                        target_candidate
                    ),

                    source_role_binding=(
                        _relation_role_binding(
                            relation=source_candidate,
                            task_tokens=source_task_tokens,
                            mediator_tokens=shared,
                        )
                    ),

                    target_role_binding=(
                        _relation_role_binding(
                            relation=target_candidate,
                            task_tokens=target_task_tokens,
                            mediator_tokens=shared,
                        )
                    ),

                    compatibility_score=float(
                        score
                    ),
                )
            )

    rows.sort(
        key=lambda row: (
            -len(
                row.shared_mediator_tokens
            ),
            -row.compatibility_score,
            row.source_unit_id,
            row.target_unit_id,
        )
    )

    # Stable pair-level de-duplication.
    seen = set()
    selected = []

    for row in rows:
        key = (
            row.source_unit_id,
            row.target_unit_id,
        )

        if key in seen:
            continue

        seen.add(key)
        selected.append(
            row
        )

        if len(selected) >= max_composites:
            break

    return tuple(
        selected
    )
