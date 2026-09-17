from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.corpus.bridge.bridge_schemas import BridgeConcept
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
    RelationRoleBindingView,
    TaskBridgeCompositeCandidate,
    lexical_tokens,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RelationComponentAuthority(str, Enum):
    CONFIRMED_KNOWN = "confirmed_known"
    CANDIDATE_INSPIRATION = "candidate_inspiration"


class RelationComponentProvenance(StrictModel):
    source_kind: Literal[
        "accepted_pattern",
        "candidate_unit",
    ]
    source_id: str

    paper_id: str = ""
    chunk_id: str = ""
    document_id: str = ""
    source_path_id: str = ""

    evidence_scope: str = ""
    pattern_support_mode: str = ""
    relation_strength: str = ""


class RelationComponentView(StrictModel):
    schema_version: str = "relation-component-v1"

    component_id: str
    label: str = ""

    subject: str
    relation: str
    object: str

    authority: RelationComponentAuthority
    provenance: RelationComponentProvenance

    accepted_pattern_id: str | None = None
    candidate_unit_id: str | None = None

    candidate_unit_score: float | None = None
    exploration_score: float | None = None

    @model_validator(mode="after")
    def validate_authority_shape(
        self,
    ) -> "RelationComponentView":
        if (
            self.authority
            == RelationComponentAuthority.CONFIRMED_KNOWN
        ):
            if not self.accepted_pattern_id:
                raise ValueError(
                    "CONFIRMED_KNOWN requires accepted_pattern_id"
                )
            if self.candidate_unit_id is not None:
                raise ValueError(
                    "CONFIRMED_KNOWN cannot carry candidate_unit_id"
                )
            if (
                self.candidate_unit_score is not None
                or self.exploration_score is not None
            ):
                raise ValueError(
                    "CONFIRMED_KNOWN cannot carry candidate scores"
                )
            if self.provenance.source_kind != "accepted_pattern":
                raise ValueError(
                    "CONFIRMED_KNOWN requires accepted_pattern provenance"
                )
            if (
                self.provenance.source_id
                != self.accepted_pattern_id
            ):
                raise ValueError(
                    "CONFIRMED_KNOWN provenance source_id must match "
                    "accepted_pattern_id"
                )
            if not all(
                str(value).strip()
                for value in (
                    self.provenance.paper_id,
                    self.provenance.chunk_id,
                    self.provenance.document_id,
                )
            ):
                raise ValueError(
                    "CONFIRMED_KNOWN requires paper/chunk/document provenance"
                )
        else:
            if not self.candidate_unit_id:
                raise ValueError(
                    "CANDIDATE_INSPIRATION requires candidate_unit_id"
                )
            if self.accepted_pattern_id is not None:
                raise ValueError(
                    "CANDIDATE_INSPIRATION cannot carry accepted_pattern_id"
                )
            if (
                self.candidate_unit_score is None
                or self.exploration_score is None
            ):
                raise ValueError(
                    "CANDIDATE_INSPIRATION requires evaluated candidate scores"
                )
            if self.provenance.source_kind != "candidate_unit":
                raise ValueError(
                    "CANDIDATE_INSPIRATION requires candidate_unit provenance"
                )
            if (
                self.provenance.source_id
                != self.candidate_unit_id
            ):
                raise ValueError(
                    "CANDIDATE_INSPIRATION provenance source_id must match "
                    "candidate_unit_id"
                )
            if not str(
                self.provenance.source_path_id
            ).strip():
                raise ValueError(
                    "CANDIDATE_INSPIRATION requires source_path_id provenance"
                )

        return self


RelationArgumentSlot = Literal[
    "subject",
    "object",
]


EndpointBindingAuthority = Literal[
    "exact",
    "equivalent",
    "partial",
]


class EndpointEquivalenceWitness(StrictModel):
    """
    Explicit, auditable endpoint-equivalence evidence.

    The core composer never infers scientific synonymy from lexical
    similarity or embedding thresholds. EQUIVALENT binding authority
    exists only when one of these witnesses is supplied by the caller.
    """

    schema_version: str = "endpoint-equivalence-witness-v1"

    witness_id: str
    left_endpoint: str
    right_endpoint: str

    witness_kind: Literal[
        "explicit_alias",
        "registry_identity",
        "task_supplied",
    ]

    provenance_ids: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_witness(
        self,
    ) -> "EndpointEquivalenceWitness":
        if not str(self.witness_id).strip():
            raise ValueError(
                "endpoint-equivalence witness_id must not be empty"
            )
        if not str(self.left_endpoint).strip():
            raise ValueError(
                "endpoint-equivalence left_endpoint must not be empty"
            )
        if not str(self.right_endpoint).strip():
            raise ValueError(
                "endpoint-equivalence right_endpoint must not be empty"
            )
        return self


class MediatorEquivalenceWitness(StrictModel):
    """
    Explicit, auditable mediator-equivalence evidence.

    The lexical mediator gate remains unchanged. This witness is a separate
    authority path used only when lexical mediator compatibility fails.
    """

    schema_version: str = "mediator-equivalence-witness-v1"

    witness_id: str
    left_mediator: str
    right_mediator: str
    canonical_mediator: str

    witness_kind: Literal[
        "explicit_alias",
        "registry_identity",
        "domain_profile_normalization",
        "domain_profile_alias",
        "document_coreference",
        "task_supplied",
    ]

    provenance_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_witness(
        self,
    ) -> "MediatorEquivalenceWitness":
        required = (
            self.witness_id,
            self.left_mediator,
            self.right_mediator,
            self.canonical_mediator,
        )
        if not all(str(value).strip() for value in required):
            raise ValueError(
                "mediator-equivalence witness requires complete identity text"
            )

        if not lexical_tokens(self.canonical_mediator):
            raise ValueError(
                "mediator-equivalence canonical_mediator produced no tokens"
            )

        return self


class RelationComponentBindingView(StrictModel):
    task_slot: RelationArgumentSlot
    mediator_slot: RelationArgumentSlot

    task_overlap_tokens: list[str] = Field(
        default_factory=list
    )
    mediator_tokens: list[str] = Field(
        default_factory=list
    )

    # S22c endpoint-fidelity authority. PARTIAL is diagnostic-only.
    binding_authority: EndpointBindingAuthority = "partial"
    matched_endpoint_atom: str = ""
    task_coverage: float = 0.0
    slot_coverage: float = 0.0
    equivalence_witness_id: str | None = None

    @model_validator(mode="after")
    def validate_endpoint_authority(
        self,
    ) -> "RelationComponentBindingView":
        if (
            self.binding_authority == "equivalent"
            and not self.equivalence_witness_id
        ):
            raise ValueError(
                "equivalent endpoint binding requires an explicit witness"
            )
        if (
            self.binding_authority != "equivalent"
            and self.equivalence_witness_id is not None
        ):
            raise ValueError(
                "only equivalent endpoint binding may carry a witness"
            )
        if not (0.0 <= float(self.task_coverage) <= 1.0):
            raise ValueError(
                "task_coverage must be within [0, 1]"
            )
        if not (0.0 <= float(self.slot_coverage) <= 1.0):
            raise ValueError(
                "slot_coverage must be within [0, 1]"
            )
        return self


class RelationalTopologyView(StrictModel):
    schema_version: str = "relational-topology-v1"

    topology_id: str

    source_component: RelationComponentView
    target_component: RelationComponentView

    source_binding: RelationComponentBindingView
    target_binding: RelationComponentBindingView

    shared_mediator_tokens: list[str] = Field(
        default_factory=list
    )
    compatibility_score: float

    epistemic_status: Literal[
        "inspiration_only"
    ] = "inspiration_only"

    requires_verification: Literal[
        True
    ] = True

    reason_codes: list[str] = Field(
        default_factory=list
    )


_ENDPOINT_ATOM_SPLIT_RE = re.compile(
    r"\s*(?:,|\band\b|\bor\b)\s*",
    flags=re.IGNORECASE,
)

_ENDPOINT_DASHES = "‐‑‒–—−"


def _stable_id(
    prefix: str,
    *parts: object,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    return (
        prefix
        + ":"
        + hashlib.sha256(
            raw
        ).hexdigest()[:20]
    )


def _json_list(
    value: Any,
) -> list[Any]:
    if isinstance(value, list):
        return value

    if value in (
        None,
        "",
    ):
        return []

    try:
        parsed = json.loads(
            str(value)
        )
    except (
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "Expected a JSON list"
        ) from exc

    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "Expected a JSON list"
        )

    return parsed


def _optional_text(
    value: Any,
) -> str | None:
    text = str(
        value or ""
    ).strip()

    return (
        text
        if text
        else None
    )


def _bridge_concept_from_pattern_mapping(
    row: Mapping[str, Any],
) -> BridgeConcept:
    concept_id = str(
        row.get(
            "source_local_id",
            "",
        )
        or row.get(
            "id",
            "",
        )
        or row.get(
            "node_id",
            "",
        )
    ).strip()

    if not concept_id:
        raise ValueError(
            "Accepted pattern row is missing a concept identifier"
        )

    return BridgeConcept.model_validate(
        {
            "id":
                concept_id,
            "concept_type":
                str(
                    row.get(
                        "concept_type",
                        "",
                    )
                ),
            "label":
                str(
                    row.get(
                        "label",
                        "",
                    )
                ),
            "source_phrase":
                str(
                    row.get(
                        "source_phrase",
                        "",
                    )
                ),
            "description":
                _optional_text(
                    row.get(
                        "description"
                    )
                ),
            "retention_lane":
                str(
                    row.get(
                        "retention_lane",
                        "",
                    )
                ),
            "evidence_scope":
                str(
                    row.get(
                        "evidence_scope",
                        "",
                    )
                ),
            "pattern_subject":
                _optional_text(
                    row.get(
                        "pattern_subject"
                    )
                ),
            "pattern_relation":
                _optional_text(
                    row.get(
                        "pattern_relation"
                    )
                ),
            "pattern_object":
                _optional_text(
                    row.get(
                        "pattern_object"
                    )
                ),
            "relation_strength":
                _optional_text(
                    row.get(
                        "relation_strength"
                    )
                ),
            "qualifiers":
                _json_list(
                    row.get(
                        "qualifiers_json",
                        row.get(
                            "qualifiers",
                            [],
                        ),
                    )
                ),
            "pattern_support_mode":
                _optional_text(
                    row.get(
                        "pattern_support_mode"
                    )
                ),
            "supporting_phrases":
                _json_list(
                    row.get(
                        "supporting_phrases_json",
                        row.get(
                            "supporting_phrases",
                            [],
                        ),
                    )
                ),
            "subject_evidence_phrase":
                _optional_text(
                    row.get(
                        "subject_evidence_phrase"
                    )
                ),
            "relation_evidence_phrase":
                _optional_text(
                    row.get(
                        "relation_evidence_phrase"
                    )
                ),
            "object_evidence_phrase":
                _optional_text(
                    row.get(
                        "object_evidence_phrase"
                    )
                ),
            "comparison_items":
                _json_list(
                    row.get(
                        "comparison_items_json",
                        row.get(
                            "comparison_items",
                            [],
                        ),
                    )
                ),
        }
    )


def confirmed_known_component_from_mapping(
    row: Mapping[str, Any],
) -> RelationComponentView:
    """
    Re-validate one serialized accepted bridge pattern and expose it only
    as a confirmed-known composition component.

    This authority permits composition use. It does not create novelty,
    discovery-candidate, or positive-premise authority.
    """

    concept = (
        _bridge_concept_from_pattern_mapping(
            row
        )
    )

    if (
        concept.retention_lane
        != "accepted_pattern"
    ):
        raise ValueError(
            "Known relation components require retention_lane=accepted_pattern"
        )

    assert concept.pattern_subject is not None
    assert concept.pattern_relation is not None
    assert concept.pattern_object is not None
    assert concept.relation_strength is not None

    accepted_pattern_id = str(
        row.get(
            "node_id",
            "",
        )
        or concept.id
    ).strip()

    return RelationComponentView(
        component_id=_stable_id(
            "relation_component",
            RelationComponentAuthority
            .CONFIRMED_KNOWN
            .value,
            accepted_pattern_id,
            concept.pattern_subject,
            concept.pattern_relation,
            concept.pattern_object,
        ),
        label=concept.label,
        subject=concept.pattern_subject,
        relation=concept.pattern_relation,
        object=concept.pattern_object,
        authority=(
            RelationComponentAuthority
            .CONFIRMED_KNOWN
        ),
        provenance=(
            RelationComponentProvenance(
                source_kind="accepted_pattern",
                source_id=accepted_pattern_id,
                paper_id=str(
                    row.get(
                        "paper_id",
                        "",
                    )
                    or ""
                ),
                chunk_id=str(
                    row.get(
                        "chunk_id",
                        "",
                    )
                    or ""
                ),
                document_id=str(
                    row.get(
                        "document_id",
                        "",
                    )
                    or ""
                ),
                evidence_scope=(
                    concept.evidence_scope
                ),
                pattern_support_mode=(
                    concept.pattern_support_mode
                    or ""
                ),
                relation_strength=(
                    concept.relation_strength
                ),
            )
        ),
        accepted_pattern_id=(
            accepted_pattern_id
        ),
        candidate_unit_id=None,
        candidate_unit_score=None,
        exploration_score=None,
    )


def candidate_inspiration_component(
    *,
    relation: CandidateRelationView,
    candidate_unit_score: float,
    exploration_score: float,
    quality_eligible: bool,
    source_path_id: str = "",
) -> RelationComponentView:
    """
    Adapt an already quality-eligible discovery candidate.

    Quality eligibility remains owned by the caller's frozen discovery
    policy. This adapter never lowers or reinterprets those gates.
    """

    if quality_eligible is not True:
        raise ValueError(
            "Candidate relation must pass the frozen discovery quality gate"
        )

    unit_id = str(
        relation.unit_id
    ).strip()

    if not unit_id:
        raise ValueError(
            "Candidate relation is missing unit_id"
        )

    return RelationComponentView(
        component_id=_stable_id(
            "relation_component",
            RelationComponentAuthority
            .CANDIDATE_INSPIRATION
            .value,
            unit_id,
            relation.proposed_subject,
            relation.proposed_relation,
            relation.proposed_object,
        ),
        label=relation.label,
        subject=relation.proposed_subject,
        relation=relation.proposed_relation,
        object=relation.proposed_object,
        authority=(
            RelationComponentAuthority
            .CANDIDATE_INSPIRATION
        ),
        provenance=(
            RelationComponentProvenance(
                source_kind="candidate_unit",
                source_id=unit_id,
                source_path_id=str(
                    source_path_id
                ),
            )
        ),
        accepted_pattern_id=None,
        candidate_unit_id=unit_id,
        candidate_unit_score=float(
            candidate_unit_score
        ),
        exploration_score=float(
            exploration_score
        ),
    )


def _endpoint_atoms(
    text: str,
) -> tuple[str, ...]:
    atoms = []
    seen = set()

    for raw in _ENDPOINT_ATOM_SPLIT_RE.split(
        str(text)
    ):
        atom = " ".join(
            raw.split()
        ).strip()

        if not atom:
            continue

        if not _endpoint_token_signatures(
            atom
        ):
            continue

        key = atom.casefold()
        if key in seen:
            continue

        seen.add(key)
        atoms.append(atom)

    return tuple(atoms)


def _normalize_endpoint_dashes(
    text: str,
) -> str:
    normalized = str(text)

    for dash in _ENDPOINT_DASHES:
        normalized = normalized.replace(
            dash,
            "-",
        )

    return normalized


def _endpoint_token_signatures(
    text: str,
) -> tuple[frozenset[str], ...]:
    """
    Generate domain-neutral orthographic token signatures.

    Both hyphen-split and intra-token-hyphen-collapsed signatures are
    retained, so e.g. "inter-particle" can match "interparticle" without
    asserting any scientific synonymy.
    """

    normalized = _normalize_endpoint_dashes(
        text
    )

    variants = [
        lexical_tokens(normalized),
        lexical_tokens(
            re.sub(
                r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])",
                "",
                normalized,
            )
        ),
    ]

    rows = []
    seen = set()

    for tokens in variants:
        if not tokens:
            continue
        key = tuple(sorted(tokens))
        if key in seen:
            continue
        seen.add(key)
        rows.append(tokens)

    return tuple(rows)


def _best_signature_overlap(
    left: str,
    right: str,
) -> tuple[
    frozenset[str],
    frozenset[str],
    frozenset[str],
    float,
    float,
]:
    rows = []

    for left_tokens in _endpoint_token_signatures(
        left
    ):
        for right_tokens in _endpoint_token_signatures(
            right
        ):
            overlap = (
                left_tokens
                & right_tokens
            )
            task_coverage = (
                len(overlap)
                / max(
                    len(left_tokens),
                    1,
                )
            )
            slot_coverage = (
                len(overlap)
                / max(
                    len(right_tokens),
                    1,
                )
            )
            rows.append(
                (
                    len(overlap),
                    task_coverage,
                    slot_coverage,
                    tuple(sorted(left_tokens)),
                    tuple(sorted(right_tokens)),
                    left_tokens,
                    right_tokens,
                    overlap,
                )
            )

    if not rows:
        return (
            frozenset(),
            frozenset(),
            frozenset(),
            0.0,
            0.0,
        )

    rows.sort(
        key=lambda row: (
            -row[0],
            -row[1],
            -row[2],
            row[3],
            row[4],
        )
    )

    best = rows[0]
    return (
        best[5],
        best[6],
        best[7],
        float(best[1]),
        float(best[2]),
    )


def _endpoint_text_exact(
    left: str,
    right: str,
) -> bool:
    left_signatures = (
        _endpoint_token_signatures(
            left
        )
    )
    right_signatures = (
        _endpoint_token_signatures(
            right
        )
    )

    return any(
        left_tokens == right_tokens
        for left_tokens
        in left_signatures
        for right_tokens
        in right_signatures
    )


def _endpoint_equivalence_witness(
    *,
    task_endpoint: str,
    slot_endpoint: str,
    witnesses: Sequence[
        EndpointEquivalenceWitness
    ],
) -> EndpointEquivalenceWitness | None:
    for witness in witnesses:
        forward = (
            _endpoint_text_exact(
                task_endpoint,
                witness.left_endpoint,
            )
            and
            _endpoint_text_exact(
                slot_endpoint,
                witness.right_endpoint,
            )
        )
        reverse = (
            _endpoint_text_exact(
                task_endpoint,
                witness.right_endpoint,
            )
            and
            _endpoint_text_exact(
                slot_endpoint,
                witness.left_endpoint,
            )
        )

        if forward or reverse:
            return witness

    return None


def resolve_mediator_equivalence_witness(
    *,
    left_mediator: str,
    right_mediator: str,
    witnesses: Sequence[
        MediatorEquivalenceWitness
    ],
) -> MediatorEquivalenceWitness | None:
    """
    Resolve only an explicitly supplied mediator equivalence.

    No embedding similarity, ontology expansion, or lexical-threshold
    relaxation occurs here.
    """

    for witness in witnesses:
        forward = (
            _endpoint_text_exact(
                left_mediator,
                witness.left_mediator,
            )
            and
            _endpoint_text_exact(
                right_mediator,
                witness.right_mediator,
            )
        )
        reverse = (
            _endpoint_text_exact(
                left_mediator,
                witness.right_mediator,
            )
            and
            _endpoint_text_exact(
                right_mediator,
                witness.left_mediator,
            )
        )

        if forward or reverse:
            return witness

    return None


def _slot_text(
    component: RelationComponentView,
    slot: RelationArgumentSlot,
) -> str:
    return (
        component.subject
        if slot == "subject"
        else component.object
    )


def _slot_tokens(
    component: RelationComponentView,
    slot: RelationArgumentSlot,
) -> frozenset[str]:
    return lexical_tokens(
        _slot_text(
            component,
            slot,
        )
    )


def _task_binding(
    *,
    component: RelationComponentView,
    task_endpoint: str,
    endpoint_atoms: tuple[str, ...],
    endpoint_equivalences: Sequence[
        EndpointEquivalenceWitness
    ],
) -> RelationComponentBindingView | None:
    rows = []

    for task_slot, mediator_slot in (
        (
            "subject",
            "object",
        ),
        (
            "object",
            "subject",
        ),
    ):
        slot_text = _slot_text(
            component,
            task_slot,
        )

        mediator_tokens = _slot_tokens(
            component,
            mediator_slot,
        )

        if not mediator_tokens:
            continue

        for atom in endpoint_atoms:
            (
                task_tokens,
                slot_tokens,
                overlap,
                task_coverage,
                slot_coverage,
            ) = _best_signature_overlap(
                atom,
                slot_text,
            )

            exact = bool(
                task_tokens
                and slot_tokens
                and task_tokens == slot_tokens
            )

            witness = None
            if not exact:
                witness = (
                    _endpoint_equivalence_witness(
                        task_endpoint=atom,
                        slot_endpoint=slot_text,
                        witnesses=endpoint_equivalences,
                    )
                )

            if exact:
                authority: EndpointBindingAuthority = (
                    "exact"
                )
            elif witness is not None:
                authority = "equivalent"
            elif overlap:
                authority = "partial"
            else:
                continue

            authority_rank = {
                "exact": 3,
                "equivalent": 2,
                "partial": 1,
            }[authority]

            rows.append(
                (
                    authority_rank,
                    len(overlap),
                    task_coverage,
                    slot_coverage,
                    task_slot,
                    atom.casefold(),
                    RelationComponentBindingView(
                        task_slot=task_slot,
                        mediator_slot=mediator_slot,
                        task_overlap_tokens=sorted(
                            overlap
                        ),
                        mediator_tokens=sorted(
                            mediator_tokens
                        ),
                        binding_authority=authority,
                        matched_endpoint_atom=atom,
                        task_coverage=task_coverage,
                        slot_coverage=slot_coverage,
                        equivalence_witness_id=(
                            None
                            if witness is None
                            else witness.witness_id
                        ),
                    ),
                )
            )

    if not rows:
        return None

    rows.sort(
        key=lambda row: (
            -row[0],
            -row[1],
            -row[2],
            -row[3],
            row[4],
            row[5],
        )
    )

    if (
        len(rows) > 1
        and rows[0][0:4]
        == rows[1][0:4]
        and rows[0][4] != rows[1][4]
    ):
        # Fail closed when equally strong task anchoring is ambiguous
        # between relation subject and object slots.
        return None

    return rows[0][6]


def topology_has_materializable_endpoint_fidelity(
    topology: RelationalTopologyView,
) -> bool:
    """
    EXACT and explicitly witnessed EQUIVALENT endpoint bindings may
    proceed toward production materialization. PARTIAL remains
    diagnostic-only.
    """

    eligible = {
        "exact",
        "equivalent",
    }

    return bool(
        topology.source_binding.binding_authority
        in eligible
        and
        topology.target_binding.binding_authority
        in eligible
    )


def _mediator_compatibility(
    *,
    source_tokens: frozenset[str],
    target_tokens: frozenset[str],
) -> tuple[
    frozenset[str],
    float,
] | None:
    shared = (
        source_tokens
        & target_tokens
    )

    if not shared:
        return None

    if source_tokens == target_tokens:
        return (
            shared,
            1.0,
        )

    # A one-token mediator is accepted only when both normalized argument
    # slots are exactly that token. This prevents generic one-token
    # pseudo-bridges such as unrelated "size" relations.
    if (
        len(source_tokens) == 1
        or len(target_tokens) == 1
    ):
        return None

    source_coverage = (
        len(shared)
        / len(source_tokens)
    )
    target_coverage = (
        len(shared)
        / len(target_tokens)
    )

    if (
        len(shared) < 2
        or min(
            source_coverage,
            target_coverage,
        ) < 0.40
    ):
        return None

    union = (
        source_tokens
        | target_tokens
    )
    jaccard = (
        len(shared)
        / len(union)
    )

    # Normalized diagnostic compatibility in [0, 1].
    # Exact mediator identity remains the maximum 1.0 above.
    score = (
        0.50 * jaccard
        + 0.25 * source_coverage
        + 0.25 * target_coverage
    )

    return (
        shared,
        float(score),
    )


def _mediator_compatibility_with_witness(
    *,
    source_text: str,
    target_text: str,
    source_tokens: frozenset[str],
    target_tokens: frozenset[str],
    mediator_equivalences: Sequence[
        MediatorEquivalenceWitness
    ],
) -> tuple[
    frozenset[str],
    float,
    MediatorEquivalenceWitness | None,
] | None:
    """
    Preserve the frozen lexical gate, then try explicit equivalence only.

    A witnessed equivalent mediator is represented downstream with tokens from
    its explicit canonical_mediator so existing materialization still has a
    non-empty shared mediator representation.
    """

    lexical = _mediator_compatibility(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )

    if lexical is not None:
        shared, score = lexical
        return (
            shared,
            score,
            None,
        )

    witness = resolve_mediator_equivalence_witness(
        left_mediator=source_text,
        right_mediator=target_text,
        witnesses=mediator_equivalences,
    )

    if witness is None:
        return None

    canonical_tokens = lexical_tokens(
        witness.canonical_mediator
    )

    if not canonical_tokens:
        raise RuntimeError(
            "validated mediator-equivalence witness lost canonical tokens"
        )

    return (
        canonical_tokens,
        1.0,
        witness,
    )


def compose_relation_component_topologies(
    *,
    components: Sequence[
        RelationComponentView
    ],
    requested_source: str,
    requested_target: str,
    max_topologies: int = 12,
    endpoint_equivalences: Sequence[
        EndpointEquivalenceWitness
    ] = (),
    mediator_equivalences: Sequence[
        MediatorEquivalenceWitness
    ] = (),
    require_confirmed_known: bool = False,
    require_candidate_anchor: bool = False,
    require_endpoint_fidelity: bool = False,
) -> tuple[
    RelationalTopologyView,
    ...,
]:
    """
    Compose role-aware relation components into task-conditioned topologies.

    Eligibility is structural:
      1. task source and target must anchor to explicit subject/object slots;
      2. the opposite argument slots are the mediator slots;
      3. mediator compatibility is checked only between those slots;
      4. the frozen lexical mediator gate is tried first;
      5. if lexical compatibility fails, only an explicit mediator-equivalence
         witness may authorize the bridge;
      6. lexical relation-wide overlap is not an eligibility condition.

    Component authority is preserved. A composed topology is always
    inspiration-only and always requires verification; this function does
    not create novelty or positive-evidence authority.
    """

    if max_topologies < 1:
        raise ValueError(
            "max_topologies must be >= 1"
        )

    source_atoms = _endpoint_atoms(
        requested_source
    )
    target_atoms = _endpoint_atoms(
        requested_target
    )

    if not source_atoms:
        raise ValueError(
            "requested_source produced no endpoint atoms"
        )

    if not target_atoms:
        raise ValueError(
            "requested_target produced no endpoint atoms"
        )

    source_rows = []
    target_rows = []

    for component in components:
        source_binding = _task_binding(
            component=component,
            task_endpoint=requested_source,
            endpoint_atoms=source_atoms,
            endpoint_equivalences=(
                endpoint_equivalences
            ),
        )

        if source_binding is not None:
            source_rows.append(
                (
                    component,
                    source_binding,
                )
            )

        target_binding = _task_binding(
            component=component,
            task_endpoint=requested_target,
            endpoint_atoms=target_atoms,
            endpoint_equivalences=(
                endpoint_equivalences
            ),
        )

        if target_binding is not None:
            target_rows.append(
                (
                    component,
                    target_binding,
                )
            )

    rows = []

    for (
        source_component,
        source_binding,
    ) in source_rows:
        source_mediator = frozenset(
            source_binding.mediator_tokens
        )
        source_mediator_text = _slot_text(
            source_component,
            source_binding.mediator_slot,
        )

        for (
            target_component,
            target_binding,
        ) in target_rows:
            if (
                source_component.component_id
                == target_component.component_id
            ):
                continue

            if (
                require_confirmed_known
                and
                source_component.authority
                != RelationComponentAuthority.CONFIRMED_KNOWN
                and
                target_component.authority
                != RelationComponentAuthority.CONFIRMED_KNOWN
            ):
                continue

            if (
                require_candidate_anchor
                and
                source_component.authority
                != RelationComponentAuthority.CANDIDATE_INSPIRATION
                and
                target_component.authority
                != RelationComponentAuthority.CANDIDATE_INSPIRATION
            ):
                continue
            if (
                require_endpoint_fidelity
                and
                (
                    source_binding.binding_authority
                    == "partial"
                    or
                    target_binding.binding_authority
                    == "partial"
                )
            ):
                continue

            target_mediator = frozenset(
                target_binding.mediator_tokens
            )
            target_mediator_text = _slot_text(
                target_component,
                target_binding.mediator_slot,
            )

            compatibility = (
                _mediator_compatibility_with_witness(
                    source_text=source_mediator_text,
                    target_text=target_mediator_text,
                    source_tokens=source_mediator,
                    target_tokens=target_mediator,
                    mediator_equivalences=(
                        mediator_equivalences
                    ),
                )
            )

            if compatibility is None:
                continue

            (
                shared,
                score,
                mediator_witness,
            ) = compatibility

            if mediator_witness is None:
                # Preserve all pre-S25e lexical topology IDs exactly.
                topology_id = _stable_id(
                    "relational_topology",
                    source_component.component_id,
                    source_binding.task_slot,
                    source_binding.mediator_slot,
                    target_component.component_id,
                    target_binding.task_slot,
                    target_binding.mediator_slot,
                    *sorted(shared),
                )
            else:
                topology_id = _stable_id(
                    "relational_topology",
                    source_component.component_id,
                    source_binding.task_slot,
                    source_binding.mediator_slot,
                    target_component.component_id,
                    target_binding.task_slot,
                    target_binding.mediator_slot,
                    "mediator_equivalence",
                    mediator_witness.witness_id,
                    *sorted(shared),
                )

            rows.append(
                RelationalTopologyView(
                    topology_id=(
                        topology_id
                    ),
                    source_component=(
                        source_component
                    ),
                    target_component=(
                        target_component
                    ),
                    source_binding=(
                        source_binding
                    ),
                    target_binding=(
                        target_binding
                    ),
                    shared_mediator_tokens=sorted(
                        shared
                    ),
                    compatibility_score=(
                        score
                    ),
                    reason_codes=[
                        (
                            "source_authority:"
                            + source_component
                            .authority
                            .value
                        ),
                        (
                            "target_authority:"
                            + target_component
                            .authority
                            .value
                        ),
                        "task_argument_slots_bound",
                        (
                            "source_endpoint_binding:"
                            + source_binding.binding_authority
                        ),
                        (
                            "target_endpoint_binding:"
                            + target_binding.binding_authority
                        ),
                        *(
                            [
                                "source_endpoint_equivalence_witness:"
                                + str(
                                    source_binding
                                    .equivalence_witness_id
                                )
                            ]
                            if (
                                source_binding
                                .equivalence_witness_id
                                is not None
                            )
                            else []
                        ),
                        *(
                            [
                                "target_endpoint_equivalence_witness:"
                                + str(
                                    target_binding
                                    .equivalence_witness_id
                                )
                            ]
                            if (
                                target_binding
                                .equivalence_witness_id
                                is not None
                            )
                            else []
                        ),
                        "endpoint_binding_fidelity_preserved",
                        "mediator_argument_slots_compatible",
                        *(
                            [
                                (
                                    "mediator_equivalence_witness:"
                                    + mediator_witness.witness_id
                                ),
                                (
                                    "mediator_equivalence_kind:"
                                    + mediator_witness.witness_kind
                                ),
                                "mediator_argument_slots_equivalent",
                                "mediator_canonical_tokens_from_explicit_witness",
                            ]
                            if mediator_witness is not None
                            else []
                        ),
                        "topology_inspiration_only",
                        "topology_requires_verification",
                    ],
                )
            )

    rows.sort(
        key=lambda row: (
            -len(
                row.shared_mediator_tokens
            ),
            -row.compatibility_score,
            row.source_component.component_id,
            row.target_component.component_id,
            row.topology_id,
        )
    )

    seen = set()
    selected = []

    for row in rows:
        if row.topology_id in seen:
            continue

        seen.add(
            row.topology_id
        )
        selected.append(
            row
        )

        if (
            len(selected)
            >= max_topologies
        ):
            break

    return tuple(
        selected
    )


def _legacy_role_binding(
    binding: RelationComponentBindingView,
) -> RelationRoleBindingView:
    return RelationRoleBindingView(
        task_subject_tokens=(
            list(
                binding.task_overlap_tokens
            )
            if binding.task_slot == "subject"
            else []
        ),
        task_predicate_tokens=[],
        task_object_tokens=(
            list(
                binding.task_overlap_tokens
            )
            if binding.task_slot == "object"
            else []
        ),
        mediator_subject_tokens=(
            list(
                binding.mediator_tokens
            )
            if binding.mediator_slot == "subject"
            else []
        ),
        mediator_predicate_tokens=[],
        mediator_object_tokens=(
            list(
                binding.mediator_tokens
            )
            if binding.mediator_slot == "object"
            else []
        ),
    )


def topology_candidate_anchor_unit_id(
    topology: RelationalTopologyView,
) -> str | None:
    """
    Return a real quality-eligible candidate unit that can anchor the
    existing DiscoveryAxis provenance contract.

    CONFIRMED_KNOWN components are never converted into fake candidates.
    """

    for component in (
        topology.source_component,
        topology.target_component,
    ):
        if (
            component.authority
            == RelationComponentAuthority.CANDIDATE_INSPIRATION
            and component.candidate_unit_id
        ):
            return (
                component.candidate_unit_id
            )

    return None


def topology_to_task_bridge_composite(
    topology: RelationalTopologyView,
) -> TaskBridgeCompositeCandidate:
    """
    Bridge the S22a topology contract into the existing task-axis
    materializer without erasing component authority.

    A topology with no candidate participant is intentionally not
    materializable under the current DiscoveryAxis provenance schema.
    It remains a valid composition result for diagnostics/future axis
    contracts, but this function fails closed rather than fabricating a
    candidate inspiration.
    """

    anchor_unit_id = (
        topology_candidate_anchor_unit_id(
            topology
        )
    )

    if anchor_unit_id is None:
        raise ValueError(
            "Relational topology has no candidate provenance anchor"
        )

    if not topology_has_materializable_endpoint_fidelity(
        topology
    ):
        raise ValueError(
            "Relational topology has only partial endpoint binding; "
            "production materialization requires exact or explicitly "
            "witnessed equivalent endpoint fidelity"
        )

    source = (
        topology.source_component
    )
    target = (
        topology.target_component
    )

    return TaskBridgeCompositeCandidate(
        composite_id=(
            topology.topology_id
        ),
        source_unit_id=(
            source.provenance.source_id
        ),
        target_unit_id=(
            target.provenance.source_id
        ),
        source_overlap_tokens=list(
            topology
            .source_binding
            .task_overlap_tokens
        ),
        target_overlap_tokens=list(
            topology
            .target_binding
            .task_overlap_tokens
        ),
        source_mediator_tokens=list(
            topology
            .source_binding
            .mediator_tokens
        ),
        target_mediator_tokens=list(
            topology
            .target_binding
            .mediator_tokens
        ),
        shared_mediator_tokens=list(
            topology
            .shared_mediator_tokens
        ),
        source_relation=(
            CandidateRelationView(
                unit_id=(
                    source
                    .provenance
                    .source_id
                ),
                label=source.label,
                proposed_subject=(
                    source.subject
                ),
                proposed_relation=(
                    source.relation
                ),
                proposed_object=(
                    source.object
                ),
            )
        ),
        target_relation=(
            CandidateRelationView(
                unit_id=(
                    target
                    .provenance
                    .source_id
                ),
                label=target.label,
                proposed_subject=(
                    target.subject
                ),
                proposed_relation=(
                    target.relation
                ),
                proposed_object=(
                    target.object
                ),
            )
        ),
        source_role_binding=(
            _legacy_role_binding(
                topology.source_binding
            )
        ),
        target_role_binding=(
            _legacy_role_binding(
                topology.target_binding
            )
        ),
        compatibility_score=(
            topology.compatibility_score
        ),
        epistemic_status="inspiration_only",
        requires_verification=True,
        reason_codes=[
            *topology.reason_codes,
            (
                "source_component_source_kind:"
                + source.provenance.source_kind
            ),
            (
                "source_component_source_id:"
                + source.provenance.source_id
            ),
            (
                "target_component_source_kind:"
                + target.provenance.source_kind
            ),
            (
                "target_component_source_id:"
                + target.provenance.source_id
            ),
            *(
                [
                    "source_component_paper_id:"
                    + source.provenance.paper_id,
                    "source_component_chunk_id:"
                    + source.provenance.chunk_id,
                    "source_component_document_id:"
                    + source.provenance.document_id,
                ]
                if (
                    source.authority
                    == RelationComponentAuthority.CONFIRMED_KNOWN
                )
                else [
                    "source_component_source_path_id:"
                    + source.provenance.source_path_id,
                ]
            ),
            *(
                [
                    "target_component_paper_id:"
                    + target.provenance.paper_id,
                    "target_component_chunk_id:"
                    + target.provenance.chunk_id,
                    "target_component_document_id:"
                    + target.provenance.document_id,
                ]
                if (
                    target.authority
                    == RelationComponentAuthority.CONFIRMED_KNOWN
                )
                else [
                    "target_component_source_path_id:"
                    + target.provenance.source_path_id,
                ]
            ),
        ],
        source_component_id=(
            source.component_id
        ),
        target_component_id=(
            target.component_id
        ),
        source_component_authority=(
            source.authority.value
        ),
        target_component_authority=(
            target.authority.value
        ),
        provenance_candidate_unit_id=(
            anchor_unit_id
        ),
        topology_id=(
            topology.topology_id
        ),
        composition_mode=(
            "relation_component_topology_v1"
        ),
    )
