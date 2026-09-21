from __future__ import annotations

import hashlib
import re
from typing import Literal, Sequence, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.relation_component_composition import (
    EndpointEquivalenceWitness,
    MediatorEquivalenceWitness,
    RelationComponentAuthority,
    RelationComponentBindingView,
    RelationComponentView,
    RelationalTopologyView,
    _endpoint_atoms,
    _mediator_compatibility_with_witness,
    _slot_text,
    _slot_tokens,
    _task_binding,
    lexical_tokens,
    topology_has_materializable_endpoint_fidelity,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


StructuredEndpointBindingMode = Literal[
    "legacy_exact_or_equivalent",
    "token_containment",
    "coordinated_head_facet",
]


EndpointFacetCoverageStatus = Literal[
    "core_facet_unresolved",
    "facet_core_supported_no_qualifier_obligation",
    "facet_and_qualifier_locally_supported",
    "facet_supported_qualifier_only_component_level",
    "facet_core_supported_qualifier_unresolved",
]


EndpointCoverageStatus = Literal[
    "not_applicable",
    "complete",
    "partial_or_qualifier_unresolved",
    "unresolved",
]


class TaskEndpointFacetCoverageView(StrictModel):
    schema_version: str = "task-endpoint-facet-coverage-v1"

    core_endpoint: str
    facet_tokens: list[str] = Field(default_factory=list)
    shared_qualifier_tokens: list[str] = Field(default_factory=list)

    core_binding_count: int = 0
    task_slot_qualified_binding_count: int = 0
    component_qualified_binding_count: int = 0

    core_binding_component_ids: list[str] = Field(default_factory=list)
    task_slot_qualified_component_ids: list[str] = Field(default_factory=list)
    component_qualified_component_ids: list[str] = Field(default_factory=list)

    status: EndpointFacetCoverageStatus

    diagnostic_only: Literal[True] = True
    coverage_authority: Literal[False] = False
    scientific_equivalence_asserted: Literal[False] = False


class TaskEndpointCoverageLedger(StrictModel):
    schema_version: str = "task-endpoint-coverage-ledger-v1"

    original_endpoint: str
    coordinated: bool
    shared_qualifier_tokens: list[str] = Field(default_factory=list)
    head_tokens: list[str] = Field(default_factory=list)
    facet_tokens: list[list[str]] = Field(default_factory=list)
    facets: list[TaskEndpointFacetCoverageView] = Field(default_factory=list)

    status: EndpointCoverageStatus

    diagnostic_only: Literal[True] = True
    coverage_authority: Literal[False] = False
    task_filter_relaxed: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    scientific_equivalence_asserted: Literal[False] = False


class TaskBackboneChainView(StrictModel):
    """
    Confirmed-known three-component task backbone.

    The chain is structural composition authority only. It creates no
    positive-premise, interaction, or novelty authority.
    """

    schema_version: str = "task-backbone-chain-v1"

    topology_id: str

    source_component: RelationComponentView
    middle_component: RelationComponentView
    target_component: RelationComponentView

    source_binding: RelationComponentBindingView
    target_binding: RelationComponentBindingView

    source_binding_mode: StructuredEndpointBindingMode
    target_binding_mode: StructuredEndpointBindingMode

    left_shared_mediator_tokens: list[str] = Field(default_factory=list)
    right_shared_mediator_tokens: list[str] = Field(default_factory=list)

    left_compatibility_score: float = 0.0
    right_compatibility_score: float = 0.0

    left_mediator_equivalence_witness_id: str | None = None
    right_mediator_equivalence_witness_id: str | None = None

    epistemic_status: Literal["inspiration_only"] = "inspiration_only"
    requires_verification: Literal[True] = True
    novelty_authority: Literal[False] = False

    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_chain(self) -> "TaskBackboneChainView":
        ids = [
            self.source_component.component_id,
            self.middle_component.component_id,
            self.target_component.component_id,
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "task backbone chain requires three distinct components"
            )

        if any(
            component.authority
            != RelationComponentAuthority.CONFIRMED_KNOWN
            for component in (
                self.source_component,
                self.middle_component,
                self.target_component,
            )
        ):
            raise ValueError(
                "task backbone chain requires CONFIRMED_KNOWN components"
            )

        allowed = {"exact", "equivalent", "structured"}
        if self.source_binding.binding_authority not in allowed:
            raise ValueError(
                "task backbone source endpoint lacks production-grade fidelity"
            )
        if self.target_binding.binding_authority not in allowed:
            raise ValueError(
                "task backbone target endpoint lacks production-grade fidelity"
            )

        for value in (
            self.left_compatibility_score,
            self.right_compatibility_score,
        ):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(
                    "task backbone compatibility score must be within [0, 1]"
                )

        return self

    @property
    def components(self) -> tuple[RelationComponentView, ...]:
        return (
            self.source_component,
            self.middle_component,
            self.target_component,
        )


TaskBackboneLike: TypeAlias = (
    RelationalTopologyView | TaskBackboneChainView
)


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return prefix + ":" + hashlib.sha256(raw).hexdigest()[:20]


def _append_unique(values: list[str], value: str) -> None:
    normalized = " ".join(str(value).split()).strip()
    if not normalized:
        return
    key = normalized.casefold()
    if all(existing.casefold() != key for existing in values):
        values.append(normalized)


def _surface_tokens(text: str) -> list[str]:
    return [
        value.casefold()
        for value in re.findall(r"[A-Za-z0-9]+", str(text))
    ]


def _coordinated_head_facet_spec(
    text: str,
) -> tuple[frozenset[str], tuple[frozenset[str], ...]] | None:
    parts = [
        " ".join(part.split()).strip()
        for part in re.split(
            r"\s*(?:,|\band\b|\bor\b)\s*",
            str(text),
            flags=re.IGNORECASE,
        )
        if " ".join(part.split()).strip()
    ]
    if len(parts) < 2:
        return None

    first_surface = _surface_tokens(parts[0])
    if len(first_surface) < 2:
        return None

    later = [_surface_tokens(part) for part in parts[1:]]
    if any(len(tokens) != 1 for tokens in later):
        return None

    head_tokens = lexical_tokens(first_surface[-2])
    first_facet = lexical_tokens(first_surface[-1])
    if not head_tokens or not first_facet:
        return None

    facets = [first_facet]
    for tokens in later:
        facet = lexical_tokens(tokens[0])
        if not facet:
            return None
        facets.append(facet)

    return head_tokens, tuple(facets)


def _coordinated_endpoint_coverage_spec(
    text: str,
) -> tuple[
    tuple[str, ...],
    frozenset[str],
    tuple[frozenset[str], ...],
] | None:
    parts = [
        " ".join(part.split()).strip()
        for part in re.split(
            r"\s*(?:,|\band\b|\bor\b)\s*",
            str(text),
            flags=re.IGNORECASE,
        )
        if " ".join(part.split()).strip()
    ]
    if len(parts) < 2:
        return None

    first_surface = _surface_tokens(parts[0])
    later = [_surface_tokens(part) for part in parts[1:]]

    if (
        len(first_surface) < 2
        or any(len(tokens) != 1 for tokens in later)
    ):
        return None

    shared_qualifiers = tuple(first_surface[:-2])
    head_tokens = lexical_tokens(first_surface[-2])
    first_facet = lexical_tokens(first_surface[-1])

    if not head_tokens or not first_facet:
        return None

    facets = [first_facet]
    for tokens in later:
        facet = lexical_tokens(tokens[0])
        if not facet:
            return None
        facets.append(facet)

    return shared_qualifiers, head_tokens, tuple(facets)


def _structured_task_binding(
    *,
    component: RelationComponentView,
    task_endpoint: str,
    endpoint_equivalences: Sequence[EndpointEquivalenceWitness],
) -> tuple[RelationComponentBindingView, StructuredEndpointBindingMode] | None:
    endpoint_atoms = _endpoint_atoms(task_endpoint)

    legacy = _task_binding(
        component=component,
        task_endpoint=task_endpoint,
        endpoint_atoms=endpoint_atoms,
        endpoint_equivalences=endpoint_equivalences,
    )

    coordination = _coordinated_head_facet_spec(task_endpoint)

    if (
        legacy is not None
        and legacy.binding_authority in {"exact", "equivalent"}
        and len(endpoint_atoms) <= 1
    ):
        return legacy, "legacy_exact_or_equivalent"

    # S28 cross-domain whole-endpoint guard:
    # for a multi-atom task endpoint, an exact/equivalent match to only one
    # atom is not whole-endpoint authority. Fall through to the structured
    # whole-endpoint path instead. This changes no thresholds and creates no
    # synonym/equivalence authority.
    rows = []

    for task_slot, mediator_slot in (
        ("subject", "object"),
        ("object", "subject"),
    ):
        slot_text = _slot_text(component, task_slot)
        slot_tokens = _slot_tokens(component, task_slot)
        mediator_tokens = _slot_tokens(component, mediator_slot)

        if not slot_tokens or not mediator_tokens:
            continue

        if coordination is None:
            endpoint_tokens = lexical_tokens(task_endpoint)
            if (
                len(endpoint_tokens) < 2
                or not endpoint_tokens.issubset(slot_tokens)
                or endpoint_tokens == slot_tokens
            ):
                continue

            overlap = set(endpoint_tokens)
            task_coverage = 1.0
            slot_coverage = len(overlap) / max(len(slot_tokens), 1)
            mode: StructuredEndpointBindingMode = "token_containment"
        else:
            head_tokens, facets = coordination
            if not head_tokens.issubset(slot_tokens):
                continue

            matched_facets = [
                facet
                for facet in facets
                if facet.issubset(slot_tokens)
            ]
            if not matched_facets:
                continue

            overlap = set(head_tokens)
            for facet in matched_facets:
                overlap.update(facet)

            task_coverage = (
                1.0 + len(matched_facets)
            ) / (
                1.0 + len(facets)
            )
            slot_coverage = len(overlap) / max(len(slot_tokens), 1)
            mode = "coordinated_head_facet"

        binding = RelationComponentBindingView(
            task_slot=task_slot,
            mediator_slot=mediator_slot,
            task_overlap_tokens=sorted(overlap),
            mediator_tokens=sorted(mediator_tokens),
            binding_authority="structured",
            matched_endpoint_atom=" ".join(str(task_endpoint).split()),
            task_coverage=float(task_coverage),
            slot_coverage=float(slot_coverage),
            equivalence_witness_id=None,
        )
        rows.append(
            (
                float(task_coverage),
                float(slot_coverage),
                task_slot,
                binding,
                mode,
            )
        )

    if not rows:
        return None

    rows.sort(
        key=lambda row: (
            -row[0],
            -row[1],
            row[2],
        )
    )

    if (
        len(rows) > 1
        and rows[0][0:2] == rows[1][0:2]
        and rows[0][2] != rows[1][2]
    ):
        return None

    return rows[0][3], rows[0][4]


def build_task_endpoint_coverage_ledger(
    *,
    components: Sequence[RelationComponentView],
    task_endpoint: str,
    endpoint_equivalences: Sequence[EndpointEquivalenceWitness] = (),
) -> TaskEndpointCoverageLedger:
    # Diagnostic-only coverage ledger for coordinated task endpoints.
    # Compound endpoints are represented as:
    # shared qualifier obligation + head + facet atoms.
    # No endpoint-equivalence, positive-premise, novelty, selection,
    # rejection, task-filter, or backbone-eligibility authority is created.
    original_endpoint = " ".join(str(task_endpoint).split()).strip()
    spec = _coordinated_endpoint_coverage_spec(original_endpoint)

    if spec is None:
        return TaskEndpointCoverageLedger(
            original_endpoint=original_endpoint,
            coordinated=False,
            status="not_applicable",
        )

    shared_qualifiers, head_tokens, facets = spec
    qualifier_tokens = frozenset(
        token
        for value in shared_qualifiers
        for token in lexical_tokens(value)
    )

    known = [
        component
        for component in components
        if component.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
    ]

    facet_rows: list[TaskEndpointFacetCoverageView] = []

    for facet in facets:
        core_endpoint = " ".join(
            [
                *sorted(head_tokens),
                *sorted(facet),
            ]
        )

        core_ids = []
        local_qualified_ids = []
        component_qualified_ids = []

        for component in known:
            resolved = _structured_task_binding(
                component=component,
                task_endpoint=core_endpoint,
                endpoint_equivalences=endpoint_equivalences,
            )
            if resolved is None:
                continue

            binding, _mode = resolved
            core_ids.append(component.component_id)

            if not qualifier_tokens:
                local_qualified_ids.append(component.component_id)
                component_qualified_ids.append(component.component_id)
                continue

            task_slot_tokens = _slot_tokens(
                component,
                binding.task_slot,
            )
            other_slot = (
                "object"
                if binding.task_slot == "subject"
                else "subject"
            )
            component_tokens = frozenset(
                set(task_slot_tokens)
                | set(_slot_tokens(component, other_slot))
            )

            if qualifier_tokens.issubset(task_slot_tokens):
                local_qualified_ids.append(component.component_id)

            if qualifier_tokens.issubset(component_tokens):
                component_qualified_ids.append(component.component_id)

        core_ids = sorted(set(core_ids))
        local_qualified_ids = sorted(set(local_qualified_ids))
        component_qualified_ids = sorted(
            set(component_qualified_ids)
        )

        if not core_ids:
            status: EndpointFacetCoverageStatus = (
                "core_facet_unresolved"
            )
        elif not qualifier_tokens:
            status = "facet_core_supported_no_qualifier_obligation"
        elif local_qualified_ids:
            status = "facet_and_qualifier_locally_supported"
        elif component_qualified_ids:
            status = "facet_supported_qualifier_only_component_level"
        else:
            status = "facet_core_supported_qualifier_unresolved"

        facet_rows.append(
            TaskEndpointFacetCoverageView(
                core_endpoint=core_endpoint,
                facet_tokens=sorted(facet),
                shared_qualifier_tokens=sorted(qualifier_tokens),
                core_binding_count=len(core_ids),
                task_slot_qualified_binding_count=len(
                    local_qualified_ids
                ),
                component_qualified_binding_count=len(
                    component_qualified_ids
                ),
                core_binding_component_ids=core_ids,
                task_slot_qualified_component_ids=local_qualified_ids,
                component_qualified_component_ids=(
                    component_qualified_ids
                ),
                status=status,
            )
        )

    complete_statuses = {
        "facet_core_supported_no_qualifier_obligation",
        "facet_and_qualifier_locally_supported",
    }

    if facet_rows and all(
        row.status in complete_statuses
        for row in facet_rows
    ):
        ledger_status: EndpointCoverageStatus = "complete"
    elif any(row.core_binding_count > 0 for row in facet_rows):
        ledger_status = "partial_or_qualifier_unresolved"
    else:
        ledger_status = "unresolved"

    return TaskEndpointCoverageLedger(
        original_endpoint=original_endpoint,
        coordinated=True,
        shared_qualifier_tokens=sorted(qualifier_tokens),
        head_tokens=sorted(head_tokens),
        facet_tokens=[
            sorted(facet)
            for facet in facets
        ],
        facets=facet_rows,
        status=ledger_status,
    )


def compose_three_component_task_backbones(
    *,
    components: Sequence[RelationComponentView],
    requested_source: str,
    requested_target: str,
    endpoint_equivalences: Sequence[EndpointEquivalenceWitness] = (),
    mediator_equivalences: Sequence[MediatorEquivalenceWitness] = (),
    max_backbones: int = 128,
) -> tuple[TaskBackboneChainView, ...]:
    if max_backbones < 1:
        raise ValueError("max_backbones must be >= 1")

    known = [
        component
        for component in components
        if component.authority
        == RelationComponentAuthority.CONFIRMED_KNOWN
    ]

    source_rows = []
    target_rows = []

    for component in known:
        source = _structured_task_binding(
            component=component,
            task_endpoint=requested_source,
            endpoint_equivalences=endpoint_equivalences,
        )
        if source is not None:
            source_rows.append((component, *source))

        target = _structured_task_binding(
            component=component,
            task_endpoint=requested_target,
            endpoint_equivalences=endpoint_equivalences,
        )
        if target is not None:
            target_rows.append((component, *target))

    rows: list[TaskBackboneChainView] = []

    for source_component, source_binding, source_mode in source_rows:
        source_mediator_text = _slot_text(
            source_component,
            source_binding.mediator_slot,
        )
        source_mediator_tokens = frozenset(
            source_binding.mediator_tokens
        )

        for target_component, target_binding, target_mode in target_rows:
            if source_component.component_id == target_component.component_id:
                continue

            target_mediator_text = _slot_text(
                target_component,
                target_binding.mediator_slot,
            )
            target_mediator_tokens = frozenset(
                target_binding.mediator_tokens
            )

            for middle in known:
                if middle.component_id in {
                    source_component.component_id,
                    target_component.component_id,
                }:
                    continue

                for left_slot, right_slot in (
                    ("subject", "object"),
                    ("object", "subject"),
                ):
                    left_text = _slot_text(middle, left_slot)
                    left_tokens = _slot_tokens(middle, left_slot)
                    right_text = _slot_text(middle, right_slot)
                    right_tokens = _slot_tokens(middle, right_slot)

                    left = _mediator_compatibility_with_witness(
                        source_text=source_mediator_text,
                        target_text=left_text,
                        source_tokens=source_mediator_tokens,
                        target_tokens=left_tokens,
                        mediator_equivalences=mediator_equivalences,
                    )
                    if left is None:
                        continue

                    right = _mediator_compatibility_with_witness(
                        source_text=right_text,
                        target_text=target_mediator_text,
                        source_tokens=right_tokens,
                        target_tokens=target_mediator_tokens,
                        mediator_equivalences=mediator_equivalences,
                    )
                    if right is None:
                        continue

                    left_shared, left_score, left_witness = left
                    right_shared, right_score, right_witness = right

                    topology_id = _stable_id(
                        "task_backbone_chain",
                        source_component.component_id,
                        source_binding.task_slot,
                        middle.component_id,
                        left_slot,
                        right_slot,
                        target_component.component_id,
                        target_binding.task_slot,
                        *sorted(left_shared),
                        *sorted(right_shared),
                    )

                    rows.append(
                        TaskBackboneChainView(
                            topology_id=topology_id,
                            source_component=source_component,
                            middle_component=middle,
                            target_component=target_component,
                            source_binding=source_binding,
                            target_binding=target_binding,
                            source_binding_mode=source_mode,
                            target_binding_mode=target_mode,
                            left_shared_mediator_tokens=sorted(left_shared),
                            right_shared_mediator_tokens=sorted(right_shared),
                            left_compatibility_score=float(left_score),
                            right_compatibility_score=float(right_score),
                            left_mediator_equivalence_witness_id=(
                                None
                                if left_witness is None
                                else left_witness.witness_id
                            ),
                            right_mediator_equivalence_witness_id=(
                                None
                                if right_witness is None
                                else right_witness.witness_id
                            ),
                            reason_codes=[
                                "confirmed_known_three_component_task_backbone",
                                "source_endpoint_fidelity_preserved",
                                "target_endpoint_fidelity_preserved",
                                "partial_endpoint_binding_not_authorized",
                                "internal_mediator_compatibility_preserved",
                                "backbone_composition_authority_only",
                                "backbone_is_not_novelty_evidence",
                            ],
                        )
                    )

    rows.sort(
        key=lambda row: (
            -min(
                row.left_compatibility_score,
                row.right_compatibility_score,
            ),
            row.source_component.component_id,
            row.middle_component.component_id,
            row.target_component.component_id,
            row.topology_id,
        )
    )

    seen = set()
    selected = []
    for row in rows:
        if row.topology_id in seen:
            continue
        seen.add(row.topology_id)
        selected.append(row)
        if len(selected) >= max_backbones:
            break

    return tuple(selected)


def task_backbone_components(
    backbone: TaskBackboneLike,
) -> tuple[RelationComponentView, ...]:
    if isinstance(backbone, TaskBackboneChainView):
        return backbone.components
    return (
        backbone.source_component,
        backbone.target_component,
    )


def task_backbone_component_ids(
    backbone: TaskBackboneLike,
) -> set[str]:
    return {
        component.component_id
        for component in task_backbone_components(backbone)
    }


def task_backbone_has_materializable_endpoint_fidelity(
    backbone: TaskBackboneLike,
) -> bool:
    if isinstance(backbone, TaskBackboneChainView):
        eligible = {"exact", "equivalent", "structured"}
        return (
            backbone.source_binding.binding_authority in eligible
            and backbone.target_binding.binding_authority in eligible
        )
    return topology_has_materializable_endpoint_fidelity(backbone)


def task_backbone_role_vocabulary_for_modifier_screen(
    backbone: TaskBackboneLike,
) -> dict[str, list[str]]:
    roles = {
        "source": [],
        "mediator": [],
        "target": [],
    }

    if isinstance(backbone, TaskBackboneChainView):
        _append_unique(
            roles["source"],
            _slot_text(
                backbone.source_component,
                backbone.source_binding.task_slot,
            ),
        )
        _append_unique(
            roles["mediator"],
            _slot_text(
                backbone.source_component,
                backbone.source_binding.mediator_slot,
            ),
        )
        _append_unique(roles["mediator"], backbone.middle_component.subject)
        _append_unique(roles["mediator"], backbone.middle_component.object)
        _append_unique(
            roles["mediator"],
            _slot_text(
                backbone.target_component,
                backbone.target_binding.mediator_slot,
            ),
        )
        _append_unique(
            roles["target"],
            _slot_text(
                backbone.target_component,
                backbone.target_binding.task_slot,
            ),
        )
        return roles

    _append_unique(
        roles["source"],
        _slot_text(
            backbone.source_component,
            backbone.source_binding.task_slot,
        ),
    )
    _append_unique(
        roles["mediator"],
        _slot_text(
            backbone.source_component,
            backbone.source_binding.mediator_slot,
        ),
    )
    _append_unique(
        roles["mediator"],
        _slot_text(
            backbone.target_component,
            backbone.target_binding.mediator_slot,
        ),
    )
    _append_unique(
        roles["target"],
        _slot_text(
            backbone.target_component,
            backbone.target_binding.task_slot,
        ),
    )
    return roles


def task_backbone_role_aliases_for_composition(
    backbone: TaskBackboneLike,
) -> dict[str, list[str]]:
    aliases = {
        "source": [],
        "mediator": [],
        "target": [],
    }

    _append_unique(
        aliases["source"],
        _slot_text(
            backbone.source_component,
            backbone.source_binding.task_slot,
        ),
    )
    _append_unique(
        aliases["source"],
        backbone.source_binding.matched_endpoint_atom,
    )
    _append_unique(
        aliases["target"],
        _slot_text(
            backbone.target_component,
            backbone.target_binding.task_slot,
        ),
    )
    _append_unique(
        aliases["target"],
        backbone.target_binding.matched_endpoint_atom,
    )

    _append_unique(
        aliases["mediator"],
        _slot_text(
            backbone.source_component,
            backbone.source_binding.mediator_slot,
        ),
    )

    if isinstance(backbone, TaskBackboneChainView):
        _append_unique(aliases["mediator"], backbone.middle_component.subject)
        _append_unique(aliases["mediator"], backbone.middle_component.object)
        _append_unique(
            aliases["mediator"],
            _slot_text(
                backbone.target_component,
                backbone.target_binding.mediator_slot,
            ),
        )
        if backbone.left_shared_mediator_tokens:
            _append_unique(
                aliases["mediator"],
                " ".join(backbone.left_shared_mediator_tokens),
            )
        if backbone.right_shared_mediator_tokens:
            _append_unique(
                aliases["mediator"],
                " ".join(backbone.right_shared_mediator_tokens),
            )
    else:
        _append_unique(
            aliases["mediator"],
            _slot_text(
                backbone.target_component,
                backbone.target_binding.mediator_slot,
            ),
        )
        if backbone.shared_mediator_tokens:
            _append_unique(
                aliases["mediator"],
                " ".join(backbone.shared_mediator_tokens),
            )

    return aliases
