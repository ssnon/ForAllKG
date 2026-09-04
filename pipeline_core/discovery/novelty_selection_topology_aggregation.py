from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

from pipeline_core.discovery.external_novelty_contracts import (
    NoveltySelectionRole,
)
from pipeline_core.discovery.novelty_selection_aggregation import (
    NonobviousnessOutcome,
    RoleAwareAggregation,
    RoleAwareAtomicClaim,
    aggregate_role_aware_nonobviousness,
)


_ALLOWED_ROLES = frozenset(
    get_args(NoveltySelectionRole)
)

_ALLOWED_OUTCOMES = frozenset(
    get_args(NonobviousnessOutcome)
)


@dataclass(frozen=True)
class TopologyAwareAtomicClaim:
    """Atomic outcome plus outcome-blind role and claim topology."""

    claim_id: str
    claim_kind: str
    novelty_selection_role: NoveltySelectionRole
    nonobviousness_outcome: NonobviousnessOutcome

    higher_order_relation_basis: tuple[str, ...] = ()
    higher_order_component_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopologyAwareAggregation:
    """Shadow-only topology-aware selection result.

    Topology does not rewrite roles.

    In particular, a NOVELTY_BEARING claim remains independently
    selection-relevant even when it is also a component of a composite
    novelty claim.

    Conversely, a REQUIRED_ENABLING_RELATION does not become
    novelty-bearing merely because it is a composite component.
    """

    selection_class: str
    action: str
    positive_nonobviousness_authority: bool

    novelty_bearing_claim_ids: tuple[str, ...]
    required_enabling_claim_ids: tuple[str, ...]
    testing_prediction_claim_ids: tuple[str, ...]
    auxiliary_claim_ids: tuple[str, ...]

    composite_claim_ids: tuple[str, ...]
    topology_edges: tuple[tuple[str, str], ...]
    nested_novelty_bearing_component_ids: tuple[str, ...]

    blocking_claim_ids: tuple[str, ...]
    unresolved_claim_ids: tuple[str, ...]
    structurally_unresolved_claim_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


def _validate_claims(
    claims: tuple[TopologyAwareAtomicClaim, ...],
) -> dict[str, TopologyAwareAtomicClaim]:
    by_id: dict[str, TopologyAwareAtomicClaim] = {}

    for claim in claims:
        claim_id = str(claim.claim_id).strip()

        if not claim_id:
            raise ValueError(
                "topology-aware claim requires claim_id"
            )

        if claim_id in by_id:
            raise ValueError(
                "duplicate topology-aware claim_id: "
                + claim_id
            )

        if (
            claim.novelty_selection_role
            not in _ALLOWED_ROLES
        ):
            raise ValueError(
                "unsupported novelty selection role: "
                + str(
                    claim.novelty_selection_role
                )
            )

        if (
            claim.nonobviousness_outcome
            not in _ALLOWED_OUTCOMES
        ):
            raise ValueError(
                "unsupported nonobviousness outcome: "
                + str(
                    claim.nonobviousness_outcome
                )
            )

        component_ids = tuple(
            str(value).strip()
            for value
            in claim.higher_order_component_claim_ids
            if str(value).strip()
        )

        if len(component_ids) != len(
            set(component_ids)
        ):
            raise ValueError(
                "duplicate topology component reference: "
                + claim_id
            )

        if (
            claim.claim_kind != "composite"
            and component_ids
        ):
            raise ValueError(
                "non-composite topology-aware claim "
                "cannot declare components: "
                + claim_id
            )

        if claim_id in component_ids:
            raise ValueError(
                "composite topology cannot self-reference: "
                + claim_id
            )

        by_id[claim_id] = claim

    for claim in claims:
        for component_id in (
            claim.higher_order_component_claim_ids
        ):
            if component_id not in by_id:
                raise ValueError(
                    "unknown topology component claim_id: "
                    + str(component_id)
                )

    return by_id


def _validate_acyclic_topology(
    claims: tuple[TopologyAwareAtomicClaim, ...],
) -> None:
    graph = {
        claim.claim_id: tuple(
            claim.higher_order_component_claim_ids
        )
        for claim in claims
    }

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(claim_id: str) -> None:
        if claim_id in visited:
            return

        if claim_id in visiting:
            raise ValueError(
                "cyclic higher-order component topology"
            )

        visiting.add(claim_id)

        for component_id in graph.get(
            claim_id,
            (),
        ):
            visit(component_id)

        visiting.remove(claim_id)
        visited.add(claim_id)

    for claim_id in graph:
        visit(claim_id)


def _base_aggregation(
    claims: tuple[TopologyAwareAtomicClaim, ...],
) -> RoleAwareAggregation:
    return aggregate_role_aware_nonobviousness(
        tuple(
            RoleAwareAtomicClaim(
                claim_id=claim.claim_id,
                novelty_selection_role=(
                    claim.novelty_selection_role
                ),
                nonobviousness_outcome=(
                    claim.nonobviousness_outcome
                ),
            )
            for claim in claims
        )
    )


def aggregate_topology_aware_nonobviousness(
    claims: tuple[TopologyAwareAtomicClaim, ...],
) -> TopologyAwareAggregation:
    """Apply role-aware semantics while validating higher-order topology.

    Policy invariants:

    1. Topology NEVER demotes NOVELTY_BEARING to an enabling role.
    2. Topology NEVER promotes an enabling/testing/auxiliary claim.
    3. Nested NOVELTY_BEARING components remain independently
       selection-relevant because their role says they independently
       carry hypothesis distinctiveness.
    4. Saturated/routine REQUIRED_ENABLING_RELATION claims remain
       nonblocking under the existing role-aware policy.
    5. A composite claim cannot contribute positive authority unless
       explicit higher-order hypothesis provenance was preserved.
    6. Missing composite provenance never becomes novelty.
    """

    by_id = _validate_claims(claims)
    _validate_acyclic_topology(claims)

    base = _base_aggregation(claims)

    composites = tuple(
        claim
        for claim in claims
        if claim.claim_kind == "composite"
    )

    composite_ids = tuple(
        claim.claim_id
        for claim in composites
    )

    topology_edges = tuple(
        (
            claim.claim_id,
            component_id,
        )
        for claim in composites
        for component_id
        in claim.higher_order_component_claim_ids
    )

    component_id_set = {
        component_id
        for _, component_id
        in topology_edges
    }

    nested_novelty = tuple(
        claim.claim_id
        for claim in claims
        if (
            claim.claim_id in component_id_set
            and claim.novelty_selection_role
            == "NOVELTY_BEARING"
        )
    )

    structurally_unresolved = tuple(
        claim.claim_id
        for claim in composites
        if (
            claim.novelty_selection_role
            == "NOVELTY_BEARING"
            and not tuple(
                value
                for value
                in claim.higher_order_relation_basis
                if str(value).strip()
            )
        )
    )

    reason_codes = list(
        base.reason_codes
    )

    if nested_novelty:
        reason_codes.append(
            "nested_novelty_bearing_components_remain_selection_relevant"
        )

    if structurally_unresolved:
        reason_codes.append(
            "novelty_bearing_composite_missing_explicit_source_basis"
        )

        for claim_id in structurally_unresolved:
            reason_codes.append(
                "higher_order_claim_structurally_unresolved:"
                + claim_id
            )

    # Structural uncertainty can only preserve or downgrade authority.
    # It must never rescue an already INELIGIBLE base result.
    if (
        structurally_unresolved
        and base.selection_class
        != "INELIGIBLE"
    ):
        selection_class = "CONDITIONAL"
        action = (
            "REFINE_HIGHER_ORDER_RELATION_SPECIFICATION"
        )
        positive = False
    else:
        selection_class = base.selection_class
        action = base.action
        positive = (
            base.positive_nonobviousness_authority
            and not structurally_unresolved
        )

    return TopologyAwareAggregation(
        selection_class=selection_class,
        action=action,
        positive_nonobviousness_authority=positive,
        novelty_bearing_claim_ids=(
            base.novelty_bearing_claim_ids
        ),
        required_enabling_claim_ids=(
            base.required_enabling_claim_ids
        ),
        testing_prediction_claim_ids=(
            base.testing_prediction_claim_ids
        ),
        auxiliary_claim_ids=(
            base.auxiliary_claim_ids
        ),
        composite_claim_ids=composite_ids,
        topology_edges=topology_edges,
        nested_novelty_bearing_component_ids=(
            nested_novelty
        ),
        blocking_claim_ids=(
            base.blocking_claim_ids
        ),
        unresolved_claim_ids=(
            base.unresolved_claim_ids
        ),
        structurally_unresolved_claim_ids=(
            structurally_unresolved
        ),
        reason_codes=tuple(
            dict.fromkeys(
                reason_codes
            )
        ),
    )
