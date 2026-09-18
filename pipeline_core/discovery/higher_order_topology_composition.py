from __future__ import annotations

import hashlib
import re
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.relation_component_composition import (
    RelationArgumentSlot,
    RelationComponentAuthority,
    RelationComponentView,
    RelationalTopologyView,
    topology_has_materializable_endpoint_fidelity,
)
from pipeline_core.discovery.task_backbone_chain import (
    TaskBackboneChainView,
    TaskBackboneLike,
    task_backbone_component_ids,
    task_backbone_components,
    task_backbone_has_materializable_endpoint_fidelity,
    task_backbone_role_aliases_for_composition,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    lexical_tokens,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


HigherOrderAttachmentRole = Literal[
    "source",
    "mediator",
    "target",
]


class ModifierEligibilityWitness(StrictModel):
    """
    Explicit upstream authority that a component is eligible to act as a
    higher-order modifier.

    The higher-order composer does not silently recreate S21L's scientific
    filtering policy. It requires an auditable witness that the upstream
    modifier screen passed all five frozen dimensions.
    """

    schema_version: str = "modifier-eligibility-witness-v2"

    witness_id: str
    modifier_component_id: str

    validated_anchor_role: HigherOrderAttachmentRole
    validated_anchor_slot: RelationArgumentSlot
    modifier_slot: RelationArgumentSlot

    validated_anchor_text: str
    modifier_text: str

    anchor_purity_pass: Literal[True] = True
    source_leakage_pass: Literal[True] = True
    target_leakage_pass: Literal[True] = True
    mediator_leakage_pass: Literal[True] = True
    genericity_pass: Literal[True] = True
    directional_role_pass: Literal[True] = True

    provenance_ids: list[str] = Field(
        default_factory=list
    )
    reason_codes: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_witness(
        self,
    ) -> "ModifierEligibilityWitness":
        if not str(self.witness_id).strip():
            raise ValueError(
                "modifier eligibility witness_id must not be empty"
            )
        if not str(self.modifier_component_id).strip():
            raise ValueError(
                "modifier eligibility component id must not be empty"
            )
        if self.validated_anchor_slot == self.modifier_slot:
            raise ValueError(
                "modifier eligibility anchor and modifier slots must be opposite"
            )
        if not str(self.validated_anchor_text).strip():
            raise ValueError(
                "modifier eligibility anchor text must not be empty"
            )
        if not str(self.modifier_text).strip():
            raise ValueError(
                "modifier eligibility modifier text must not be empty"
            )
        if not self.provenance_ids:
            raise ValueError(
                "modifier eligibility witness requires provenance_ids"
            )
        return self


class EligibleModifierComponent(StrictModel):
    component: RelationComponentView
    eligibility: ModifierEligibilityWitness

    @model_validator(mode="after")
    def validate_component_identity(
        self,
    ) -> "EligibleModifierComponent":
        if (
            self.component.component_id
            != self.eligibility.modifier_component_id
        ):
            raise ValueError(
                "modifier eligibility witness component id mismatch"
            )

        anchor_text = _slot_text(
            self.component,
            self.eligibility.validated_anchor_slot,
        )
        modifier_text = _slot_text(
            self.component,
            self.eligibility.modifier_slot,
        )

        if not _text_exact(
            anchor_text,
            self.eligibility.validated_anchor_text,
        ):
            raise ValueError(
                "modifier eligibility anchor text/slot mismatch"
            )
        if not _text_exact(
            modifier_text,
            self.eligibility.modifier_text,
        ):
            raise ValueError(
                "modifier eligibility modifier text/slot mismatch"
            )
        return self


class HigherOrderRoleBindingView(StrictModel):
    source_endpoint_aliases: list[str] = Field(
        default_factory=list
    )
    target_endpoint_aliases: list[str] = Field(
        default_factory=list
    )
    mediator_aliases: list[str] = Field(
        default_factory=list
    )

    modifier_anchor_role: HigherOrderAttachmentRole
    modifier_anchor_slot: RelationArgumentSlot
    modifier_slot: RelationArgumentSlot

    modifier_anchor_text: str
    modifier_text: str

    anchor_match_mode: Literal[
        "exact_endpoint",
        "compatible_endpoint",
        "compatible_mediator",
    ]
    anchor_overlap_tokens: list[str] = Field(
        default_factory=list
    )
    anchor_compatibility_score: float = 0.0

    @model_validator(mode="after")
    def validate_role_binding(
        self,
    ) -> "HigherOrderRoleBindingView":
        if self.modifier_anchor_slot == self.modifier_slot:
            raise ValueError(
                "modifier anchor and modifier slots must be opposite"
            )
        if not str(self.modifier_anchor_text).strip():
            raise ValueError(
                "modifier anchor text must not be empty"
            )
        if not str(self.modifier_text).strip():
            raise ValueError(
                "modifier text must not be empty"
            )
        if not (0.0 <= float(self.anchor_compatibility_score) <= 1.0):
            raise ValueError(
                "anchor compatibility score must be within [0, 1]"
            )
        return self


class HigherOrderTopologyCandidate(StrictModel):
    """
    Shadow-only C ⊣ (A-M-B) composition object.

    A-M-B remains a verified task-conditioned relational backbone.
    C is an upstream-eligible modifier component attached to exactly one
    backbone role. Neither the modifier nor the topology obtains novelty
    authority at composition time.
    """

    schema_version: str = "higher-order-topology-candidate-v2"

    topology_id: str
    backbone_topology_id: str

    backbone: TaskBackboneLike
    modifier_component: RelationComponentView
    modifier_eligibility: ModifierEligibilityWitness
    role_binding: HigherOrderRoleBindingView

    modifier_authority: RelationComponentAuthority

    epistemic_status: Literal[
        "inspiration_only"
    ] = "inspiration_only"

    requires_verification: Literal[
        True
    ] = True

    novelty_authority: Literal[
        False
    ] = False

    shadow_only: Literal[
        True
    ] = True

    reason_codes: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_higher_order_contract(
        self,
    ) -> "HigherOrderTopologyCandidate":
        if self.backbone_topology_id != self.backbone.topology_id:
            raise ValueError(
                "higher-order backbone_topology_id mismatch"
            )
        if (
            self.modifier_component.component_id
            != self.modifier_eligibility.modifier_component_id
        ):
            raise ValueError(
                "higher-order modifier eligibility mismatch"
            )
        if (
            self.modifier_authority
            != self.modifier_component.authority
        ):
            raise ValueError(
                "higher-order modifier authority mismatch"
            )
        if (
            self.role_binding.modifier_anchor_role
            != self.modifier_eligibility.validated_anchor_role
        ):
            raise ValueError(
                "higher-order modifier anchor role lost upstream authority"
            )
        if (
            self.role_binding.modifier_anchor_slot
            != self.modifier_eligibility.validated_anchor_slot
        ):
            raise ValueError(
                "higher-order modifier anchor slot lost upstream authority"
            )
        if (
            self.role_binding.modifier_slot
            != self.modifier_eligibility.modifier_slot
        ):
            raise ValueError(
                "higher-order modifier slot lost upstream authority"
            )
        if not _text_exact(
            self.role_binding.modifier_anchor_text,
            self.modifier_eligibility.validated_anchor_text,
        ):
            raise ValueError(
                "higher-order modifier anchor text lost upstream authority"
            )
        if not _text_exact(
            self.role_binding.modifier_text,
            self.modifier_eligibility.modifier_text,
        ):
            raise ValueError(
                "higher-order modifier text lost upstream authority"
            )
        if not task_backbone_has_materializable_endpoint_fidelity(
            self.backbone
        ):
            raise ValueError(
                "higher-order topology requires a high-fidelity backbone"
            )
        return self


_DASHES = "‐‑‒–—−"


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
        + hashlib.sha256(raw).hexdigest()[:20]
    )


def _normalize_dashes(
    text: str,
) -> str:
    normalized = str(text)
    for dash in _DASHES:
        normalized = normalized.replace(dash, "-")
    return normalized


def _token_signatures(
    text: str,
) -> tuple[frozenset[str], ...]:
    normalized = _normalize_dashes(text)

    variants = (
        lexical_tokens(normalized),
        lexical_tokens(
            re.sub(
                r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])",
                "",
                normalized,
            )
        ),
    )

    out = []
    seen = set()

    for tokens in variants:
        if not tokens:
            continue

        key = tuple(sorted(tokens))
        if key in seen:
            continue

        seen.add(key)
        out.append(tokens)

    return tuple(out)


def _text_exact(
    left: str,
    right: str,
) -> bool:
    return any(
        a == b
        for a in _token_signatures(left)
        for b in _token_signatures(right)
    )


def _slot_text(
    component: RelationComponentView,
    slot: RelationArgumentSlot,
) -> str:
    return (
        component.subject
        if slot == "subject"
        else component.object
    )


def _opposite_slot(
    slot: RelationArgumentSlot,
) -> RelationArgumentSlot:
    return (
        "object"
        if slot == "subject"
        else "subject"
    )


def _append_unique(
    values: list[str],
    value: str,
) -> None:
    normalized = " ".join(str(value).split()).strip()
    if not normalized:
        return

    key = normalized.casefold()
    if all(
        existing.casefold() != key
        for existing in values
    ):
        values.append(normalized)


def _backbone_role_aliases(
    backbone: TaskBackboneLike,
) -> dict[
    HigherOrderAttachmentRole,
    list[str],
]:
    return task_backbone_role_aliases_for_composition(
        backbone
    )


def _mediator_anchor_compatibility(
    left: str,
    right: str,
) -> tuple[
    list[str],
    float,
] | None:
    best: tuple[
        list[str],
        float,
    ] | None = None

    for left_tokens in _token_signatures(left):
        for right_tokens in _token_signatures(right):
            shared = (
                left_tokens
                & right_tokens
            )

            if not shared:
                continue

            if left_tokens == right_tokens:
                candidate = (
                    sorted(shared),
                    1.0,
                )
            else:
                if (
                    len(left_tokens) == 1
                    or len(right_tokens) == 1
                ):
                    continue

                left_coverage = (
                    len(shared)
                    / len(left_tokens)
                )
                right_coverage = (
                    len(shared)
                    / len(right_tokens)
                )

                if (
                    len(shared) < 2
                    or min(
                        left_coverage,
                        right_coverage,
                    ) < 0.40
                ):
                    continue

                jaccard = (
                    len(shared)
                    / len(
                        left_tokens
                        | right_tokens
                    )
                )

                score = (
                    0.50 * jaccard
                    + 0.25 * left_coverage
                    + 0.25 * right_coverage
                )

                candidate = (
                    sorted(shared),
                    float(score),
                )

            if (
                best is None
                or candidate[1] > best[1]
                or (
                    candidate[1] == best[1]
                    and len(candidate[0]) > len(best[0])
                )
            ):
                best = candidate

    return best


def _endpoint_anchor_compatibility(
    left: str,
    right: str,
) -> tuple[
    list[str],
    float,
] | None:
    """
    Re-verify upstream endpoint-role containment without reinterpreting
    scientific identity.

    This mirrors the frozen higher-order modifier screen:
      containment(shared / min(|left|, |right|)) >= 0.80

    It is used only for TaskBackboneChainView endpoint anchors. Legacy
    two-component higher-order topology composition remains exact-only.
    """
    left_tokens = lexical_tokens(left)
    right_tokens = lexical_tokens(right)

    if not left_tokens or not right_tokens:
        return None

    shared = left_tokens & right_tokens
    if not shared:
        return None

    containment = (
        len(shared)
        / min(
            len(left_tokens),
            len(right_tokens),
        )
    )
    if containment < 0.80:
        return None

    return (
        sorted(shared),
        float(containment),
    )


def _match_slot_to_role(
    *,
    slot_text: str,
    role: HigherOrderAttachmentRole,
    aliases: Sequence[str],
    allow_compatible_endpoint: bool = False,
) -> tuple[
    Literal[
        "exact_endpoint",
        "compatible_endpoint",
        "compatible_mediator",
    ],
    list[str],
    float,
] | None:
    if role in {
        "source",
        "target",
    }:
        for alias in aliases:
            if _text_exact(
                slot_text,
                alias,
            ):
                tokens = sorted(
                    lexical_tokens(slot_text)
                    & lexical_tokens(alias)
                )
                return (
                    "exact_endpoint",
                    tokens,
                    1.0,
                )

        if not allow_compatible_endpoint:
            return None

        best = None
        for alias in aliases:
            compatibility = _endpoint_anchor_compatibility(
                slot_text,
                alias,
            )
            if compatibility is None:
                continue

            shared, score = compatibility
            candidate = (
                "compatible_endpoint",
                shared,
                score,
            )
            if (
                best is None
                or candidate[2] > best[2]
                or (
                    candidate[2] == best[2]
                    and len(candidate[1]) > len(best[1])
                )
            ):
                best = candidate

        return best

    best = None

    for alias in aliases:
        compatibility = (
            _mediator_anchor_compatibility(
                slot_text,
                alias,
            )
        )

        if compatibility is None:
            continue

        shared, score = compatibility
        candidate = (
            "compatible_mediator",
            shared,
            score,
        )

        if (
            best is None
            or candidate[2] > best[2]
            or (
                candidate[2] == best[2]
                and len(candidate[1]) > len(best[1])
            )
        ):
            best = candidate

    return best


def _modifier_leaks_exactly_into_backbone(
    *,
    modifier_text: str,
    aliases: dict[
        HigherOrderAttachmentRole,
        list[str],
    ],
) -> bool:
    return any(
        _text_exact(
            modifier_text,
            alias,
        )
        for role_aliases in aliases.values()
        for alias in role_aliases
    )


def compose_higher_order_topologies(
    *,
    backbones: Sequence[
        TaskBackboneLike
    ],
    modifiers: Sequence[
        EligibleModifierComponent
    ],
    max_topologies: int = 512,
    require_confirmed_known_backbone: bool = True,
) -> tuple[
    HigherOrderTopologyCandidate,
    ...,
]:
    """
    Compose shadow-only C ⊣ (A-M-B) candidates.

    Eligibility contract:
      1. the A-M-B backbone must already have exact/equivalent endpoint
         fidelity;
      2. optional default requires both backbone components to be
         CONFIRMED_KNOWN;
      3. a modifier must carry an explicit upstream eligibility witness;
      4. the witness fixes which component slot is the backbone anchor,
         which slot is C, and which backbone role was validated upstream;
      5. the composer may verify that binding but must never reinterpret
         the relation under another role or orientation;
      6. an anchor that also binds another backbone role fails closed;
      7. an exact C=A/B/M restatement is rejected defensively.

    Anchor semantic purity is upstream authority. The generic core requires
    anchor_purity_pass=True but does not encode domain-specific alias families.

    No novelty authority is created.
    """

    if max_topologies < 1:
        raise ValueError(
            "max_topologies must be >= 1"
        )

    rows = []

    for backbone in backbones:
        if not task_backbone_has_materializable_endpoint_fidelity(
            backbone
        ):
            continue

        if (
            require_confirmed_known_backbone
            and any(
                component.authority
                != RelationComponentAuthority.CONFIRMED_KNOWN
                for component in task_backbone_components(
                    backbone
                )
            )
        ):
            continue

        aliases = _backbone_role_aliases(
            backbone
        )

        backbone_component_ids = (
            task_backbone_component_ids(
                backbone
            )
        )

        for modifier in modifiers:
            component = modifier.component
            eligibility = modifier.eligibility

            if (
                component.component_id
                in backbone_component_ids
            ):
                continue

            role = eligibility.validated_anchor_role
            anchor_slot = eligibility.validated_anchor_slot
            modifier_slot = eligibility.modifier_slot

            anchor_text = _slot_text(
                component,
                anchor_slot,
            )
            modifier_text = _slot_text(
                component,
                modifier_slot,
            )

            if not str(modifier_text).strip():
                continue

            if _modifier_leaks_exactly_into_backbone(
                modifier_text=modifier_text,
                aliases=aliases,
            ):
                continue

            match = _match_slot_to_role(
                slot_text=anchor_text,
                role=role,
                aliases=aliases[role],
                allow_compatible_endpoint=isinstance(
                    backbone,
                    TaskBackboneChainView,
                ),
            )

            if match is None:
                continue

            conflicting_role_match = any(
                _match_slot_to_role(
                    slot_text=anchor_text,
                    role=other_role,
                    aliases=aliases[other_role],
                    allow_compatible_endpoint=isinstance(
                        backbone,
                        TaskBackboneChainView,
                    ),
                )
                is not None
                for other_role in (
                    "source",
                    "mediator",
                    "target",
                )
                if other_role != role
            )

            if conflicting_role_match:
                continue

            (
                match_mode,
                overlap_tokens,
                compatibility_score,
            ) = match

            topology_id = _stable_id(
                "higher_order_topology",
                backbone.topology_id,
                component.component_id,
                modifier.eligibility.witness_id,
                role,
                anchor_slot,
                modifier_slot,
                modifier_text,
            )

            rows.append(
                HigherOrderTopologyCandidate(
                    topology_id=topology_id,
                    backbone_topology_id=(
                        backbone.topology_id
                    ),
                    backbone=backbone,
                    modifier_component=component,
                    modifier_eligibility=(
                        modifier.eligibility
                    ),
                    role_binding=(
                        HigherOrderRoleBindingView(
                            source_endpoint_aliases=(
                                aliases["source"]
                            ),
                            target_endpoint_aliases=(
                                aliases["target"]
                            ),
                            mediator_aliases=(
                                aliases["mediator"]
                            ),
                            modifier_anchor_role=role,
                            modifier_anchor_slot=anchor_slot,
                            modifier_slot=modifier_slot,
                            modifier_anchor_text=anchor_text,
                            modifier_text=modifier_text,
                            anchor_match_mode=(
                                match_mode
                            ),
                            anchor_overlap_tokens=(
                                overlap_tokens
                            ),
                            anchor_compatibility_score=(
                                compatibility_score
                            ),
                        )
                    ),
                    modifier_authority=(
                        component.authority
                    ),
                    epistemic_status=(
                        "inspiration_only"
                    ),
                    requires_verification=True,
                    novelty_authority=False,
                    shadow_only=True,
                    reason_codes=[
                        "higher_order_topology_shadow_only",
                        "backbone_endpoint_fidelity_preserved",
                        "modifier_role_bound_upstream_witness_required",
                        (
                            "modifier_attachment_role:"
                            + role
                        ),
                        (
                            "modifier_authority:"
                            + component.authority.value
                        ),
                        "confirmed_known_is_not_novel_evidence",
                        "higher_order_topology_requires_verification",
                    ],
                )
            )

    rows.sort(
        key=lambda row: (
            row.backbone_topology_id,
            row.role_binding.modifier_anchor_role,
            row.modifier_component.component_id,
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

        if len(selected) >= max_topologies:
            break

    return tuple(selected)
