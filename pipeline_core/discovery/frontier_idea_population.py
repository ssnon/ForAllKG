from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisPlan,
)
from pipeline_core.discovery.higher_order_competing_explanations import (
    CompetingExplanationSet,
)
from pipeline_core.discovery.higher_order_tension_extractor import (
    ScientificTensionCandidateSet,
)
from pipeline_core.discovery.open_world_discovery_axis import (
    OpenWorldExternalAxisBundle,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


FrontierIdeaForm = Literal[
    "RELATION_AXIS",
    "HIGHER_ORDER_TOPOLOGY",
    "TENSION_SEED",
    "COMPETING_EXPLANATION_SEED",
]

FrontierSourceKind = Literal[
    "KG_AXIS",
    "OPEN_WORLD_AXIS",
    "HIGHER_ORDER",
    "DIRECT_HIGHER_ORDER",
    "TENSION_DERIVED",
]

FrontierTaskRelationMode = Literal[
    "DIRECT",
    "SUBORDINATE",
    "UNKNOWN",
]


class FrontierRelationSignature(StrictModel):
    subject: str
    relation: str
    object: str
    candidate_unit_id: str = ""


class FrontierTopologySignature(StrictModel):
    task_source: str = ""
    task_target: str = ""
    modifier_text: str = ""
    modifier_anchor_role: str = ""
    modifier_anchor_text: str = ""
    modifier_component_id: str = ""
    modifier_candidate_unit_id: str = ""
    modifier_relation_text: str = ""
    backbone_component_ids: list[str] = Field(default_factory=list)
    backbone_relation_texts: list[str] = Field(default_factory=list)


class FrontierTensionSignature(StrictModel):
    tension_type: str
    requested_source: str
    requested_target: str
    tension_statement: str
    basis_relation_texts: list[str] = Field(default_factory=list)


class FrontierCompetingExplanationSignature(StrictModel):
    tension_id: str
    tension_type: str
    explanation_type: str
    explanation_role: str
    requested_source: str
    requested_target: str
    explanation_statement: str
    discriminator_requirement: str
    basis_relation_texts: list[str] = Field(default_factory=list)


class FrontierIdeaSourceLineage(StrictModel):
    source_kind: FrontierSourceKind
    source_artifact: str
    source_artifact_sha256: str
    source_object_id: str

    source_plan_id: str | None = None
    source_plan_sha256: str | None = None
    source_bundle_id: str | None = None
    source_bundle_sha256: str | None = None

    source_context_ids: list[str] = Field(default_factory=list)
    external_work_ids: list[str] = Field(default_factory=list)
    compatible_grounded_statement_ids: list[str] = Field(default_factory=list)

    external_literature_lineage: bool = False
    candidate_or_unverified_lineage: bool = False


class FrontierIdea(StrictModel):
    schema_version: Literal[
        "frontier-idea-v1"
    ] = "frontier-idea-v1"

    idea_id: str
    idea_form: FrontierIdeaForm
    source_kind: FrontierSourceKind

    source_context_id: str
    source_context_sha256: str

    source_lineage: list[FrontierIdeaSourceLineage] = Field(
        min_length=1,
    )

    rendered_scientific_intent: str = Field(min_length=1)
    task_relation_mode: FrontierTaskRelationMode

    relation_signature: FrontierRelationSignature | None = None
    topology_signature: FrontierTopologySignature | None = None
    tension_signature: FrontierTensionSignature | None = None
    competing_explanation_signature: (
        FrontierCompetingExplanationSignature | None
    ) = None

    exact_scientific_signature: str

    requires_verification: Literal[True] = True
    epistemic_status: Literal[
        "INSPIRATION_ONLY"
    ] = "INSPIRATION_ONLY"

    positive_premise_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_form_signature(self) -> "FrontierIdea":
        expected = {
            "RELATION_AXIS": self.relation_signature,
            "HIGHER_ORDER_TOPOLOGY": self.topology_signature,
            "TENSION_SEED": self.tension_signature,
            "COMPETING_EXPLANATION_SEED":
                self.competing_explanation_signature,
        }
        active = [
            key
            for key, value in expected.items()
            if value is not None
        ]
        if active != [self.idea_form]:
            raise ValueError(
                "frontier idea form/signature mismatch: "
                f"idea_form={self.idea_form!r}, "
                f"active={active!r}"
            )
        return self


class FrontierExactDuplicateGroup(StrictModel):
    exact_scientific_signature: str
    idea_ids: list[str] = Field(min_length=2)
    source_kinds: list[FrontierSourceKind] = Field(min_length=1)
    cross_source: bool


class FrontierOverlapDiagnostic(StrictModel):
    idea_id_a: str
    idea_id_b: str
    source_kind_a: FrontierSourceKind
    source_kind_b: FrontierSourceKind

    method: Literal[
        "TOKEN_JACCARD_V1"
    ] = "TOKEN_JACCARD_V1"
    score: float = Field(ge=0.0, le=1.0)

    cross_source: Literal[True] = True
    possible_duplicate: bool

    diagnostic_only: Literal[True] = True
    candidate_deletion_authority: Literal[False] = False


FrontierStructuralMatchMethod = Literal[
    "SAME_CANDIDATE_UNIT_V1",
    "EXACT_NORMALIZED_RELATION_COMPONENT_V1",
]


class FrontierStructuralOverlapDiagnostic(StrictModel):
    idea_id_a: str
    idea_id_b: str
    source_kind_a: FrontierSourceKind
    source_kind_b: FrontierSourceKind

    match_methods: list[
        FrontierStructuralMatchMethod
    ] = Field(min_length=1)
    shared_candidate_unit_ids: list[str] = Field(default_factory=list)
    shared_relation_texts: list[str] = Field(default_factory=list)
    relation_roles_a: list[str] = Field(min_length=1)
    relation_roles_b: list[str] = Field(min_length=1)

    cross_source: Literal[True] = True
    diagnostic_only: Literal[True] = True
    whole_idea_duplicate_asserted: Literal[False] = False
    candidate_deletion_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class FrontierIdeaPopulation(StrictModel):
    schema_version: Literal[
        "frontier-idea-population-v1"
    ] = "frontier-idea-population-v1"

    population_id: str
    population_sha256: str

    source_context_id: str
    source_context_sha256: str

    research_question: str
    task_source: str
    task_target: str

    ideas: list[FrontierIdea] = Field(default_factory=list)

    total_idea_count: int = Field(ge=0)
    idea_count_by_source_kind: dict[str, int] = Field(default_factory=dict)
    idea_count_by_idea_form: dict[str, int] = Field(default_factory=dict)
    task_relation_mode_counts: dict[str, int] = Field(default_factory=dict)

    exact_duplicate_groups: list[FrontierExactDuplicateGroup] = Field(
        default_factory=list
    )
    cross_source_exact_duplicate_group_count: int = Field(ge=0)

    overlap_diagnostics: list[FrontierOverlapDiagnostic] = Field(
        default_factory=list
    )
    cross_source_overlap_pair_count: int = Field(ge=0)

    structural_overlap_diagnostics: list[
        FrontierStructuralOverlapDiagnostic
    ] = Field(default_factory=list)
    cross_source_structural_overlap_pair_count: int = Field(ge=0)

    external_literature_lineage_idea_count: int = Field(ge=0)
    candidate_or_unverified_lineage_idea_count: int = Field(ge=0)

    ordering_policy: Literal[
        "SOURCE_FAMILY_THEN_UPSTREAM_ORDER_V1"
    ] = "SOURCE_FAMILY_THEN_UPSTREAM_ORDER_V1"

    shadow_only: Literal[True] = True
    new_llm_calls: Literal[False] = False
    new_retrieval_calls: Literal[False] = False
    positive_premise_authority_created: Literal[False] = False
    novelty_authority_created: Literal[False] = False
    generation_authority_created: Literal[False] = False
    production_selection_authority: Literal[False] = False
    stage8_input_changed: Literal[False] = False
    canonical_graph_mutated: Literal[False] = False


_SOURCE_ORDER: dict[str, int] = {
    "KG_AXIS": 0,
    "OPEN_WORLD_AXIS": 1,
    "HIGHER_ORDER": 2,
    "DIRECT_HIGHER_ORDER": 3,
    "TENSION_DERIVED": 4,
}

_FORM_ORDER: dict[str, int] = {
    "RELATION_AXIS": 0,
    "HIGHER_ORDER_TOPOLOGY": 1,
    "TENSION_SEED": 2,
    "COMPETING_EXPLANATION_SEED": 3,
}

_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)?"
)


def _canonical_json(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def _stable_id(
    prefix: str,
    *parts: object,
    length: int = 20,
) -> str:
    raw = "|".join(
        str(part)
        for part in parts
    )
    return (
        f"{prefix}:"
        + _sha256_text(raw)[:length]
    )


def _norm(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(
        r"[^a-z0-9α-ω가-힣]+",
        " ",
        text,
    )
    return " ".join(text.split())


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(
            str(value)
        )
    }


def _token_jaccard(
    left: str,
    right: str,
) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _relation_text(
    row: dict[str, Any],
) -> str | None:
    subject = str(
        row.get("subject") or ""
    ).strip()
    relation = str(
        row.get("relation") or ""
    ).strip()
    obj = str(
        row.get("object") or ""
    ).strip()
    if not (
        subject
        and relation
        and obj
    ):
        return None
    return (
        f"{subject} --{relation}--> {obj}"
    )


def _collect_relation_rows(
    value: object,
) -> tuple[list[str], list[str]]:
    component_ids: list[str] = []
    relation_texts: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            text = _relation_text(node)
            component_id = str(
                node.get("component_id")
                or ""
            ).strip()
            if text:
                if text not in relation_texts:
                    relation_texts.append(text)
                if (
                    component_id
                    and component_id
                    not in component_ids
                ):
                    component_ids.append(
                        component_id
                    )
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)

    return (
        component_ids,
        relation_texts,
    )


def _task_endpoints_from_topology(
    topology: dict[str, Any],
    *,
    fallback_source: str,
    fallback_target: str,
) -> tuple[str, str]:
    backbone = topology.get(
        "backbone"
    )
    if isinstance(backbone, dict):
        source = str(
            backbone.get(
                "requested_source"
            )
            or ""
        ).strip()
        target = str(
            backbone.get(
                "requested_target"
            )
            or ""
        ).strip()
        if source or target:
            return (
                source
                or fallback_source,
                target
                or fallback_target,
            )

    return (
        fallback_source,
        fallback_target,
    )


def _topology_signature(
    topology: dict[str, Any],
    *,
    task_source: str,
    task_target: str,
) -> FrontierTopologySignature:
    binding = topology.get(
        "role_binding"
    )
    if not isinstance(binding, dict):
        binding = {}

    source, target = (
        _task_endpoints_from_topology(
            topology,
            fallback_source=task_source,
            fallback_target=task_target,
        )
    )

    modifier = topology.get(
        "modifier_component"
    )
    if not isinstance(
        modifier,
        dict,
    ):
        modifier = {}

    modifier_component_id = str(
        modifier.get("component_id")
        or ""
    ).strip()
    modifier_candidate_unit_id = str(
        modifier.get("candidate_unit_id")
        or ""
    ).strip()
    modifier_relation_text = (
        _relation_text(modifier)
        or ""
    )

    modifier_text = str(
        binding.get("modifier_text")
        or ""
    ).strip()

    if not modifier_text:
        slot = str(
            binding.get("modifier_slot")
            or ""
        )
        if slot in {
            "subject",
            "object",
        }:
            modifier_text = str(
                modifier.get(slot)
                or ""
            ).strip()

    anchor_role = str(
        binding.get(
            "modifier_anchor_role"
        )
        or binding.get(
            "anchor_role"
        )
        or ""
    ).strip()

    anchor_text = str(
        binding.get(
            "modifier_anchor_text"
        )
        or ""
    ).strip()

    component_ids, relation_texts = (
        _collect_relation_rows(
            topology.get("backbone")
        )
    )

    return FrontierTopologySignature(
        task_source=source,
        task_target=target,
        modifier_text=modifier_text,
        modifier_anchor_role=anchor_role,
        modifier_anchor_text=anchor_text,
        modifier_component_id=(
            modifier_component_id
        ),
        modifier_candidate_unit_id=(
            modifier_candidate_unit_id
        ),
        modifier_relation_text=(
            modifier_relation_text
        ),
        backbone_component_ids=(
            component_ids
        ),
        backbone_relation_texts=(
            relation_texts
        ),
    )


def _axis_exact_signature(
    subject: str,
    relation: str,
    obj: str,
) -> str:
    canonical = (
        "RELATION_AXIS|"
        + "|".join(
            [
                _norm(subject),
                _norm(relation),
                _norm(obj),
            ]
        )
    )
    return _sha256_text(
        canonical
    )


def _topology_exact_signature(
    signature: FrontierTopologySignature,
) -> str:
    canonical = {
        "idea_form":
            "HIGHER_ORDER_TOPOLOGY",
        "task_source":
            _norm(signature.task_source),
        "task_target":
            _norm(signature.task_target),
        "modifier_text":
            _norm(signature.modifier_text),
        "modifier_anchor_role":
            _norm(
                signature.modifier_anchor_role
            ),
        "modifier_anchor_text":
            _norm(
                signature.modifier_anchor_text
            ),
        "backbone_relation_texts": sorted(
            _norm(row)
            for row
            in signature.backbone_relation_texts
        ),
    }
    return _sha256_text(
        _canonical_json(
            canonical
        )
    )


def _tension_exact_signature(
    signature: FrontierTensionSignature,
) -> str:
    canonical = {
        "idea_form": "TENSION_SEED",
        "tension_type":
            _norm(signature.tension_type),
        "requested_source":
            _norm(
                signature.requested_source
            ),
        "requested_target":
            _norm(
                signature.requested_target
            ),
        "tension_statement":
            _norm(
                signature.tension_statement
            ),
        "basis_relation_texts": sorted(
            _norm(row)
            for row
            in signature.basis_relation_texts
        ),
    }
    return _sha256_text(
        _canonical_json(
            canonical
        )
    )


def _explanation_exact_signature(
    signature: FrontierCompetingExplanationSignature,
) -> str:
    canonical = {
        "idea_form":
            "COMPETING_EXPLANATION_SEED",
        "tension_type":
            _norm(signature.tension_type),
        "explanation_type":
            _norm(
                signature.explanation_type
            ),
        "requested_source":
            _norm(
                signature.requested_source
            ),
        "requested_target":
            _norm(
                signature.requested_target
            ),
        "explanation_statement":
            _norm(
                signature.explanation_statement
            ),
        "basis_relation_texts": sorted(
            _norm(row)
            for row
            in signature.basis_relation_texts
        ),
    }
    return _sha256_text(
        _canonical_json(
            canonical
        )
    )


def _idea_id(
    *,
    source_kind: FrontierSourceKind,
    source_object_id: str,
    exact_scientific_signature: str,
) -> str:
    return _stable_id(
        "frontier_idea",
        source_kind,
        source_object_id,
        exact_scientific_signature,
    )


def axis_ideas(
    *,
    plan: DiscoveryAxisPlan,
    source_kind: Literal[
        "KG_AXIS",
        "OPEN_WORLD_AXIS",
    ],
    source_artifact: str,
    source_artifact_sha256: str,
    source_context_id: str,
    source_context_sha256: str,
    open_world_bundle: OpenWorldExternalAxisBundle | None = None,
) -> list[FrontierIdea]:
    provenance_by_axis = {}

    if source_kind == "OPEN_WORLD_AXIS":
        if open_world_bundle is None:
            raise ValueError(
                "OPEN_WORLD_AXIS requires "
                "OpenWorldExternalAxisBundle"
            )
        if (
            plan.source_bundle_id
            != open_world_bundle.bundle_id
            or plan.source_bundle_sha256
            != open_world_bundle.bundle_sha256
        ):
            raise ValueError(
                "open-world plan/bundle lineage mismatch"
            )
        if (
            plan.source_dual_context_id
            != open_world_bundle.source_dual_context_id
            or plan.source_dual_context_sha256
            != open_world_bundle.source_dual_context_sha256
        ):
            raise ValueError(
                "open-world plan/bundle dual-context mismatch"
            )
        provenance_by_axis = {
            row.axis_id: row
            for row
            in open_world_bundle.provenance
        }

    ideas = []

    for axis in plan.axes:
        relation_signature = (
            FrontierRelationSignature(
                subject=axis.proposed_subject,
                relation=axis.proposed_relation,
                object=axis.proposed_object,
                candidate_unit_id=str(
                    axis.candidate_unit_id
                    or ""
                ),
            )
        )
        exact = _axis_exact_signature(
            axis.proposed_subject,
            axis.proposed_relation,
            axis.proposed_object,
        )

        provenance = provenance_by_axis.get(
            axis.axis_id
        )
        if (
            source_kind
            == "OPEN_WORLD_AXIS"
            and provenance is None
        ):
            raise ValueError(
                "open-world axis missing provenance: "
                f"{axis.axis_id}"
            )

        external_work_ids = (
            list(
                provenance.source_work_ids
            )
            if provenance is not None
            else []
        )
        compatible_grounded = (
            list(
                provenance
                .compatible_grounded_statement_ids
            )
            if provenance is not None
            else []
        )

        lineage = FrontierIdeaSourceLineage(
            source_kind=source_kind,
            source_artifact=source_artifact,
            source_artifact_sha256=(
                source_artifact_sha256
            ),
            source_object_id=axis.axis_id,
            source_plan_id=plan.plan_id,
            source_plan_sha256=(
                plan.plan_sha256
            ),
            source_bundle_id=(
                plan.source_bundle_id
            ),
            source_bundle_sha256=(
                plan.source_bundle_sha256
            ),
            source_context_ids=[
                plan.source_dual_context_id
            ],
            external_work_ids=(
                external_work_ids
            ),
            compatible_grounded_statement_ids=(
                compatible_grounded
            ),
            external_literature_lineage=(
                source_kind
                == "OPEN_WORLD_AXIS"
            ),
            candidate_or_unverified_lineage=(
                bool(
                    axis.requires_verification
                )
            ),
        )

        intent = (
            f"{axis.label}: "
            f"{axis.proposed_subject} "
            f"--{axis.proposed_relation}--> "
            f"{axis.proposed_object}"
        )

        ideas.append(
            FrontierIdea(
                idea_id=_idea_id(
                    source_kind=source_kind,
                    source_object_id=axis.axis_id,
                    exact_scientific_signature=(
                        exact
                    ),
                ),
                idea_form="RELATION_AXIS",
                source_kind=source_kind,
                source_context_id=(
                    source_context_id
                ),
                source_context_sha256=(
                    source_context_sha256
                ),
                source_lineage=[
                    lineage
                ],
                rendered_scientific_intent=(
                    intent
                ),
                task_relation_mode=(
                    "SUBORDINATE"
                ),
                relation_signature=(
                    relation_signature
                ),
                exact_scientific_signature=(
                    exact
                ),
            )
        )

    return ideas


def topology_ideas(
    *,
    topologies: Sequence[
        dict[str, Any]
    ],
    source_kind: Literal[
        "HIGHER_ORDER",
        "DIRECT_HIGHER_ORDER",
    ],
    source_artifact: str,
    source_artifact_sha256: str,
    source_context_id: str,
    source_context_sha256: str,
    task_source: str,
    task_target: str,
) -> list[FrontierIdea]:
    ideas = []

    for topology in topologies:
        topology_id = str(
            topology.get(
                "topology_id"
            )
            or ""
        ).strip()

        if not topology_id:
            raise ValueError(
                "higher-order topology lacks topology_id"
            )

        signature = _topology_signature(
            topology,
            task_source=task_source,
            task_target=task_target,
        )

        exact = _topology_exact_signature(
            signature
        )

        modifier = topology.get(
            "modifier_component"
        )
        if not isinstance(
            modifier,
            dict,
        ):
            modifier = {}

        authority = str(
            modifier.get("authority")
            or ""
        ).casefold()

        candidate_lineage = (
            "candidate" in authority
            or source_kind
            == "DIRECT_HIGHER_ORDER"
        )

        lineage = FrontierIdeaSourceLineage(
            source_kind=source_kind,
            source_artifact=source_artifact,
            source_artifact_sha256=(
                source_artifact_sha256
            ),
            source_object_id=topology_id,
            source_context_ids=[
                source_context_id
            ],
            external_literature_lineage=False,
            candidate_or_unverified_lineage=(
                candidate_lineage
            ),
        )

        backbone_preview = (
            " ; ".join(
                signature
                .backbone_relation_texts[:3]
            )
            or "recorded task backbone"
        )
        role = (
            signature.modifier_anchor_role
            or "task role"
        )
        modifier_text = (
            signature.modifier_text
            or "candidate modifier"
        )

        intent = (
            f"{signature.task_source} -> "
            f"{signature.task_target} "
            f"with {modifier_text} attached "
            f"to {role}; backbone: "
            f"{backbone_preview}"
        )

        ideas.append(
            FrontierIdea(
                idea_id=_idea_id(
                    source_kind=source_kind,
                    source_object_id=(
                        topology_id
                    ),
                    exact_scientific_signature=(
                        exact
                    ),
                ),
                idea_form=(
                    "HIGHER_ORDER_TOPOLOGY"
                ),
                source_kind=source_kind,
                source_context_id=(
                    source_context_id
                ),
                source_context_sha256=(
                    source_context_sha256
                ),
                source_lineage=[
                    lineage
                ],
                rendered_scientific_intent=(
                    intent
                ),
                task_relation_mode=(
                    "SUBORDINATE"
                ),
                topology_signature=(
                    signature
                ),
                exact_scientific_signature=(
                    exact
                ),
            )
        )

    return ideas


def tension_ideas(
    *,
    tensions: ScientificTensionCandidateSet,
    source_artifact: str,
    source_artifact_sha256: str,
    source_context_id: str,
    source_context_sha256: str,
) -> list[FrontierIdea]:
    ideas = []

    for row in tensions.candidates:
        signature = (
            FrontierTensionSignature(
                tension_type=(
                    row.tension_type
                ),
                requested_source=(
                    row.requested_source
                ),
                requested_target=(
                    row.requested_target
                ),
                tension_statement=(
                    row.tension_statement
                ),
                basis_relation_texts=(
                    list(
                        row
                        .basis_relation_texts
                    )
                ),
            )
        )

        exact = _tension_exact_signature(
            signature
        )

        lineage = FrontierIdeaSourceLineage(
            source_kind="TENSION_DERIVED",
            source_artifact=source_artifact,
            source_artifact_sha256=(
                source_artifact_sha256
            ),
            source_object_id=(
                row.tension_id
            ),
            source_context_ids=(
                list(
                    row.source_context_ids
                )
            ),
            external_literature_lineage=False,
            candidate_or_unverified_lineage=(
                bool(
                    row
                    .candidate_inspiration_involved
                )
            ),
        )

        ideas.append(
            FrontierIdea(
                idea_id=_idea_id(
                    source_kind=(
                        "TENSION_DERIVED"
                    ),
                    source_object_id=(
                        row.tension_id
                    ),
                    exact_scientific_signature=(
                        exact
                    ),
                ),
                idea_form="TENSION_SEED",
                source_kind=(
                    "TENSION_DERIVED"
                ),
                source_context_id=(
                    source_context_id
                ),
                source_context_sha256=(
                    source_context_sha256
                ),
                source_lineage=[
                    lineage
                ],
                rendered_scientific_intent=(
                    row.tension_statement
                ),
                task_relation_mode="UNKNOWN",
                tension_signature=(
                    signature
                ),
                exact_scientific_signature=(
                    exact
                ),
            )
        )

    return ideas


def competing_explanation_ideas(
    *,
    explanations: CompetingExplanationSet,
    source_artifact: str,
    source_artifact_sha256: str,
    source_context_id: str,
    source_context_sha256: str,
) -> list[FrontierIdea]:
    ideas = []

    for pair in explanations.pairs:
        for row in pair.explanations:
            signature = (
                FrontierCompetingExplanationSignature(
                    tension_id=(
                        row.tension_id
                    ),
                    tension_type=(
                        row.tension_type
                    ),
                    explanation_type=(
                        row.explanation_type
                    ),
                    explanation_role=(
                        row.explanation_role
                    ),
                    requested_source=(
                        pair.requested_source
                    ),
                    requested_target=(
                        pair.requested_target
                    ),
                    explanation_statement=(
                        row.explanation_statement
                    ),
                    discriminator_requirement=(
                        row
                        .discriminator_requirement
                    ),
                    basis_relation_texts=(
                        list(
                            row
                            .basis_relation_texts
                        )
                    ),
                )
            )

            exact = (
                _explanation_exact_signature(
                    signature
                )
            )

            lineage = (
                FrontierIdeaSourceLineage(
                    source_kind=(
                        "TENSION_DERIVED"
                    ),
                    source_artifact=(
                        source_artifact
                    ),
                    source_artifact_sha256=(
                        source_artifact_sha256
                    ),
                    source_object_id=(
                        row.explanation_id
                    ),
                    source_context_ids=(
                        list(
                            row
                            .source_context_ids
                        )
                    ),
                    external_literature_lineage=(
                        False
                    ),
                    candidate_or_unverified_lineage=(
                        bool(
                            row
                            .candidate_inspiration_involved
                        )
                    ),
                )
            )

            ideas.append(
                FrontierIdea(
                    idea_id=_idea_id(
                        source_kind=(
                            "TENSION_DERIVED"
                        ),
                        source_object_id=(
                            row.explanation_id
                        ),
                        exact_scientific_signature=(
                            exact
                        ),
                    ),
                    idea_form=(
                        "COMPETING_EXPLANATION_SEED"
                    ),
                    source_kind=(
                        "TENSION_DERIVED"
                    ),
                    source_context_id=(
                        source_context_id
                    ),
                    source_context_sha256=(
                        source_context_sha256
                    ),
                    source_lineage=[
                        lineage
                    ],
                    rendered_scientific_intent=(
                        row.explanation_statement
                    ),
                    task_relation_mode=(
                        "UNKNOWN"
                    ),
                    competing_explanation_signature=(
                        signature
                    ),
                    exact_scientific_signature=(
                        exact
                    ),
                )
            )

    return ideas


def _dedupe_groups(
    ideas: Sequence[FrontierIdea],
) -> list[FrontierExactDuplicateGroup]:
    by_signature: dict[
        str,
        list[FrontierIdea],
    ] = {}

    for idea in ideas:
        by_signature.setdefault(
            idea.exact_scientific_signature,
            [],
        ).append(idea)

    groups = []

    for signature in sorted(
        by_signature
    ):
        rows = by_signature[
            signature
        ]
        if len(rows) < 2:
            continue

        source_kinds = sorted(
            {
                row.source_kind
                for row in rows
            },
            key=lambda value:
                _SOURCE_ORDER[value],
        )

        groups.append(
            FrontierExactDuplicateGroup(
                exact_scientific_signature=(
                    signature
                ),
                idea_ids=[
                    row.idea_id
                    for row in rows
                ],
                source_kinds=source_kinds,
                cross_source=(
                    len(
                        set(
                            source_kinds
                        )
                    )
                    > 1
                ),
            )
        )

    return groups


def _overlap_diagnostics(
    ideas: Sequence[FrontierIdea],
    *,
    minimum_score: float = 0.35,
    possible_duplicate_score: float = 0.80,
) -> list[FrontierOverlapDiagnostic]:
    rows = []

    for index, left in enumerate(
        ideas
    ):
        for right in ideas[
            index + 1:
        ]:
            if (
                left.source_kind
                == right.source_kind
            ):
                continue

            score = _token_jaccard(
                left.rendered_scientific_intent,
                right.rendered_scientific_intent,
            )

            if score < minimum_score:
                continue

            rows.append(
                FrontierOverlapDiagnostic(
                    idea_id_a=left.idea_id,
                    idea_id_b=right.idea_id,
                    source_kind_a=(
                        left.source_kind
                    ),
                    source_kind_b=(
                        right.source_kind
                    ),
                    score=round(
                        float(score),
                        6,
                    ),
                    possible_duplicate=(
                        score
                        >= possible_duplicate_score
                    ),
                )
            )

    rows.sort(
        key=lambda row: (
            -row.score,
            row.idea_id_a,
            row.idea_id_b,
        )
    )
    return rows


def _idea_relation_components(
    idea: FrontierIdea,
) -> list[dict[str, str]]:
    """Expose relation-level building blocks without flattening idea forms.

    Candidate-unit identity is used only as a lineage equality witness.
    Exact normalized relation text is a separate weaker structural witness.
    Neither witness asserts whole-idea duplication or scientific equivalence.
    """

    rows: list[dict[str, str]] = []

    def add(
        *,
        role: str,
        relation_text: str,
        candidate_unit_id: str = "",
    ) -> None:
        relation_text = str(
            relation_text or ""
        ).strip()
        candidate_unit_id = str(
            candidate_unit_id or ""
        ).strip()
        if not relation_text and not candidate_unit_id:
            return
        rows.append(
            {
                "role": role,
                "relation_text": relation_text,
                "relation_key": _norm(
                    relation_text
                ),
                "candidate_unit_id": (
                    candidate_unit_id
                ),
            }
        )

    if idea.relation_signature is not None:
        signature = idea.relation_signature
        add(
            role="PRIMARY_RELATION",
            relation_text=(
                f"{signature.subject} "
                f"--{signature.relation}--> "
                f"{signature.object}"
            ),
            candidate_unit_id=(
                signature.candidate_unit_id
            ),
        )

    if idea.topology_signature is not None:
        signature = idea.topology_signature
        add(
            role="MODIFIER_RELATION",
            relation_text=(
                signature.modifier_relation_text
            ),
            candidate_unit_id=(
                signature.modifier_candidate_unit_id
            ),
        )
        for relation_text in (
            signature.backbone_relation_texts
        ):
            add(
                role="BACKBONE_RELATION",
                relation_text=relation_text,
            )

    if idea.tension_signature is not None:
        for relation_text in (
            idea.tension_signature
            .basis_relation_texts
        ):
            add(
                role="BASIS_RELATION",
                relation_text=relation_text,
            )

    if (
        idea.competing_explanation_signature
        is not None
    ):
        for relation_text in (
            idea.competing_explanation_signature
            .basis_relation_texts
        ):
            add(
                role="BASIS_RELATION",
                relation_text=relation_text,
            )

    return rows


def _structural_overlap_diagnostics(
    ideas: Sequence[FrontierIdea],
) -> list[FrontierStructuralOverlapDiagnostic]:
    rows = []
    components = {
        idea.idea_id:
            _idea_relation_components(idea)
        for idea in ideas
    }

    for index, left in enumerate(ideas):
        for right in ideas[index + 1:]:
            if (
                left.source_kind
                == right.source_kind
            ):
                continue

            methods = set()
            shared_candidate_ids = set()
            shared_relation_texts = set()
            roles_a = set()
            roles_b = set()

            for a in components[left.idea_id]:
                for b in components[right.idea_id]:
                    same_candidate = (
                        bool(a["candidate_unit_id"])
                        and a["candidate_unit_id"]
                        == b["candidate_unit_id"]
                    )
                    same_relation = (
                        bool(a["relation_key"])
                        and a["relation_key"]
                        == b["relation_key"]
                    )

                    if not (
                        same_candidate
                        or same_relation
                    ):
                        continue

                    roles_a.add(a["role"])
                    roles_b.add(b["role"])

                    if same_candidate:
                        methods.add(
                            "SAME_CANDIDATE_UNIT_V1"
                        )
                        shared_candidate_ids.add(
                            a["candidate_unit_id"]
                        )

                    if same_relation:
                        methods.add(
                            "EXACT_NORMALIZED_RELATION_COMPONENT_V1"
                        )
                        shared_relation_texts.add(
                            a["relation_text"]
                        )

            if not methods:
                continue

            rows.append(
                FrontierStructuralOverlapDiagnostic(
                    idea_id_a=left.idea_id,
                    idea_id_b=right.idea_id,
                    source_kind_a=(
                        left.source_kind
                    ),
                    source_kind_b=(
                        right.source_kind
                    ),
                    match_methods=sorted(
                        methods
                    ),
                    shared_candidate_unit_ids=sorted(
                        shared_candidate_ids
                    ),
                    shared_relation_texts=sorted(
                        shared_relation_texts
                    ),
                    relation_roles_a=sorted(
                        roles_a
                    ),
                    relation_roles_b=sorted(
                        roles_b
                    ),
                )
            )

    rows.sort(
        key=lambda row: (
            row.idea_id_a,
            row.idea_id_b,
            tuple(row.match_methods),
        )
    )
    return rows


def build_frontier_idea_population(
    *,
    source_context_id: str,
    source_context_sha256: str,
    research_question: str,
    task_source: str,
    task_target: str,
    kg_axis_plan: DiscoveryAxisPlan,
    kg_axis_artifact: str,
    kg_axis_artifact_sha256: str,
    open_world_axis_plan: DiscoveryAxisPlan | None = None,
    open_world_axis_artifact: str | None = None,
    open_world_axis_artifact_sha256: str | None = None,
    open_world_axis_bundle: OpenWorldExternalAxisBundle | None = None,
    higher_order_topologies: Sequence[
        dict[str, Any]
    ] = (),
    higher_order_topology_artifact: str | None = None,
    higher_order_topology_artifact_sha256: str | None = None,
    direct_higher_order_topologies: Sequence[
        dict[str, Any]
    ] = (),
    direct_higher_order_topology_artifact: str | None = None,
    direct_higher_order_topology_artifact_sha256: str | None = None,
    tension_candidates: ScientificTensionCandidateSet | None = None,
    tension_artifact: str | None = None,
    tension_artifact_sha256: str | None = None,
    competing_explanations: CompetingExplanationSet | None = None,
    competing_explanations_artifact: str | None = None,
    competing_explanations_artifact_sha256: str | None = None,
) -> FrontierIdeaPopulation:
    if not (
        source_context_id
        and source_context_sha256
    ):
        raise ValueError(
            "frontier population requires "
            "source context identity"
        )

    ideas = []

    ideas.extend(
        axis_ideas(
            plan=kg_axis_plan,
            source_kind="KG_AXIS",
            source_artifact=(
                kg_axis_artifact
            ),
            source_artifact_sha256=(
                kg_axis_artifact_sha256
            ),
            source_context_id=(
                source_context_id
            ),
            source_context_sha256=(
                source_context_sha256
            ),
        )
    )

    if (
        open_world_axis_plan
        is not None
    ):
        if not (
            open_world_axis_artifact
            and open_world_axis_artifact_sha256
            and open_world_axis_bundle
            is not None
        ):
            raise ValueError(
                "open-world frontier input "
                "requires plan artifact, "
                "artifact SHA, and bundle"
            )

        ideas.extend(
            axis_ideas(
                plan=(
                    open_world_axis_plan
                ),
                source_kind=(
                    "OPEN_WORLD_AXIS"
                ),
                source_artifact=(
                    open_world_axis_artifact
                ),
                source_artifact_sha256=(
                    open_world_axis_artifact_sha256
                ),
                source_context_id=(
                    source_context_id
                ),
                source_context_sha256=(
                    source_context_sha256
                ),
                open_world_bundle=(
                    open_world_axis_bundle
                ),
            )
        )

    if higher_order_topologies:
        if not (
            higher_order_topology_artifact
            and higher_order_topology_artifact_sha256
        ):
            raise ValueError(
                "higher-order topologies require "
                "artifact path and SHA"
            )

        ideas.extend(
            topology_ideas(
                topologies=(
                    higher_order_topologies
                ),
                source_kind=(
                    "HIGHER_ORDER"
                ),
                source_artifact=(
                    higher_order_topology_artifact
                ),
                source_artifact_sha256=(
                    higher_order_topology_artifact_sha256
                ),
                source_context_id=(
                    source_context_id
                ),
                source_context_sha256=(
                    source_context_sha256
                ),
                task_source=task_source,
                task_target=task_target,
            )
        )

    if direct_higher_order_topologies:
        if not (
            direct_higher_order_topology_artifact
            and direct_higher_order_topology_artifact_sha256
        ):
            raise ValueError(
                "direct higher-order topologies "
                "require artifact path and SHA"
            )

        ideas.extend(
            topology_ideas(
                topologies=(
                    direct_higher_order_topologies
                ),
                source_kind=(
                    "DIRECT_HIGHER_ORDER"
                ),
                source_artifact=(
                    direct_higher_order_topology_artifact
                ),
                source_artifact_sha256=(
                    direct_higher_order_topology_artifact_sha256
                ),
                source_context_id=(
                    source_context_id
                ),
                source_context_sha256=(
                    source_context_sha256
                ),
                task_source=task_source,
                task_target=task_target,
            )
        )

    if tension_candidates is not None:
        if not (
            tension_artifact
            and tension_artifact_sha256
        ):
            raise ValueError(
                "tension candidates require "
                "artifact path and SHA"
            )

        ideas.extend(
            tension_ideas(
                tensions=tension_candidates,
                source_artifact=(
                    tension_artifact
                ),
                source_artifact_sha256=(
                    tension_artifact_sha256
                ),
                source_context_id=(
                    source_context_id
                ),
                source_context_sha256=(
                    source_context_sha256
                ),
            )
        )

    if competing_explanations is not None:
        if not (
            competing_explanations_artifact
            and competing_explanations_artifact_sha256
        ):
            raise ValueError(
                "competing explanations require "
                "artifact path and SHA"
            )

        ideas.extend(
            competing_explanation_ideas(
                explanations=(
                    competing_explanations
                ),
                source_artifact=(
                    competing_explanations_artifact
                ),
                source_artifact_sha256=(
                    competing_explanations_artifact_sha256
                ),
                source_context_id=(
                    source_context_id
                ),
                source_context_sha256=(
                    source_context_sha256
                ),
            )
        )

    ideas.sort(
        key=lambda row: (
            _SOURCE_ORDER[
                row.source_kind
            ],
            _FORM_ORDER[
                row.idea_form
            ],
            row.source_lineage[
                0
            ].source_object_id,
            row.idea_id,
        )
    )

    exact_groups = _dedupe_groups(
        ideas
    )
    overlap = _overlap_diagnostics(
        ideas
    )
    structural_overlap = (
        _structural_overlap_diagnostics(
            ideas
        )
    )

    by_source = Counter(
        row.source_kind
        for row in ideas
    )
    by_form = Counter(
        row.idea_form
        for row in ideas
    )
    by_task = Counter(
        row.task_relation_mode
        for row in ideas
    )

    external_lineage_count = sum(
        any(
            lineage.external_literature_lineage
            for lineage
            in row.source_lineage
        )
        for row in ideas
    )

    candidate_lineage_count = sum(
        any(
            lineage.candidate_or_unverified_lineage
            for lineage
            in row.source_lineage
        )
        for row in ideas
    )

    population_id = _stable_id(
        "frontier_idea_population",
        source_context_id,
        source_context_sha256,
        *[
            row.idea_id
            for row in ideas
        ],
    )

    body = {
        "schema_version":
            "frontier-idea-population-v1",
        "population_id":
            population_id,
        "source_context_id":
            source_context_id,
        "source_context_sha256":
            source_context_sha256,
        "research_question":
            research_question,
        "task_source":
            task_source,
        "task_target":
            task_target,
        "ideas": [
            row.model_dump(
                mode="json"
            )
            for row in ideas
        ],
        "total_idea_count":
            len(ideas),
        "idea_count_by_source_kind":
            dict(
                sorted(
                    by_source.items()
                )
            ),
        "idea_count_by_idea_form":
            dict(
                sorted(
                    by_form.items()
                )
            ),
        "task_relation_mode_counts":
            dict(
                sorted(
                    by_task.items()
                )
            ),
        "exact_duplicate_groups": [
            row.model_dump(
                mode="json"
            )
            for row in exact_groups
        ],
        "cross_source_exact_duplicate_group_count":
            sum(
                row.cross_source
                for row
                in exact_groups
            ),
        "overlap_diagnostics": [
            row.model_dump(
                mode="json"
            )
            for row in overlap
        ],
        "cross_source_overlap_pair_count":
            len(overlap),
        "structural_overlap_diagnostics": [
            row.model_dump(
                mode="json"
            )
            for row in structural_overlap
        ],
        "cross_source_structural_overlap_pair_count":
            len(structural_overlap),
        "external_literature_lineage_idea_count":
            external_lineage_count,
        "candidate_or_unverified_lineage_idea_count":
            candidate_lineage_count,
        "ordering_policy":
            "SOURCE_FAMILY_THEN_UPSTREAM_ORDER_V1",
        "shadow_only":
            True,
        "new_llm_calls":
            False,
        "new_retrieval_calls":
            False,
        "positive_premise_authority_created":
            False,
        "novelty_authority_created":
            False,
        "generation_authority_created":
            False,
        "production_selection_authority":
            False,
        "stage8_input_changed":
            False,
        "canonical_graph_mutated":
            False,
    }

    population_sha256 = _sha256_text(
        _canonical_json(
            body
        )
    )

    return FrontierIdeaPopulation(
        **body,
        population_sha256=(
            population_sha256
        ),
    )


__all__ = [
    "FrontierCompetingExplanationSignature",
    "FrontierExactDuplicateGroup",
    "FrontierIdea",
    "FrontierIdeaForm",
    "FrontierIdeaPopulation",
    "FrontierIdeaSourceLineage",
    "FrontierOverlapDiagnostic",
    "FrontierRelationSignature",
    "FrontierStructuralMatchMethod",
    "FrontierStructuralOverlapDiagnostic",
    "FrontierSourceKind",
    "FrontierTaskRelationMode",
    "FrontierTensionSignature",
    "FrontierTopologySignature",
    "axis_ideas",
    "build_frontier_idea_population",
    "competing_explanation_ideas",
    "tension_ideas",
    "topology_ideas",
]
