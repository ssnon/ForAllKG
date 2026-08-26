from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

import networkx as nx

from domains.sers.bridge_signatures import (
    normalize_sers_bridge_text,
)
from domains.sers.context_contracts import (
    SERSContextBinding,
    SERSContextFact,
    SERSContextProvenance,
    SERSContextSignature,
)
from pipeline_core.discovery.discovery_contracts import (
    DiscoveryInspiration,
)
from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisEvidenceStatement,
)


class SERSContextCompilationError(ValueError):
    pass


_CLAIM_NODE_TYPES = frozenset({
    "MechanismClaim",
    "ObservationClaim",
    "BridgeConcept",
})


_STRUCTURAL_RELATIONS = frozenset({
    "HAS_COMPONENT",
    "HAS_SUPPORT",
    "HAS_MORPHOLOGY",
    "HAS_ARCHITECTURE",
    "HAS_STRUCTURAL_MOTIF",
})


_CONTEXT_RELATIONS = frozenset({
    *_STRUCTURAL_RELATIONS,
    "USES_ANALYTE",
    "USES_REPORTER",
    "USES_RAMAN_REPORTER",
    "USES_OPTICAL_CONDITION",
    "HAS_MEASUREMENT_GEOMETRY",
    "HAS_ENVIRONMENT",
})


_MATERIAL_NODE_TYPES = frozenset({
    "Metal",
    "Material",
})


def _stable_digest(
    payload: Any,
    *,
    length: int = 20,
) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(
        raw
    ).hexdigest()[:length]


def _infer_paper_id(
    node_id: str,
) -> str | None:
    if not node_id.startswith(
        "paper::"
    ):
        return None

    parts = node_id.split("::")

    if len(parts) < 3:
        return None

    return parts[1] or None


def _node_type(
    graph: nx.Graph,
    node_id: str,
) -> str:
    attrs = graph.nodes[node_id]

    return str(
        attrs.get("type")
        or attrs.get("node_type")
        or ""
    )


def _node_label(
    graph: nx.Graph,
    node_id: str,
) -> str:
    attrs = graph.nodes[node_id]

    return str(
        attrs.get("label")
        or attrs.get("statement")
        or attrs.get("metric")
        or node_id
    )


def _node_paper_ids(
    graph: nx.Graph,
    node_ids: Iterable[str],
) -> list[str]:
    values: set[str] = set()

    for node_id in node_ids:
        if node_id not in graph:
            continue

        attrs = graph.nodes[node_id]

        explicit = attrs.get(
            "source_paper_id"
        )

        if explicit:
            values.add(
                str(explicit)
            )

        raw_many = attrs.get(
            "source_paper_ids"
        )

        if isinstance(
            raw_many,
            (list, tuple, set),
        ):
            values.update(
                str(value)
                for value in raw_many
                if value
            )

        inferred = _infer_paper_id(
            str(node_id)
        )

        if inferred:
            values.add(
                inferred
            )

    return sorted(values)


def _edge_identifier(
    *,
    source: str,
    target: str,
    relation: str,
    key: Any,
    data: dict[str, Any],
) -> str:
    explicit = (
        data.get("edge_id")
        or data.get("relation_id")
        or data.get("id")
    )

    if explicit:
        return str(explicit)

    return (
        "context_edge:"
        + _stable_digest({
            "source": source,
            "relation": relation,
            "target": target,
            "key": str(key),
        })
    )


def _incoming_edges(
    graph: nx.Graph,
    node_id: str,
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    if graph.is_multigraph():
        iterator = graph.in_edges(
            node_id,
            keys=True,
            data=True,
        )

        for source, target, key, data in iterator:
            relation = str(
                data.get("relation")
                or data.get(
                    "relation_type"
                )
                or ""
            )

            rows.append({
                "source": str(source),
                "target": str(target),
                "key": key,
                "relation": relation,
                "data": dict(data),
                "edge_id":
                    _edge_identifier(
                        source=str(source),
                        target=str(target),
                        relation=relation,
                        key=key,
                        data=dict(data),
                    ),
            })

    else:
        iterator = graph.in_edges(
            node_id,
            data=True,
        )

        for source, target, data in iterator:
            relation = str(
                data.get("relation")
                or data.get(
                    "relation_type"
                )
                or ""
            )

            rows.append({
                "source": str(source),
                "target": str(target),
                "key": None,
                "relation": relation,
                "data": dict(data),
                "edge_id":
                    _edge_identifier(
                        source=str(source),
                        target=str(target),
                        relation=relation,
                        key=None,
                        data=dict(data),
                    ),
            })

    return sorted(
        rows,
        key=lambda row: (
            row["relation"],
            row["source"],
            row["target"],
            str(row["key"]),
            row["edge_id"],
        ),
    )


def _outgoing_edges(
    graph: nx.Graph,
    node_id: str,
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    if graph.is_multigraph():
        iterator = graph.out_edges(
            node_id,
            keys=True,
            data=True,
        )

        for source, target, key, data in iterator:
            relation = str(
                data.get("relation")
                or data.get(
                    "relation_type"
                )
                or ""
            )

            rows.append({
                "source": str(source),
                "target": str(target),
                "key": key,
                "relation": relation,
                "data": dict(data),
                "edge_id":
                    _edge_identifier(
                        source=str(source),
                        target=str(target),
                        relation=relation,
                        key=key,
                        data=dict(data),
                    ),
            })

    else:
        iterator = graph.out_edges(
            node_id,
            data=True,
        )

        for source, target, data in iterator:
            relation = str(
                data.get("relation")
                or data.get(
                    "relation_type"
                )
                or ""
            )

            rows.append({
                "source": str(source),
                "target": str(target),
                "key": None,
                "relation": relation,
                "data": dict(data),
                "edge_id":
                    _edge_identifier(
                        source=str(source),
                        target=str(target),
                        relation=relation,
                        key=None,
                        data=dict(data),
                    ),
            })

    return sorted(
        rows,
        key=lambda row: (
            row["relation"],
            row["target"],
            row["source"],
            str(row["key"]),
            row["edge_id"],
        ),
    )


def _is_gap_label(
    value: str,
) -> bool:
    text = (
        normalize_sers_bridge_text(
            value
        )
    )

    return bool(
        re.search(
            r"\b(?:"
            r"nano[- ]?gap|"
            r"nanoparticle gap|"
            r"interparticle gap|"
            r"particle gap|"
            r"interior gap"
            r")\b",
            text,
            re.I,
        )
    )


def _explicit_material_state(
    value: str,
) -> str | None:
    text = (
        normalize_sers_bridge_text(
            value
        )
    )

    if re.search(
        r"\b(?:"
        r"cu2o|"
        r"cuo|"
        r"copper oxide|"
        r"silver oxide|"
        r"gold oxide|"
        r"metallic|"
        r"zero[- ]?valent"
        r")\b",
        text,
        re.I,
    ):
        return value

    if re.search(
        r"\b(?:cu|ag|au)\s*\(\s*0\s*\)",
        text,
        re.I,
    ):
        return value

    return None


def _relation_context_semantics(
    *,
    relation: str,
    target_label: str,
) -> tuple[str, str] | None:
    if relation == "HAS_COMPONENT":
        return (
            "material_identity",
            "component",
        )

    if relation == "HAS_SUPPORT":
        return (
            "support",
            "support",
        )

    if relation == "HAS_MORPHOLOGY":
        return (
            "morphology",
            "morphology",
        )

    if relation == "HAS_ARCHITECTURE":
        return (
            "architecture",
            "architecture",
        )

    if relation == "HAS_STRUCTURAL_MOTIF":
        if _is_gap_label(
            target_label
        ):
            return (
                "gap_regime",
                "gap_regime",
            )

        return (
            "structural_motif",
            "structural_motif",
        )

    if relation == "USES_ANALYTE":
        return (
            "analyte",
            "analyte",
        )

    if relation in {
        "USES_REPORTER",
        "USES_RAMAN_REPORTER",
    }:
        return (
            "reporter",
            "reporter",
        )

    if relation == "USES_OPTICAL_CONDITION":
        return (
            "optical_condition",
            "optical_condition",
        )

    if relation == "HAS_MEASUREMENT_GEOMETRY":
        return (
            "measurement_geometry",
            "measurement_geometry",
        )

    if relation == "HAS_ENVIRONMENT":
        return (
            "environment",
            "environment",
        )

    return None


def _node_context_semantics(
    *,
    node_type: str,
    label: str,
) -> tuple[str, str] | None:
    if node_type == "PlasmonicSubstrate":
        return (
            "substrate",
            "plasmonic_substrate",
        )

    if node_type in _MATERIAL_NODE_TYPES:
        return (
            "material_identity",
            "component",
        )

    if node_type == "Support":
        return (
            "support",
            "support",
        )

    if node_type == "Morphology":
        return (
            "morphology",
            "morphology",
        )

    if node_type == "StructuralMotif":
        if _is_gap_label(
            label
        ):
            return (
                "gap_regime",
                "gap_regime",
            )

        return (
            "structural_motif",
            "structural_motif",
        )

    if node_type == "OpticalCondition":
        return (
            "optical_condition",
            "optical_condition",
        )

    if node_type == "Analyte":
        return (
            "analyte",
            "analyte",
        )

    if node_type == "RamanReporter":
        return (
            "reporter",
            "reporter",
        )

    return None


class _FactAccumulator:
    def __init__(
        self,
        *,
        source_ref_id: str,
    ) -> None:
        self.source_ref_id = (
            source_ref_id
        )

        self._rows: dict[
            tuple[
                str,
                str,
                str,
                str,
            ],
            dict[str, Any],
        ] = {}


    def add(
        self,
        *,
        dimension: str,
        scientific_role: str,
        knowledge_state: str,
        value: str | None,
        provenance: SERSContextProvenance,
        binding: SERSContextBinding | None = None,
        discriminator: str | None = None,
        tags: Iterable[str] = (),
    ) -> None:
        normalized = (
            normalize_sers_bridge_text(
                value
            )
            if value is not None
            else ""
        )

        semantic_discriminator = (
            discriminator
            if discriminator is not None
            else normalized
        )

        binding_key = (
            binding.model_dump_json()
            if binding is not None
            else ""
        )

        key = (
            dimension,
            scientific_role,
            knowledge_state,
            semantic_discriminator,
            binding_key,
        )

        row = self._rows.setdefault(
            key,
            {
                "dimension":
                    dimension,
                "scientific_role":
                    scientific_role,
                "knowledge_state":
                    knowledge_state,
                "value":
                    value,
                "normalized_value":
                    (
                        normalized
                        if value is not None
                        else None
                    ),
                "binding":
                    binding,
                "provenance": [],
                "tags": set(),
                "discriminator":
                    semantic_discriminator,
            },
        )

        provenance_key = (
            provenance.model_dump_json()
        )

        existing_keys = {
            item.model_dump_json()
            for item in row[
                "provenance"
            ]
        }

        if provenance_key not in existing_keys:
            row[
                "provenance"
            ].append(
                provenance
            )

        row["tags"].update(
            str(tag)
            for tag in tags
            if str(tag)
        )


    def build(
        self,
    ) -> list[SERSContextFact]:
        facts: list[
            SERSContextFact
        ] = []

        for key in sorted(
            self._rows
        ):
            row = self._rows[key]

            fact_id = (
                "sers_context_fact:"
                + _stable_digest({
                    "source_ref_id":
                        self.source_ref_id,
                    "dimension":
                        row["dimension"],
                    "scientific_role":
                        row[
                            "scientific_role"
                        ],
                    "knowledge_state":
                        row[
                            "knowledge_state"
                        ],
                    "discriminator":
                        row[
                            "discriminator"
                        ],
                    "binding":
                        (
                            row["binding"].model_dump(
                                mode="json"
                            )
                            if row["binding"]
                            is not None
                            else None
                        ),
                })
            )

            provenance = sorted(
                row["provenance"],
                key=lambda item:
                    item.model_dump_json(),
            )

            facts.append(
                SERSContextFact(
                    fact_id=fact_id,
                    dimension=row[
                        "dimension"
                    ],
                    scientific_role=row[
                        "scientific_role"
                    ],
                    knowledge_state=row[
                        "knowledge_state"
                    ],
                    value=row["value"],
                    normalized_value=row[
                        "normalized_value"
                    ],
                    binding=row[
                        "binding"
                    ],
                    provenance=provenance,
                    tags=sorted(
                        row["tags"]
                    ),
                )
            )

        return facts


class SERSContextCompiler:
    """Compile claim-local SERS scientific context without LLM calls.

    The compiler deliberately performs only bounded graph closure.

    Axis source:
        BridgeConcept
        <- GROUNDS_SEMANTIC_CANDIDATE <- exact anchors
        -> direct context from non-claim anchors
        -> direct APPLIES_TO targets from claim anchors
        -> direct context from those targets

    Grounded source:
        scientific_support_node_ids
        -> APPLIES_TO targets
        -> direct context from roots/targets

    Whole-paper inheritance is explicitly forbidden.
    """

    def __init__(
        self,
        *,
        graph: nx.Graph,
        domain_profile_id: str,
    ) -> None:
        if not graph.is_directed():
            raise ValueError(
                "SERS context compiler requires "
                "a directed graph"
            )

        if not (
            domain_profile_id or ""
        ).strip():
            raise ValueError(
                "domain_profile_id is required"
            )

        self.graph = graph
        self.domain_profile_id = (
            str(domain_profile_id)
        )


    def _require_node(
        self,
        node_id: str,
    ) -> None:
        if node_id not in self.graph:
            raise SERSContextCompilationError(
                "missing graph node: "
                f"{node_id}"
            )


    def _provenance(
        self,
        *,
        kind: str,
        node_ids: Iterable[str],
        edge_ids: Iterable[str] = (),
        statement_ids: Iterable[str] = (),
        candidate_unit_ids: Iterable[str] = (),
    ) -> SERSContextProvenance:
        node_ids = sorted({
            str(value)
            for value in node_ids
            if str(value)
        })

        for node_id in node_ids:
            self._require_node(
                node_id
            )

        return SERSContextProvenance(
            kind=kind,
            node_ids=node_ids,
            edge_ids=sorted({
                str(value)
                for value in edge_ids
                if str(value)
            }),
            paper_ids=(
                _node_paper_ids(
                    self.graph,
                    node_ids,
                )
            ),
            statement_ids=sorted({
                str(value)
                for value
                in statement_ids
                if str(value)
            }),
            candidate_unit_ids=sorted({
                str(value)
                for value
                in candidate_unit_ids
                if str(value)
            }),
        )


    def _add_node_fact(
        self,
        *,
        accumulator: _FactAccumulator,
        node_id: str,
        provenance_kind: str,
        statement_ids: Iterable[str] = (),
        candidate_unit_ids: Iterable[str] = (),
    ) -> None:
        self._require_node(
            node_id
        )

        label = _node_label(
            self.graph,
            node_id,
        )

        semantics = (
            _node_context_semantics(
                node_type=_node_type(
                    self.graph,
                    node_id,
                ),
                label=label,
            )
        )

        if semantics is None:
            return

        dimension, role = semantics

        provenance = self._provenance(
            kind=provenance_kind,
            node_ids=[
                node_id
            ],
            statement_ids=(
                statement_ids
            ),
            candidate_unit_ids=(
                candidate_unit_ids
            ),
        )

        accumulator.add(
            dimension=dimension,
            scientific_role=role,
            knowledge_state="explicit",
            value=label,
            provenance=provenance,
            binding=SERSContextBinding(
                basis="node",
                owner_ref_id=node_id,
                owner_label=label,
                owner_type=_node_type(
                    self.graph,
                    node_id,
                ),
            ),
        )


    def _add_direct_context_edges(
        self,
        *,
        accumulator: _FactAccumulator,
        source_node_id: str,
        provenance_kind: str,
        statement_ids: Iterable[str] = (),
        candidate_unit_ids: Iterable[str] = (),
    ) -> None:
        self._require_node(
            source_node_id
        )

        for edge in _outgoing_edges(
            self.graph,
            source_node_id,
        ):
            relation = edge[
                "relation"
            ]

            if relation not in (
                _CONTEXT_RELATIONS
            ):
                continue

            target_id = edge[
                "target"
            ]

            self._require_node(
                target_id
            )

            target_label = (
                _node_label(
                    self.graph,
                    target_id,
                )
            )

            semantics = (
                _relation_context_semantics(
                    relation=relation,
                    target_label=(
                        target_label
                    ),
                )
            )

            if semantics is None:
                continue

            dimension, role = semantics

            provenance = self._provenance(
                kind=provenance_kind,
                node_ids=[
                    source_node_id,
                    target_id,
                ],
                edge_ids=[
                    edge["edge_id"]
                ],
                statement_ids=(
                    statement_ids
                ),
                candidate_unit_ids=(
                    candidate_unit_ids
                ),
            )

            accumulator.add(
                dimension=dimension,
                scientific_role=role,
                knowledge_state="explicit",
                value=target_label,
                provenance=provenance,
                binding=SERSContextBinding(
                    basis="direct_edge",
                    owner_ref_id=source_node_id,
                    owner_label=_node_label(
                        self.graph,
                        source_node_id,
                    ),
                    owner_type=_node_type(
                        self.graph,
                        source_node_id,
                    ),
                    relation=relation,
                ),
            )

            # Material state is intentionally explicit-or-unknown.
            #
            # Crucially this consults only the directly connected material
            # node. A CuO node elsewhere in the same paper cannot leak into
            # this signature.
            if (
                relation
                == "HAS_COMPONENT"
                and _node_type(
                    self.graph,
                    target_id,
                )
                in _MATERIAL_NODE_TYPES
            ):
                subject_tag = (
                    "material_subject:"
                    + normalize_sers_bridge_text(
                        target_label
                    )
                )

                state = (
                    _explicit_material_state(
                        target_label
                    )
                )

                if state is None:
                    accumulator.add(
                        dimension="material_state",
                        scientific_role="material_state",
                        knowledge_state="unknown",
                        value=None,
                        provenance=provenance,
                        binding=SERSContextBinding(
                            basis="derived_material_state",
                            owner_ref_id=target_id,
                            owner_label=target_label,
                            owner_type=_node_type(
                                self.graph,
                                target_id,
                            ),
                        ),
                        discriminator=(
                            subject_tag
                        ),
                        tags=[
                            subject_tag
                        ],
                    )

                else:
                    accumulator.add(
                        dimension="material_state",
                        scientific_role="material_state",
                        knowledge_state="explicit",
                        value=state,
                        provenance=provenance,
                        binding=SERSContextBinding(
                            basis="derived_material_state",
                            owner_ref_id=target_id,
                            owner_label=target_label,
                            owner_type=_node_type(
                                self.graph,
                                target_id,
                            ),
                        ),
                        discriminator=(
                            subject_tag
                            + ":"
                            + normalize_sers_bridge_text(
                                state
                            )
                        ),
                        tags=[
                            subject_tag
                        ],
                    )


    def _build_signature(
        self,
        *,
        scope: str,
        source_ref_id: str,
        accumulator: _FactAccumulator,
    ) -> SERSContextSignature:
        facts = accumulator.build()

        if not facts:
            raise SERSContextCompilationError(
                "claim-local closure produced "
                "no SERS context facts for "
                f"{source_ref_id}"
            )

        signature_id = (
            "sers_context_signature:"
            + _stable_digest({
                "domain_profile_id":
                    self.domain_profile_id,
                "scope":
                    scope,
                "source_ref_id":
                    source_ref_id,
                "fact_ids": [
                    row.fact_id
                    for row in facts
                ],
            })
        )

        return SERSContextSignature(
            signature_id=signature_id,
            domain_profile_id=(
                self.domain_profile_id
            ),
            scope=scope,
            source_ref_id=(
                source_ref_id
            ),
            facts=facts,
        )


    def compile_axis_inspiration(
        self,
        inspiration: DiscoveryInspiration,
    ) -> SERSContextSignature:
        source_ref_id = (
            inspiration.inspiration_id
        )

        candidate_label = (
            inspiration.candidate_unit_label
        )

        if not candidate_label:
            raise SERSContextCompilationError(
                "axis inspiration has no "
                "candidate_unit_label"
            )

        inspiration_nodes = {
            str(node_id)
            for node_id
            in inspiration.node_ids
        }

        bridge_ids: list[str] = []

        for node_id in sorted(
            inspiration_nodes
        ):
            self._require_node(
                node_id
            )

            if (
                _node_type(
                    self.graph,
                    node_id,
                )
                == "BridgeConcept"
                and _node_label(
                    self.graph,
                    node_id,
                )
                == candidate_label
            ):
                bridge_ids.append(
                    node_id
                )

        if len(bridge_ids) != 1:
            raise SERSContextCompilationError(
                "expected exactly one "
                "candidate BridgeConcept for "
                f"{candidate_label!r}; "
                f"found {len(bridge_ids)}"
            )

        bridge_id = bridge_ids[0]

        # Candidate scientific context is defined by the exact
        # BridgeConcept's direct semantic grounding closure.
        #
        # Do NOT intersect these anchors with inspiration.node_ids.
        # DiscoveryInspiration.node_ids is traversal/path provenance and
        # may omit auxiliary semantic anchors that directly ground the
        # candidate BridgeConcept (for example, the 3D-Si support anchor
        # in the inserted-pyramid SERS candidate).
        #
        # Whole-paper inheritance is still forbidden: only nodes with a
        # direct GROUNDS_SEMANTIC_CANDIDATE edge into this exact bridge
        # are admitted.
        anchor_edges = [
            edge
            for edge in _incoming_edges(
                self.graph,
                bridge_id,
            )
            if (
                edge["relation"]
                == "GROUNDS_SEMANTIC_CANDIDATE"
            )
        ]

        anchor_ids = sorted({
            edge["source"]
            for edge in anchor_edges
        })

        if not anchor_ids:
            raise SERSContextCompilationError(
                "candidate BridgeConcept has no "
                "source-local grounding anchors"
            )

        expected_lineage_anchors = {
            str(value)
            for value in (
                inspiration.candidate_entry_anchor_id,
                inspiration.candidate_exit_anchor_id,
            )
            if str(value)
        }

        missing_lineage = (
            expected_lineage_anchors
            - set(anchor_ids)
        )

        if missing_lineage:
            raise SERSContextCompilationError(
                "candidate entry/exit lineage is "
                "not grounded to BridgeConcept: "
                + ", ".join(
                    sorted(
                        missing_lineage
                    )
                )
            )

        accumulator = _FactAccumulator(
            source_ref_id=(
                source_ref_id
            )
        )

        candidate_unit_ids = [
            inspiration.candidate_unit_id
        ] if (
            inspiration.candidate_unit_id
        ) else []

        context_roots: dict[
            str,
            str,
        ] = {}

        for anchor_id in anchor_ids:
            anchor_type = _node_type(
                self.graph,
                anchor_id,
            )

            # Non-claim semantic anchors already carry context directly.
            # Claim anchors instead expose their scientific owner through
            # the same direct APPLIES_TO closure used by grounded-premise
            # compilation.
            if anchor_type not in _CLAIM_NODE_TYPES:
                context_roots[
                    anchor_id
                ] = "axis_anchor"

            for edge in _outgoing_edges(
                self.graph,
                anchor_id,
            ):
                if (
                    edge["relation"]
                    != "APPLIES_TO"
                ):
                    continue

                target_id = edge[
                    "target"
                ]

                self._require_node(
                    target_id
                )

                # This remains strictly claim-local: only a direct
                # APPLIES_TO target of an exact candidate-grounding
                # anchor is admitted. Whole-paper inheritance remains
                # forbidden.
                context_roots[
                    target_id
                ] = "axis_direct_claim"

        for root_id in sorted(
            context_roots
        ):
            root_kind = (
                context_roots[
                    root_id
                ]
            )

            self._add_node_fact(
                accumulator=accumulator,
                node_id=root_id,
                provenance_kind=(
                    root_kind
                ),
                candidate_unit_ids=(
                    candidate_unit_ids
                ),
            )

            self._add_direct_context_edges(
                accumulator=accumulator,
                source_node_id=root_id,
                provenance_kind=(
                    "axis_structural_edge"
                ),
                candidate_unit_ids=(
                    candidate_unit_ids
                ),
            )

        return self._build_signature(
            scope="axis_inspiration",
            source_ref_id=(
                source_ref_id
            ),
            accumulator=accumulator,
        )


    def compile_grounded_statement(
        self,
        statement: HypothesisEvidenceStatement,
    ) -> SERSContextSignature:
        source_ref_id = (
            statement.statement_id
        )

        support_ids = sorted({
            str(node_id)
            for node_id
            in statement.scientific_support_node_ids
        })

        if not support_ids:
            raise SERSContextCompilationError(
                "grounded statement has no "
                "scientific_support_node_ids: "
                f"{statement.statement_id}"
            )

        for node_id in support_ids:
            self._require_node(
                node_id
            )

        context_roots: dict[
            str,
            str,
        ] = {}

        for support_id in support_ids:
            support_type = _node_type(
                self.graph,
                support_id,
            )

            if support_type not in (
                _CLAIM_NODE_TYPES
            ):
                context_roots[
                    support_id
                ] = (
                    "grounded_support_node"
                )

            for edge in _outgoing_edges(
                self.graph,
                support_id,
            ):
                if (
                    edge["relation"]
                    != "APPLIES_TO"
                ):
                    continue

                target_id = edge[
                    "target"
                ]

                self._require_node(
                    target_id
                )

                context_roots[
                    target_id
                ] = (
                    "grounded_applies_to_target"
                )

        accumulator = _FactAccumulator(
            source_ref_id=(
                source_ref_id
            )
        )

        for root_id in sorted(
            context_roots
        ):
            root_kind = (
                context_roots[
                    root_id
                ]
            )

            self._add_node_fact(
                accumulator=accumulator,
                node_id=root_id,
                provenance_kind=(
                    root_kind
                ),
                statement_ids=[
                    statement.statement_id
                ],
            )

            self._add_direct_context_edges(
                accumulator=accumulator,
                source_node_id=root_id,
                provenance_kind=(
                    "grounded_structural_edge"
                ),
                statement_ids=[
                    statement.statement_id
                ],
            )

        return self._build_signature(
            scope="grounded_premise",
            source_ref_id=(
                source_ref_id
            ),
            accumulator=accumulator,
        )
