from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from pipeline_core.discovery.discovery_axis_contracts import (
    DiscoveryAxisPlan,
    DiscoveryAxisPlannerPolicy,
)
from pipeline_core.discovery.discovery_axis_planner import (
    DiscoveryAxisPlanner,
    _sha256_json,
    _stable_id,
)
from pipeline_core.discovery.discovery_bundle import (
    DiscoveryBundleBuilder,
)
from pipeline_core.discovery.discovery_contracts import (
    DiscoveryBundle,
    DiscoveryInspiration,
)
from pipeline_core.discovery.dual_hypothesis_context import (
    DualHypothesisContext,
)
from pipeline_core.discovery.relation_component_composition import (
    EndpointEquivalenceWitness,
    MediatorEquivalenceWitness,
    RelationComponentAuthority,
    RelationComponentView,
    candidate_inspiration_component,
    compose_relation_component_topologies,
    confirmed_known_component_from_mapping,
    topology_has_materializable_endpoint_fidelity,
    topology_to_task_bridge_composite,
)
from pipeline_core.discovery.task_bridge_candidate_composition import (
    CandidateRelationView,
    candidate_relation_from_mapping,
    compose_task_bridge_candidates,
)
from pipeline_core.discovery.task_bridge_composite_axis import (
    materialize_task_bridge_composite_axis,
)

import scripts.discovery.build_discovery_bundle as bundle_cli


GRAMMAR = re.compile(
    r"^How does (.+?) relate to (.+?)\?$",
    flags=re.DOTALL,
)

MAX_TASK_AXES = 2
MAX_TOTAL_AXES = 3

TASK_AXIS_SOURCE_MODES = {
    "task_conditioned_composite_bridge_projection",
    "task_conditioned_relation_component_topology",
}

MIN_EXPLORATION_SCORE = 0.05
MAX_GROUNDING_SEMANTIC_OVERLAP = 0.95
MAX_CONTEXT_SWITCH_PENALTY = 0.50


def _normalized_endpoint_text(value: object) -> str:
    return " ".join(
        str(value or "").split()
    ).casefold()


def _load_endpoint_equivalences(
    path: Path | None,
    *,
    requested_source: str,
    requested_target: str,
) -> tuple[EndpointEquivalenceWitness, ...]:
    # Task-scoped and fail-closed. No scientific synonymy is inferred.
    if path is None:
        return ()

    if not path.is_file():
        raise ValueError(
            "endpoint-equivalence file does not exist: "
            + str(path)
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "endpoint-equivalence file must contain a JSON object"
        )

    if (
        payload.get("schema_version")
        != "endpoint-equivalence-set-v1"
    ):
        raise ValueError(
            "endpoint-equivalence file has unsupported schema_version"
        )

    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise ValueError(
            "endpoint-equivalence file requires object scope"
        )

    expected_source = _normalized_endpoint_text(
        requested_source
    )
    expected_target = _normalized_endpoint_text(
        requested_target
    )
    file_source = _normalized_endpoint_text(
        scope.get("requested_source")
    )
    file_target = _normalized_endpoint_text(
        scope.get("requested_target")
    )

    if file_source != expected_source:
        raise ValueError(
            "endpoint-equivalence source scope does not match task: "
            f"{file_source!r} != {expected_source!r}"
        )

    if file_target != expected_target:
        raise ValueError(
            "endpoint-equivalence target scope does not match task: "
            f"{file_target!r} != {expected_target!r}"
        )

    rows = payload.get("witnesses")
    if not isinstance(rows, list):
        raise ValueError(
            "endpoint-equivalence witnesses must be a list"
        )

    witnesses = []
    seen = set()

    for index, row in enumerate(rows):
        try:
            witness = EndpointEquivalenceWitness.model_validate(
                row
            )
        except Exception as exc:
            raise ValueError(
                "endpoint-equivalence witness failed validation: "
                f"index={index}"
            ) from exc

        if witness.witness_id in seen:
            raise ValueError(
                "duplicate endpoint-equivalence witness_id: "
                + witness.witness_id
            )

        if not witness.provenance_ids:
            raise ValueError(
                "endpoint-equivalence witness requires provenance_ids: "
                + witness.witness_id
            )

        seen.add(witness.witness_id)
        witnesses.append(witness)

    return tuple(witnesses)


def _normalize_mediator_identity(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _mediator_pair_key(
    witness: MediatorEquivalenceWitness,
) -> tuple[str, str]:
    left = _normalize_mediator_identity(witness.left_mediator)
    right = _normalize_mediator_identity(witness.right_mediator)
    return tuple(sorted((left, right)))


def _merge_mediator_equivalences(
    *groups: tuple[MediatorEquivalenceWitness, ...],
) -> tuple[MediatorEquivalenceWitness, ...]:
    by_pair = {}
    by_id = {}
    for group in groups:
        for witness in group:
            existing_id = by_id.get(witness.witness_id)
            if existing_id is not None and existing_id != witness:
                raise ValueError(
                    "conflicting mediator-equivalence witness_id: "
                    + witness.witness_id
                )
            by_id[witness.witness_id] = witness
            pair = _mediator_pair_key(witness)
            existing = by_pair.get(pair)
            if existing is None:
                by_pair[pair] = witness
                continue
            if (
                _normalize_mediator_identity(existing.canonical_mediator)
                != _normalize_mediator_identity(witness.canonical_mediator)
            ):
                raise ValueError(
                    "conflicting canonical mediator for equivalent pair: "
                    + repr(pair)
                )
            if witness.witness_id < existing.witness_id:
                by_pair[pair] = witness
    return tuple(by_pair[key] for key in sorted(by_pair))


def _profile_mediator_equivalences(
    domain_profile: str,
) -> tuple[MediatorEquivalenceWitness, ...]:
    profile = str(domain_profile or "").strip().casefold()
    if profile != "sers_au_ag":
        return ()
    return (
        MediatorEquivalenceWitness(
            witness_id=(
                "mediator:eq:profile:sers:"
                "surface-enhanced-raman-scattering"
            ),
            left_mediator="surface-enhanced Raman scattering",
            right_mediator="SERS",
            canonical_mediator="sers",
            witness_kind="domain_profile_normalization",
            provenance_ids=[
                "domain_profile:sers_au_ag:surface-enhanced Raman scattering=SERS"
            ],
        ),
    )


def _used_endpoint_equivalence_witness_ids(
    topologies: tuple[object, ...],
) -> list[str]:
    used = set()

    for topology in topologies:
        for binding in (
            topology.source_binding,
            topology.target_binding,
        ):
            witness_id = (
                binding.equivalence_witness_id
            )
            if witness_id:
                used.add(str(witness_id))

    return sorted(used)


def _endpoint_fidelity_counts(
    topologies: tuple[object, ...],
) -> tuple[int, int]:
    exact = 0
    equivalent = 0

    for topology in topologies:
        authorities = {
            topology.source_binding.binding_authority,
            topology.target_binding.binding_authority,
        }
        if authorities == {"exact"}:
            exact += 1
        elif "partial" not in authorities:
            equivalent += 1

    return exact, equivalent


def _load_confirmed_known_components(
    path: Path,
) -> tuple[
    RelationComponentView,
    ...,
]:
    if not path.is_file():
        raise ValueError(
            "accepted-pattern table does not exist: "
            + str(path)
        )

    components = []
    seen = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise ValueError(
                "accepted-pattern table has no header"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            try:
                component = (
                    confirmed_known_component_from_mapping(
                        row
                    )
                )
            except Exception as exc:
                raise ValueError(
                    "accepted-pattern row failed "
                    "fail-closed validation: "
                    f"row={row_number}"
                ) from exc

            if (
                component.component_id
                in seen
            ):
                continue

            seen.add(
                component.component_id
            )
            components.append(
                component
            )

    return tuple(
        components
    )


def _json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _write_model(
    path: Path,
    model: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        model.model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )


def _lane_id(
    *,
    question: str,
    unit_id: str,
    path_id: str,
) -> str:
    payload = (
        question
        + "|"
        + unit_id
        + "|"
        + path_id
    )
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]
    return (
        "task_lane_source:"
        + digest
    )


def _parse_question(
    question: str,
) -> tuple[str, str] | None:
    match = GRAMMAR.fullmatch(
        question.strip()
    )
    if match is None:
        return None

    source = match.group(1).strip()
    target = match.group(2).strip()

    if not source or not target:
        return None

    return source, target


def _resolve_task_endpoints(
    *,
    question: str,
    requested_source: str | None,
    requested_target: str | None,
) -> tuple[str, str] | None:
    source_supplied = (
        requested_source is not None
    )
    target_supplied = (
        requested_target is not None
    )

    if source_supplied != target_supplied:
        raise ValueError(
            "Explicit task endpoints must be "
            "provided together."
        )

    if source_supplied:
        assert requested_source is not None
        assert requested_target is not None

        source = requested_source.strip()
        target = requested_target.strip()

        if not source or not target:
            raise ValueError(
                "Explicit task endpoints must "
                "be non-empty."
            )

        return source, target

    return _parse_question(
        question
    )


def _endpoint_resolution_observability(
    *,
    question: str,
    requested_source: str | None,
    requested_target: str | None,
) -> tuple[str, bool]:
    explicit_pair = (
        requested_source is not None
        and requested_target is not None
    )

    mode = (
        "EXPLICIT_RUNNER_ENDPOINTS_V1"
        if explicit_pair
        else "LEGACY_QUESTION_GRAMMAR_V1"
    )

    return (
        mode,
        _parse_question(question) is not None,
    )


def _candidate_mapping(
    row: dict[str, Any],
) -> dict[str, Any] | None:
    value = row.get(
        "candidate_unit"
    )
    if isinstance(value, dict):
        return value

    value = row.get(
        "candidate_unit_selection"
    )
    if isinstance(value, dict):
        candidate = value.get(
            "candidate_unit"
        )
        if isinstance(candidate, dict):
            return candidate

    return None


def _candidate_unit_id(
    row: dict[str, Any],
) -> str:
    unit = _candidate_mapping(row)

    if unit is not None:
        return str(
            unit.get("unit_id", "")
            or
            unit.get("candidate_unit_id", "")
            or
            ""
        )

    return ""


def _sanitize_lane_inspiration(
    *,
    inspiration: DiscoveryInspiration,
    inspiration_id: str,
) -> DiscoveryInspiration:
    payload = inspiration.model_dump(
        mode="json"
    )

    payload["inspiration_id"] = (
        inspiration_id
    )

    payload["source_mode"] = (
        "task_composite_source_lane_full_pool"
    )

    payload[
        "max_semantic_similarity_to_selected"
    ] = 0.0

    payload[
        "semantic_diversity_mode"
    ] = "disabled"

    reasons = [
        str(value)
        for value in payload.get(
            "reason_codes",
            [],
        )
        if (
            not str(value).startswith(
                "bundle_rank:"
            )
            and
            str(value)
            !=
            "semantic_diversity_relaxed"
        )
    ]

    reasons.extend(
        [
            "task_composite_source_lane",
            "full_pool_representative",
            "not_generic_bundle_selected",
        ]
    )

    payload["reason_codes"] = sorted(
        set(reasons)
    )

    return DiscoveryInspiration.model_validate(
        payload
    )


def _quality_eligible(
    inspiration: DiscoveryInspiration,
    *,
    min_candidate_unit_score: float,
) -> bool:
    return bool(
        inspiration.path_type
        == "CANDIDATE_EXPLORATION"
        and
        inspiration.candidate_unit_id
        and
        float(
            inspiration.candidate_unit_score
        )
        >= float(
            min_candidate_unit_score
        )
        and
        float(
            inspiration.exploration_score
        )
        >= MIN_EXPLORATION_SCORE
        and
        float(
            inspiration.semantic_similarity_to_grounding
        )
        <= MAX_GROUNDING_SEMANTIC_OVERLAP
        and
        float(
            inspiration.context_switch_penalty
        )
        <= MAX_CONTEXT_SWITCH_PENALTY
    )


def _replay_and_capture(
    *,
    final_traversal: Path,
    candidate_traversal: Path,
    domain_profile: str,
    discovery_top_k: int,
    min_candidate_unit_score: float,
    expected_bundle: DiscoveryBundle,
    scratch_bundle: Path,
) -> tuple[
    DiscoveryBundleBuilder,
    DiscoveryBundle,
    list[dict[str, Any]],
]:
    captured: list[
        dict[str, Any]
    ] = []

    builders: list[
        DiscoveryBundleBuilder
    ] = []

    original = (
        DiscoveryBundleBuilder
        ._enrich_candidate_path
    )

    def wrapped(
        self: DiscoveryBundleBuilder,
        **kwargs: Any,
    ) -> dict[str, Any]:
        enriched = original(
            self,
            **kwargs,
        )

        if not builders:
            builders.append(self)

        captured.append(
            enriched
        )

        return enriched

    old_argv = list(sys.argv)

    DiscoveryBundleBuilder._enrich_candidate_path = (
        wrapped
    )

    try:
        sys.argv = [
            "build_discovery_bundle",
            "--traversal",
            str(final_traversal),
            "--traversal",
            str(candidate_traversal),
            "--domain-profile",
            domain_profile,
            "--top-k",
            str(discovery_top_k),
            "--min-reserved-candidate-unit-score",
            str(min_candidate_unit_score),
            "--output",
            str(scratch_bundle),
        ]

        rc = bundle_cli.main()

    finally:
        sys.argv = old_argv
        (
            DiscoveryBundleBuilder
            ._enrich_candidate_path
        ) = original

    if rc not in (
        0,
        None,
    ):
        raise RuntimeError(
            "DiscoveryBundle replay failed: "
            f"rc={rc}"
        )

    if not builders:
        raise RuntimeError(
            "DiscoveryBundle replay captured no builder"
        )

    replay = DiscoveryBundle.model_validate_json(
        scratch_bundle.read_text(
            encoding="utf-8"
        )
    )

    if (
        replay.bundle_sha256
        !=
        expected_bundle.bundle_sha256
    ):
        raise RuntimeError(
            "A17F replay changed the generic "
            "DiscoveryBundle SHA: "
            f"{replay.bundle_sha256} != "
            f"{expected_bundle.bundle_sha256}"
        )

    return (
        builders[0],
        replay,
        captured,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen A17F task-conditioned "
            "2-task + 1-generic discovery-axis plan "
            "without changing DiscoveryBundle selection."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
    )
    parser.add_argument(
        "--requested-source",
        default=None,
    )
    parser.add_argument(
        "--requested-target",
        default=None,
    )
    parser.add_argument(
        "--final-traversal",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--candidate-traversal",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--discovery-bundle",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--dual-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--accepted-patterns",
        default=None,
        type=Path,
        help=(
            "Optional bridge_patterns.csv containing "
            "validated accepted RelationPattern rows. "
            "When omitted, the frozen candidate-only "
            "task-composition path is unchanged."
        ),
    )
    parser.add_argument(
        "--endpoint-equivalences",
        default=None,
        type=Path,
        help=(
            "Optional task-scoped endpoint-equivalence-set-v1 JSON. "
            "Used only with --accepted-patterns. Equivalence remains "
            "explicit and auditable; it is never inferred."
        ),
    )
    parser.add_argument(
        "--domain-profile",
        required=True,
    )
    parser.add_argument(
        "--discovery-top-k",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--min-candidate-unit-score",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--max-axes",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--output-dual-context",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-axis-plan",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-report",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    bundle = (
        DiscoveryBundle
        .model_validate_json(
            args.discovery_bundle.read_text(
                encoding="utf-8"
            )
        )
    )

    old_dual = (
        DualHypothesisContext
        .model_validate_json(
            args.dual_context.read_text(
                encoding="utf-8"
            )
        )
    )

    planner = DiscoveryAxisPlanner(
        DiscoveryAxisPlannerPolicy(
            max_axes=args.max_axes,
        )
    )

    generic_plan = planner.build(
        old_dual
    )

    parsed = _resolve_task_endpoints(
        question=args.question,
        requested_source=(
            args.requested_source
        ),
        requested_target=(
            args.requested_target
        ),
    )

    (
        endpoint_resolution_mode,
        question_grammar_matched,
    ) = _endpoint_resolution_observability(
        question=args.question,
        requested_source=(
            args.requested_source
        ),
        requested_target=(
            args.requested_target
        ),
    )

    # --------------------------------------------------------------
    # Neither explicit endpoints nor exact legacy grammar apply:
    # preserve old production behavior exactly.
    # --------------------------------------------------------------
    if parsed is None:
        _write_model(
            args.output_dual_context,
            old_dual,
        )
        _write_model(
            args.output_axis_plan,
            generic_plan,
        )

        args.output_report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output_report.write_text(
            json.dumps(
                {
                    "schema_version":
                        "task-conditioned-axis-plan-report-v1",
                    "status":
                        "GRAMMAR_NOT_APPLICABLE",
                    "grammar":
                        "HOW_DOES_RELATE_TO_GRAMMAR_V1",
                    "endpoint_resolution_mode":
                        endpoint_resolution_mode,
                    "requested_source":
                        None,
                    "requested_target":
                        None,
                    "question_grammar_matched":
                        question_grammar_matched,
                    "generic_bundle_changed":
                        False,
                    "task_axis_count":
                        0,
                    "generic_axis_count":
                        len(generic_plan.axes),
                    "architecture_tuning":
                        False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "A17F_STATUS=GRAMMAR_NOT_APPLICABLE"
        )
        print(
            "TASK_AXIS_COUNT=0"
        )
        print(
            "GENERIC_BUNDLE_CHANGED=False"
        )
        return 0

    requested_source, requested_target = (
        parsed
    )

    if (
        args.endpoint_equivalences is not None
        and args.accepted_patterns is None
    ):
        raise ValueError(
            "--endpoint-equivalences requires --accepted-patterns"
        )

    endpoint_equivalences = (
        _load_endpoint_equivalences(
            args.endpoint_equivalences,
            requested_source=requested_source,
            requested_target=requested_target,
        )
    )

    mediator_equivalences = _merge_mediator_equivalences(
        _profile_mediator_equivalences(args.domain_profile),
    )

    scratch_bundle = (
        args.output_report.parent
        / ".a17f.generic_bundle.replay.json"
    )

    (
        builder,
        replay_bundle,
        enriched_rows,
    ) = _replay_and_capture(
        final_traversal=(
            args.final_traversal
        ),
        candidate_traversal=(
            args.candidate_traversal
        ),
        domain_profile=(
            args.domain_profile
        ),
        discovery_top_k=(
            args.discovery_top_k
        ),
        min_candidate_unit_score=(
            args.min_candidate_unit_score
        ),
        expected_bundle=bundle,
        scratch_bundle=scratch_bundle,
    )

    # --------------------------------------------------------------
    # Materialize every captured candidate path with neutral
    # non-selection semantics, then apply the frozen quality gates.
    # --------------------------------------------------------------
    rows_by_unit: dict[
        str,
        list[
            tuple[
                DiscoveryInspiration,
                dict[str, Any],
            ]
        ],
    ] = {}

    relation_by_unit: dict[
        str,
        CandidateRelationView,
    ] = {}

    for enriched in enriched_rows:
        unit = _candidate_mapping(
            enriched
        )

        if unit is None:
            continue

        unit_id = _candidate_unit_id(
            enriched
        )

        if not unit_id:
            continue

        relation = (
            candidate_relation_from_mapping(
                unit
            )
        )

        relation_by_unit.setdefault(
            unit_id,
            relation,
        )

        materialized = (
            builder._materialize_inspiration(
                corpus_id=(
                    bundle.corpus_id
                ),
                rank=1,
                row=enriched,
                semantic_mode=(
                    replay_bundle
                    .semantic_diversity_mode
                ),
            )
        )

        rows_by_unit.setdefault(
            unit_id,
            [],
        ).append(
            (
                materialized,
                enriched,
            )
        )

    representatives: dict[
        str,
        tuple[
            DiscoveryInspiration,
            dict[str, Any],
        ],
    ] = {}

    for unit_id, rows in (
        rows_by_unit.items()
    ):
        eligible = [
            pair
            for pair in rows
            if _quality_eligible(
                pair[0],
                min_candidate_unit_score=(
                    args
                    .min_candidate_unit_score
                ),
            )
        ]

        if not eligible:
            continue

        eligible.sort(
            key=lambda pair: (
                -float(
                    pair[0]
                    .exploration_score
                ),
                str(
                    pair[0]
                    .source_path_id
                ),
            )
        )

        representatives[
            unit_id
        ] = eligible[0]

    known_components: tuple[
        RelationComponentView,
        ...,
    ] = ()

    candidate_components: tuple[
        RelationComponentView,
        ...,
    ] = ()

    relation_component_topologies = ()
    endpoint_fidelity_topologies = ()
    relation_component_composites = ()
    exact_endpoint_topology_count = 0
    equivalent_endpoint_topology_count = 0
    used_endpoint_equivalence_witness_ids: list[str] = []

    if (
        args.accepted_patterns
        is not None
    ):
        known_components = (
            _load_confirmed_known_components(
                args.accepted_patterns
            )
        )

        candidate_rows = []

        for unit_id in sorted(
            representatives
        ):
            relation = (
                relation_by_unit.get(
                    unit_id
                )
            )

            if relation is None:
                continue

            (
                inspiration,
                _,
            ) = representatives[
                unit_id
            ]

            candidate_rows.append(
                candidate_inspiration_component(
                    relation=relation,
                    candidate_unit_score=float(
                        inspiration
                        .candidate_unit_score
                    ),
                    exploration_score=float(
                        inspiration
                        .exploration_score
                    ),
                    quality_eligible=True,
                    source_path_id=str(
                        inspiration
                        .source_path_id
                    ),
                )
            )

        candidate_components = tuple(
            candidate_rows
        )

        if known_components:
            relation_component_topologies = (
                compose_relation_component_topologies(
                    components=(
                        *known_components,
                        *candidate_components,
                    ),
                    requested_source=(
                        requested_source
                    ),
                    requested_target=(
                        requested_target
                    ),
                    endpoint_equivalences=(
                        endpoint_equivalences
                    ),
                    mediator_equivalences=(
                        mediator_equivalences
                    ),
                    max_topologies=12,
                    require_confirmed_known=True,
                )
            )

            endpoint_fidelity_topologies = (
                compose_relation_component_topologies(
                    components=(
                        *known_components,
                        *candidate_components,
                    ),
                    requested_source=(
                        requested_source
                    ),
                    requested_target=(
                        requested_target
                    ),
                    endpoint_equivalences=(
                        endpoint_equivalences
                    ),
                    mediator_equivalences=(
                        mediator_equivalences
                    ),
                    max_topologies=12,
                    require_confirmed_known=True,
                    require_endpoint_fidelity=True,
                )
            )

            materializable_topologies = (
                compose_relation_component_topologies(
                    components=(
                        *known_components,
                        *candidate_components,
                    ),
                    requested_source=(
                        requested_source
                    ),
                    requested_target=(
                        requested_target
                    ),
                    endpoint_equivalences=(
                        endpoint_equivalences
                    ),
                    mediator_equivalences=(
                        mediator_equivalences
                    ),
                    max_topologies=12,
                    require_confirmed_known=True,
                    require_candidate_anchor=True,
                    require_endpoint_fidelity=True,
                )
            )

            relation_component_composites = tuple(
                topology_to_task_bridge_composite(
                    topology
                )
                for topology
                in materializable_topologies
            )

            (
                exact_endpoint_topology_count,
                equivalent_endpoint_topology_count,
            ) = _endpoint_fidelity_counts(
                endpoint_fidelity_topologies
            )
            used_endpoint_equivalence_witness_ids = (
                _used_endpoint_equivalence_witness_ids(
                    endpoint_fidelity_topologies
                )
            )

    all_relations = [
        relation_by_unit[unit_id]
        for unit_id in sorted(
            relation_by_unit
        )
    ]

    legacy_composites = (
        compose_task_bridge_candidates(
            candidates=all_relations,
            requested_source=(
                requested_source
            ),
            requested_target=(
                requested_target
            ),
            max_composites=12,
        )
    )

    if relation_component_composites:
        relation_component_ids = {
            row.composite_id
            for row
            in relation_component_composites
        }

        composites = (
            *relation_component_composites,
            *(
                row
                for row
                in legacy_composites
                if (
                    row.composite_id
                    not in relation_component_ids
                )
            ),
        )
    else:
        composites = (
            legacy_composites
        )

    # Frozen A17F7D choice:
    # earliest GLOBAL A10 rank per quality-eligible source.
    selected_composites = []

    seen_sources: set[str] = set()

    final_cap = min(
        max(
            int(args.max_axes),
            1,
        ),
        MAX_TOTAL_AXES,
    )

    task_cap = min(
        MAX_TASK_AXES,
        max(
            final_cap - 1,
            0,
        ),
    )

    for composite in composites:
        source_id = (
            composite.provenance_candidate_unit_id
            or composite.source_unit_id
        )

        if source_id in seen_sources:
            continue

        if (
            source_id
            not in representatives
        ):
            continue

        # Legacy composites still require lexical source overlap.
        # Relation-component topologies may use an explicit equivalence
        # witness whose authority is carried by the topology contract.
        if (
            not composite.source_overlap_tokens
            and composite.composition_mode
            != "relation_component_topology_v1"
        ):
            continue

        selected_composites.append(
            composite
        )
        seen_sources.add(
            source_id
        )

        if (
            len(selected_composites)
            >=
            task_cap
        ):
            break

    # --------------------------------------------------------------
    # If no task source survives, preserve generic behavior.
    # --------------------------------------------------------------
    if not selected_composites:
        _write_model(
            args.output_dual_context,
            old_dual,
        )
        _write_model(
            args.output_axis_plan,
            generic_plan,
        )

        args.output_report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output_report.write_text(
            json.dumps(
                {
                    "schema_version":
                        "task-conditioned-axis-plan-report-v1",
                    "status":
                        "NO_TASK_COMPOSITE",
                    "grammar":
                        "HOW_DOES_RELATE_TO_GRAMMAR_V1",
                    "endpoint_resolution_mode":
                        endpoint_resolution_mode,
                    "requested_source":
                        requested_source,
                    "requested_target":
                        requested_target,
                    "question_grammar_matched":
                        question_grammar_matched,
                    "generic_bundle_changed":
                        False,
                    "quality_eligible_source_count":
                        len(representatives),
                    "global_composite_count":
                        len(composites),
                    "confirmed_known_component_count":
                        len(known_components),
                    "candidate_inspiration_component_count":
                        len(candidate_components),
                    "relation_component_topology_count":
                        len(
                            relation_component_topologies
                        ),
                    "materializable_relation_component_topology_count":
                        len(
                            relation_component_composites
                        ),
                    "endpoint_fidelity_relation_component_topology_count":
                        len(
                            endpoint_fidelity_topologies
                        ),
                    "partial_endpoint_relation_component_topology_count":
                        sum(
                            not topology_has_materializable_endpoint_fidelity(
                                row
                            )
                            for row
                            in relation_component_topologies
                        ),
                    "known_known_endpoint_fidelity_topology_count":
                        sum(
                            (
                                row.source_component.authority
                                ==
                                RelationComponentAuthority
                                .CONFIRMED_KNOWN
                            )
                            and
                            (
                                row.target_component.authority
                                ==
                                RelationComponentAuthority
                                .CONFIRMED_KNOWN
                            )
                            for row
                            in endpoint_fidelity_topologies
                        ),
                    "known_known_topology_count":
                        sum(
                            (
                                row.source_component.authority
                                ==
                                RelationComponentAuthority
                                .CONFIRMED_KNOWN
                            )
                            and
                            (
                                row.target_component.authority
                                ==
                                RelationComponentAuthority
                                .CONFIRMED_KNOWN
                            )
                            for row
                            in relation_component_topologies
                        ),
                    "accepted_patterns_input":
                        (
                            None
                            if args.accepted_patterns is None
                            else str(args.accepted_patterns)
                        ),
                    "endpoint_equivalences_input":
                        (
                            None
                            if args.endpoint_equivalences is None
                            else str(args.endpoint_equivalences)
                        ),
                    "endpoint_equivalence_witness_count":
                        len(endpoint_equivalences),
                    "exact_endpoint_relation_component_topology_count":
                        exact_endpoint_topology_count,
                    "equivalent_endpoint_relation_component_topology_count":
                        equivalent_endpoint_topology_count,
                    "used_endpoint_equivalence_witness_ids":
                        used_endpoint_equivalence_witness_ids,
                    "task_axis_count":
                        0,
                    "generic_axis_count":
                        len(generic_plan.axes),
                    "architecture_tuning":
                        False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "A17F_STATUS=NO_TASK_COMPOSITE"
        )
        print(
            "TASK_AXIS_COUNT=0"
        )
        print(
            "GENERIC_BUNDLE_CHANGED=False"
        )
        return 0

    # --------------------------------------------------------------
    # Build task-lane inspirations.
    # --------------------------------------------------------------
    lane_inspirations = []

    for composite in (
        selected_composites
    ):
        source_id = (
            composite.provenance_candidate_unit_id
            or composite.source_unit_id
        )

        inspiration, _ = (
            representatives[
                source_id
            ]
        )

        lane = (
            _sanitize_lane_inspiration(
                inspiration=inspiration,
                inspiration_id=_lane_id(
                    question=args.question,
                    unit_id=source_id,
                    path_id=(
                        inspiration
                        .source_path_id
                    ),
                ),
            )
        )

        if (
            lane
            .eligible_as_positive_premise
            is not False
        ):
            raise RuntimeError(
                "Task-lane inspiration became "
                "positive-premise eligible"
            )

        lane_inspirations.append(
            lane
        )

    new_dual = (
        DualHypothesisContext.build(
            old_dual.grounded_context,
            old_dual.discovery_bundle,
            task_lane_inspirations=(
                lane_inspirations
            ),
        )
    )

    if (
        new_dual
        .discovery_bundle
        .bundle_sha256
        !=
        old_dual
        .discovery_bundle
        .bundle_sha256
    ):
        raise RuntimeError(
            "Generic DiscoveryBundle changed "
            "while adding task lane"
        )

    # --------------------------------------------------------------
    # Materialize task composite axes.
    # --------------------------------------------------------------
    final_axes = []

    for rank, (
        composite,
        lane,
    ) in enumerate(
        zip(
            selected_composites,
            lane_inspirations,
        ),
        start=1,
    ):
        planner_score = (
            planner._score(
                lane
            )
        )

        source_axis = (
            planner._materialize_axis(
                dual_context_sha256=(
                    new_dual
                    .dual_context_sha256
                ),
                item=lane,
                planner_score=(
                    planner_score
                ),
                axis_rank=rank,
            )
        )

        composite_axis = (
            materialize_task_bridge_composite_axis(
                composite=composite,
                source_axis=source_axis,
                requested_source=(
                    requested_source
                ),
                requested_target=(
                    requested_target
                ),
                axis_rank=rank,
            )
        )

        final_axes.append(
            composite_axis
        )

    # --------------------------------------------------------------
    # Frozen generic reserve:
    # preserve original generic-plan order.
    # Rematerialize against the NEW dual SHA.
    # --------------------------------------------------------------
    inspiration_by_id = {
        item.inspiration_id:
        item
        for item in (
            old_dual
            .discovery_bundle
            .inspirations
        )
    }

    for old_axis in generic_plan.axes:
        if len(final_axes) >= final_cap:
            break

        inspiration = (
            inspiration_by_id.get(
                old_axis.inspiration_id
            )
        )

        if inspiration is None:
            raise RuntimeError(
                "Generic axis inspiration missing "
                "from unchanged DiscoveryBundle: "
                + old_axis.inspiration_id
            )

        rank = (
            len(final_axes)
            + 1
        )

        generic_axis = (
            planner._materialize_axis(
                dual_context_sha256=(
                    new_dual
                    .dual_context_sha256
                ),
                item=inspiration,
                planner_score=(
                    planner._score(
                        inspiration
                    )
                ),
                axis_rank=rank,
            )
        )

        final_axes.append(
            generic_axis
        )

    if not final_axes:
        raise RuntimeError(
            "A17F produced zero final axes"
        )

    if (
        len(final_axes)
        > final_cap
    ):
        raise RuntimeError(
            "A17F exceeded frozen axis cap"
        )

    if (
        final_cap >= 2
        and
        not any(
            axis.source_mode
            not in TASK_AXIS_SOURCE_MODES
            for axis in final_axes
        )
    ):
        raise RuntimeError(
            "A17F violated generic-reserve invariant"
        )

    plan_id = _stable_id(
        "discovery_axis_plan",
        new_dual.dual_context_sha256,
        bundle.bundle_sha256,
        *[
            axis.axis_id
            for axis in final_axes
        ],
    )

    payload = {
        "schema_version":
            generic_plan.schema_version,
        "plan_id":
            plan_id,
        "source_dual_context_id":
            new_dual.dual_context_id,
        "source_dual_context_sha256":
            new_dual.dual_context_sha256,
        "source_bundle_id":
            bundle.bundle_id,
        "source_bundle_sha256":
            bundle.bundle_sha256,
        "corpus_id":
            bundle.corpus_id,
        "axes":
            [
                axis.model_dump(
                    mode="json"
                )
                for axis in final_axes
            ],
        "excluded_inspiration_ids":
            generic_plan
            .excluded_inspiration_ids,
        "policy":
            generic_plan
            .policy
            .model_dump(
                mode="json"
            ),
    }

    final_plan = DiscoveryAxisPlan(
        **payload,
        plan_sha256=_sha256_json(
            payload
        ),
    )

    _write_model(
        args.output_dual_context,
        new_dual,
    )

    _write_model(
        args.output_axis_plan,
        final_plan,
    )

    report = {
        "schema_version":
            "task-conditioned-axis-plan-report-v1",
        "status":
            "TASK_CONDITIONED",
        "grammar":
            "HOW_DOES_RELATE_TO_GRAMMAR_V1",
        "endpoint_resolution_mode":
            endpoint_resolution_mode,
        "requested_source":
            requested_source,
        "requested_target":
            requested_target,
        "question_grammar_matched":
            question_grammar_matched,
        "generic_bundle_changed":
            False,
        "generic_bundle_sha256":
            bundle.bundle_sha256,
        "replay_bundle_sha256":
            replay_bundle.bundle_sha256,
        "quality_eligible_source_count":
            len(representatives),
        "global_composite_count":
            len(composites),
        "confirmed_known_component_count":
            len(known_components),
        "candidate_inspiration_component_count":
            len(candidate_components),
        "relation_component_topology_count":
            len(
                relation_component_topologies
            ),
        "materializable_relation_component_topology_count":
            len(
                relation_component_composites
            ),
        "endpoint_fidelity_relation_component_topology_count":
            len(
                endpoint_fidelity_topologies
            ),
        "partial_endpoint_relation_component_topology_count":
            sum(
                not topology_has_materializable_endpoint_fidelity(
                    row
                )
                for row
                in relation_component_topologies
            ),
        "known_known_endpoint_fidelity_topology_count":
            sum(
                (
                    row.source_component.authority
                    ==
                    RelationComponentAuthority
                    .CONFIRMED_KNOWN
                )
                and
                (
                    row.target_component.authority
                    ==
                    RelationComponentAuthority
                    .CONFIRMED_KNOWN
                )
                for row
                in endpoint_fidelity_topologies
            ),
        "known_known_topology_count":
            sum(
                (
                    row.source_component.authority
                    ==
                    RelationComponentAuthority
                    .CONFIRMED_KNOWN
                )
                and
                (
                    row.target_component.authority
                    ==
                    RelationComponentAuthority
                    .CONFIRMED_KNOWN
                )
                for row
                in relation_component_topologies
            ),
        "selected_relation_component_topology_ids":
            [
                str(row.topology_id)
                for row
                in selected_composites
                if (
                    row.composition_mode
                    ==
                    "relation_component_topology_v1"
                )
            ],
        "accepted_patterns_input":
            (
                None
                if args.accepted_patterns is None
                else str(args.accepted_patterns)
            ),
        "endpoint_equivalences_input":
            (
                None
                if args.endpoint_equivalences is None
                else str(args.endpoint_equivalences)
            ),
        "endpoint_equivalence_witness_count":
            len(endpoint_equivalences),
        "exact_endpoint_relation_component_topology_count":
            exact_endpoint_topology_count,
        "equivalent_endpoint_relation_component_topology_count":
            equivalent_endpoint_topology_count,
        "used_endpoint_equivalence_witness_ids":
            used_endpoint_equivalence_witness_ids,
        "selected_source_unit_ids":
            [
                (
                    row.provenance_candidate_unit_id
                    or row.source_unit_id
                )
                for row in (
                    selected_composites
                )
            ],
        "selected_composite_ids":
            [
                row.composite_id
                for row in (
                    selected_composites
                )
            ],
        "task_axis_count":
            sum(
                axis.source_mode
                in TASK_AXIS_SOURCE_MODES
                for axis in final_axes
            ),
        "generic_axis_count":
            sum(
                axis.source_mode
                not in TASK_AXIS_SOURCE_MODES
                for axis in final_axes
            ),
        "axis_modes":
            [
                axis.source_mode
                for axis in final_axes
            ],
        "axis_ids":
            [
                axis.axis_id
                for axis in final_axes
            ],
        "dual_context_id":
            new_dual.dual_context_id,
        "dual_context_sha256":
            new_dual.dual_context_sha256,
        "axis_plan_id":
            final_plan.plan_id,
        "axis_plan_sha256":
            final_plan.plan_sha256,
        "architecture_tuning":
            False,
    }

    args.output_report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "A17F_STATUS=TASK_CONDITIONED"
    )
    print(
        "TASK_AXIS_COUNT=",
        report["task_axis_count"],
    )
    print(
        "GENERIC_AXIS_COUNT=",
        report["generic_axis_count"],
    )
    print(
        "AXIS_MODES=",
        report["axis_modes"],
    )
    print(
        "SELECTED_SOURCE_UNIT_IDS=",
        report[
            "selected_source_unit_ids"
        ],
    )
    print(
        "GENERIC_BUNDLE_CHANGED=False"
    )
    print(
        "DUAL_CONTEXT_SHA256=",
        new_dual.dual_context_sha256,
    )
    print(
        "AXIS_PLAN_SHA256=",
        final_plan.plan_sha256,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
