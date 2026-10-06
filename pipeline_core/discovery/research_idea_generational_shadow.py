from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from pipeline_core.discovery.research_idea_contracts import (
    ResearchIdeaNode,
    ResearchIdeaSearchState,
    SearchCostLedger,
)
from pipeline_core.discovery.research_idea_feedback import (
    candidate_to_idea_index,
    compile_feedback_coverage_audit,
    compile_idea_feedback_directives,
    compile_idea_outcome_observations,
)
from pipeline_core.discovery.research_idea_projection import (
    project_current_idea_artifacts,
)
from pipeline_core.discovery.research_idea_search_contracts import (
    ExactKernelDuplicateGroup,
    GenerationalIdeaSearchShadowReport,
    IdeaGenerationState,
    IdeaSearchCandidateState,
    IdeaSearchDirective,
)
from pipeline_core.discovery.research_idea_semantics import (
    assess_idea_transition,
)


_PROFILE_ORDER = (
    "TASK_NEAR_VALIDATION",
    "MECHANISM_FOCUSED",
    "DISCRIMINATING_TEST",
    "HIGH_INFORMATION",
    "EXPLORATORY_BRIDGE",
    "REFRAME",
)

_PROFILE_DIMENSIONS = {
    "TASK_NEAR_VALIDATION": (
        "task_relevance",
        "falsifiability",
        "operationalizability",
        "information_gain",
    ),
    "MECHANISM_FOCUSED": (
        "mechanistic_coherence",
        "falsifiability",
        "discriminating_power",
    ),
    "DISCRIMINATING_TEST": (
        "discriminating_power",
        "falsifiability",
        "information_gain",
        "operationalizability",
    ),
    "HIGH_INFORMATION": (
        "information_gain",
        "discriminating_power",
        "falsifiability",
    ),
    "EXPLORATORY_BRIDGE": (
        "task_relevance",
        "mechanistic_coherence",
        "falsifiability",
        "discriminating_power",
        "information_gain",
    ),
    "REFRAME": (
        "task_relevance",
        "discriminating_power",
        "information_gain",
        "falsifiability",
    ),
}

_LEVEL_RANK = {
    "LOW": 0,
    "INDETERMINATE": 1,
    "MODERATE": 2,
    "HIGH": 3,
}

_BAND_RANK = {
    "DEFER": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}

_BURDEN_RANK = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
}


def _canonical(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = _canonical(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:20]}"


def _exact_kernel_groups(
    nodes: Sequence[ResearchIdeaNode],
) -> list[ExactKernelDuplicateGroup]:
    grouped: dict[str, list[ResearchIdeaNode]] = defaultdict(list)
    for node in nodes:
        grouped[node.kernel_sha256].append(node)
    return [
        ExactKernelDuplicateGroup(
            kernel_sha256=kernel,
            idea_ids=[row.idea_id for row in rows],
            generation_indices=sorted({row.generation_index for row in rows}),
        )
        for kernel, rows in sorted(grouped.items())
        if len(rows) > 1
    ]


def _transition_assessments(
    nodes: Sequence[ResearchIdeaNode],
) -> list[Any]:
    by_id = {node.idea_id: node for node in nodes}
    rows = []
    for node in nodes:
        if node.generation_index == 0:
            continue
        parents = []
        for parent_id in node.parent_idea_ids:
            parent = by_id.get(parent_id)
            if parent is None:
                raise ValueError(
                    "ResearchIdea transition references unknown parent: "
                    + parent_id
                )
            if parent.generation_index >= node.generation_index:
                raise ValueError(
                    "ResearchIdea parent generation must precede child generation"
                )
            parents.append(parent)
        rows.append(
            assess_idea_transition(
                parents=parents,
                proposed=node,
                operator_id=node.operator_id,
            )
        )
    return rows


def _pareto_layers(
    rows: Sequence[tuple[str, tuple[int, ...]]],
) -> dict[str, int]:
    remaining = {idea_id: vector for idea_id, vector in rows}
    result: dict[str, int] = {}
    layer = 1
    while remaining:
        current = []
        for idea_id, vector in remaining.items():
            dominated = any(
                other_id != idea_id
                and all(a >= b for a, b in zip(other, vector))
                and any(a > b for a, b in zip(other, vector))
                for other_id, other in remaining.items()
            )
            if not dominated:
                current.append(idea_id)
        if not current:
            current = sorted(remaining)
        for idea_id in current:
            result[idea_id] = layer
            remaining.pop(idea_id, None)
        layer += 1
    return result


def _evaluation_levels(row: Any) -> dict[str, int]:
    dimensions = (
        "task_relevance",
        "mechanistic_coherence",
        "falsifiability",
        "discriminating_power",
        "operationalizability",
        "information_gain",
    )
    return {
        name: _LEVEL_RANK.get(str(getattr(getattr(row, name), "level", "")), 1)
        for name in dimensions
    }


def _underexplored(visits: int, population_size: int) -> float:
    return math.sqrt(
        max(0.0, math.log(2.0 + max(1, population_size)))
        / (1.0 + max(0, visits))
    )


def _copy_state(state: ResearchIdeaSearchState) -> ResearchIdeaSearchState:
    return state.model_copy(deep=True)


def _updated_search_states(
    *,
    nodes: Sequence[ResearchIdeaNode],
    pool: Any,
    evaluation: Any,
    observations: Sequence[Any],
    prior_search_states: Mapping[str, ResearchIdeaSearchState] | None,
) -> list[ResearchIdeaSearchState]:
    prior = dict(prior_search_states or {})
    candidate_to_idea = candidate_to_idea_index(pool=pool, nodes=nodes)
    eval_counts: Counter[str] = Counter()
    for row in evaluation.evaluations:
        idea_id = candidate_to_idea.get(str(row.candidate_id))
        if idea_id:
            eval_counts[idea_id] += 1

    materialization_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    for row in observations:
        materialization_counts[row.idea_id] += int(
            row.materialization_status is not None
        )
        verification_counts[row.idea_id] += int(
            row.residual_epistemic_state is not None
            or row.prospective_identifiability is not None
        )

    states = []
    for node in nodes:
        base = _copy_state(prior[node.idea_id]) if node.idea_id in prior else (
            ResearchIdeaSearchState(idea_id=node.idea_id)
        )
        verification_delta = verification_counts[node.idea_id]
        cost = base.compute_spent.model_copy(
            update={
                "verification_calls": (
                    base.compute_spent.verification_calls + verification_delta
                )
            }
        )
        states.append(
            base.model_copy(
                update={
                    "visit_count": base.visit_count + eval_counts[node.idea_id],
                    "materialization_attempt_count": (
                        base.materialization_attempt_count
                        + materialization_counts[node.idea_id]
                    ),
                    "verification_attempt_count": (
                        base.verification_attempt_count + verification_delta
                    ),
                    "compute_spent": cost,
                    "search_status": "ACTIVE",
                }
            )
        )
    return states


def _idea_candidate_states(
    *,
    nodes: Sequence[ResearchIdeaNode],
    pool: Any,
    evaluation: Any,
    selection: Any,
    directives: Sequence[IdeaSearchDirective],
    search_states: Sequence[ResearchIdeaSearchState],
) -> list[IdeaSearchCandidateState]:
    candidate_to_idea = candidate_to_idea_index(pool=pool, nodes=nodes)
    directive_by_id = {row.idea_id: row for row in directives}
    search_by_id = {row.idea_id: row for row in search_states}
    retained = set(map(str, selection.retained_candidate_ids))
    candidate_by_id = {
        str(row.candidate_id): row
        for row in pool.candidates
    }

    eval_rows_by_idea: dict[str, list[Any]] = defaultdict(list)
    for row in evaluation.evaluations:
        idea_id = candidate_to_idea.get(str(row.candidate_id))
        if idea_id:
            eval_rows_by_idea[idea_id].append(row)

    population_size = max(1, len(eval_rows_by_idea))
    states: list[IdeaSearchCandidateState] = []
    for idea_id, rows in sorted(eval_rows_by_idea.items()):
        # Current pool normally has one candidate per source object. Keep the
        # aggregation deterministic if future projections produce more than one.
        candidate_ids = sorted(str(row.candidate_id) for row in rows)
        profile_union = list(dict.fromkeys(
            profile
            for row in rows
            for profile in row.eligible_profiles
        ))
        levels = {
            name: max(_evaluation_levels(row)[name] for row in rows)
            for name in (
                "task_relevance",
                "mechanistic_coherence",
                "falsifiability",
                "discriminating_power",
                "operationalizability",
                "information_gain",
            )
        }
        burden = min(
            (str(row.verification_burden) for row in rows),
            key=lambda value: _BURDEN_RANK.get(value, 1),
        )
        legacy_families = sorted({
            str(candidate_by_id[candidate_id].conceptual_family_signature)
            for candidate_id in candidate_ids
        })
        directive = directive_by_id.get(idea_id)
        search = search_by_id[idea_id]
        states.append(
            IdeaSearchCandidateState(
                idea_id=idea_id,
                candidate_ids=candidate_ids,
                eligible_profiles=profile_union,
                dimension_levels=levels,
                verification_burden=burden,
                legacy_portfolio_retained=any(
                    candidate_id in retained
                    for candidate_id in candidate_ids
                ),
                legacy_family_signatures=legacy_families,
                priority_band=(
                    directive.priority_band if directive else "MEDIUM"
                ),
                underexplored_bonus=_underexplored(
                    search.visit_count,
                    population_size,
                ),
                visit_count=search.visit_count,
            )
        )

    for profile in _PROFILE_ORDER:
        eligible = [row for row in states if profile in row.eligible_profiles]
        vectors = [
            (
                row.idea_id,
                tuple(
                    row.dimension_levels[name]
                    for name in _PROFILE_DIMENSIONS[profile]
                ),
            )
            for row in eligible
        ]
        layers = _pareto_layers(vectors) if vectors else {}
        for row in eligible:
            row.pareto_layer_by_profile[profile] = layers[row.idea_id]
    return states


def _allocate_g2_parents(
    *,
    nodes: Sequence[ResearchIdeaNode],
    candidate_states: Sequence[IdeaSearchCandidateState],
    directives: Sequence[IdeaSearchDirective],
    max_g2_parents: int,
    max_g2_per_profile: int,
) -> tuple[list[IdeaSearchCandidateState], list[str], dict[str, int]]:
    if max_g2_parents < 1 or max_g2_per_profile < 1:
        raise ValueError("G2 parent limits must be >= 1")

    node_by_id = {node.idea_id: node for node in nodes}
    directive_by_id = {row.idea_id: row for row in directives}
    working = [row.model_copy(deep=True) for row in candidate_states]
    by_id = {row.idea_id: row for row in working}

    selected_ids: list[str] = []
    selected_kernels: set[str] = set()
    profile_counts: Counter[str] = Counter()

    def eligible(row: IdeaSearchCandidateState, profile: str) -> bool:
        if row.idea_id in selected_ids:
            return False
        directive = directive_by_id.get(row.idea_id)
        if directive is not None and not directive.idea_reproductive:
            return False
        if profile not in row.eligible_profiles:
            return False
        kernel = node_by_id[row.idea_id].kernel_sha256
        if kernel in selected_kernels:
            return False
        return True

    def sort_key(row: IdeaSearchCandidateState, profile: str) -> tuple[Any, ...]:
        layer = row.pareto_layer_by_profile.get(profile, 999)
        vector = tuple(
            -row.dimension_levels[name]
            for name in _PROFILE_DIMENSIONS[profile]
        )
        return (
            -_BAND_RANK[row.priority_band],
            layer,
            vector,
            -row.underexplored_bonus,
            _BURDEN_RANK.get(row.verification_burden, 1),
            row.idea_id,
        )

    progress = True
    while len(selected_ids) < max_g2_parents and progress:
        progress = False
        for profile in _PROFILE_ORDER:
            if profile_counts[profile] >= max_g2_per_profile:
                continue
            pool = [row for row in working if eligible(row, profile)]
            if not pool:
                continue
            pool.sort(key=lambda row: sort_key(row, profile))
            chosen = pool[0]
            chosen.selected_for_g2 = True
            chosen.assigned_profile = profile
            chosen.selection_reason_codes = [
                "PROFILE_ELIGIBLE",
                "PARETO_LAYER_WITHIN_PROFILE",
                "FEEDBACK_PRIORITY_BAND",
                "UNDEREXPLORED_TIE_BREAK",
                "EXACT_KERNEL_NOT_ALREADY_SELECTED",
            ]
            selected_ids.append(chosen.idea_id)
            selected_kernels.add(node_by_id[chosen.idea_id].kernel_sha256)
            profile_counts[profile] += 1
            progress = True
            if len(selected_ids) >= max_g2_parents:
                break

    # If profile quotas leave spare capacity, retain non-dominated candidates by
    # their best available profile without collapsing all dimensions to a scalar.
    while len(selected_ids) < max_g2_parents:
        candidates = []
        for row in working:
            if row.idea_id in selected_ids:
                continue
            directive = directive_by_id.get(row.idea_id)
            if directive is not None and not directive.idea_reproductive:
                continue
            kernel = node_by_id[row.idea_id].kernel_sha256
            if kernel in selected_kernels:
                continue
            layers = list(row.pareto_layer_by_profile.values())
            if not layers:
                continue
            candidates.append(row)
        if not candidates:
            break
        candidates.sort(
            key=lambda row: (
                -_BAND_RANK[row.priority_band],
                min(row.pareto_layer_by_profile.values()),
                -row.underexplored_bonus,
                _BURDEN_RANK.get(row.verification_burden, 1),
                row.idea_id,
            )
        )
        chosen = candidates[0]
        best_profile = min(
            chosen.pareto_layer_by_profile,
            key=lambda profile: (
                chosen.pareto_layer_by_profile[profile],
                _PROFILE_ORDER.index(profile),
            ),
        )
        chosen.selected_for_g2 = True
        chosen.assigned_profile = best_profile
        chosen.selection_reason_codes = [
            "REMAINING_CAPACITY",
            "BEST_AVAILABLE_PARETO_LAYER",
            "FEEDBACK_PRIORITY_BAND",
            "UNDEREXPLORED_TIE_BREAK",
            "EXACT_KERNEL_NOT_ALREADY_SELECTED",
        ]
        selected_ids.append(chosen.idea_id)
        selected_kernels.add(node_by_id[chosen.idea_id].kernel_sha256)
        profile_counts[best_profile] += 1

    return working, selected_ids, dict(sorted(profile_counts.items()))


def build_generational_idea_search_shadow(
    *,
    population: Any,
    evolution_report: Any,
    pool: Any,
    evaluation: Any,
    selection: Any,
    materialization: Any,
    residual_state: Mapping[str, Any] | None = None,
    prospective_by_hypothesis: Any = None,
    prior_search_states: Mapping[str, ResearchIdeaSearchState] | None = None,
    max_g2_parents: int = 8,
    max_g2_per_profile: int = 2,
) -> GenerationalIdeaSearchShadowReport:
    if evolution_report.source_population_id != population.population_id:
        raise ValueError("Idea Evolution/Frontier population lineage mismatch")
    if pool.source_population_id != population.population_id:
        raise ValueError("scientific portfolio pool/Frontier population mismatch")
    if pool.source_evolution_report_id != evolution_report.report_id:
        raise ValueError("scientific portfolio pool/Idea Evolution mismatch")
    if evaluation.source_pool_id != pool.pool_id:
        raise ValueError("scientific portfolio evaluation/pool mismatch")
    if selection.source_pool_id != pool.pool_id:
        raise ValueError("scientific portfolio selection/pool mismatch")
    if selection.source_evaluation_report_id != evaluation.report_id:
        raise ValueError("scientific portfolio selection/evaluation mismatch")
    if materialization.source_selection_id != selection.selection_id:
        raise ValueError("scientific portfolio materialization/selection mismatch")

    nodes = project_current_idea_artifacts(
        population=population,
        evolution_report=evolution_report,
    )
    g0 = [node for node in nodes if node.generation_index == 0]
    g1 = [node for node in nodes if node.generation_index == 1]
    transitions = _transition_assessments(nodes)
    duplicate_groups = _exact_kernel_groups(nodes)

    baseline_observations = compile_idea_outcome_observations(
        pool=pool,
        materialization=materialization,
        nodes=nodes,
        residual_state=None,
        prospective_by_hypothesis=None,
    )
    baseline_directives = compile_idea_feedback_directives(
        baseline_observations,
        all_idea_ids=[node.idea_id for node in nodes],
    )

    observations = compile_idea_outcome_observations(
        pool=pool,
        materialization=materialization,
        nodes=nodes,
        residual_state=residual_state,
        prospective_by_hypothesis=prospective_by_hypothesis,
    )
    feedback_coverage = compile_feedback_coverage_audit(observations)
    directives = compile_idea_feedback_directives(
        observations,
        all_idea_ids=[node.idea_id for node in nodes],
    )
    search_states = _updated_search_states(
        nodes=nodes,
        pool=pool,
        evaluation=evaluation,
        observations=observations,
        prior_search_states=prior_search_states,
    )

    baseline_candidate_states = _idea_candidate_states(
        nodes=nodes,
        pool=pool,
        evaluation=evaluation,
        selection=selection,
        directives=baseline_directives,
        search_states=search_states,
    )
    (
        _baseline_candidate_states,
        baseline_g2_parent_ids,
        baseline_g2_profile_counts,
    ) = _allocate_g2_parents(
        nodes=nodes,
        candidate_states=baseline_candidate_states,
        directives=baseline_directives,
        max_g2_parents=max_g2_parents,
        max_g2_per_profile=max_g2_per_profile,
    )

    candidate_states = _idea_candidate_states(
        nodes=nodes,
        pool=pool,
        evaluation=evaluation,
        selection=selection,
        directives=directives,
        search_states=search_states,
    )
    candidate_states, g2_parent_ids, g2_profile_counts = _allocate_g2_parents(
        nodes=nodes,
        candidate_states=candidate_states,
        directives=directives,
        max_g2_parents=max_g2_parents,
        max_g2_per_profile=max_g2_per_profile,
    )

    selected_set = set(g2_parent_ids)
    search_states = [
        row.model_copy(
            update={
                "search_status": "ELITE" if row.idea_id in selected_set else "ACTIVE"
            }
        )
        for row in search_states
    ]

    candidate_to_idea = candidate_to_idea_index(pool=pool, nodes=nodes)
    evaluated_idea_ids = list(dict.fromkeys(
        candidate_to_idea[str(row.candidate_id)]
        for row in evaluation.evaluations
        if str(row.candidate_id) in candidate_to_idea
    ))
    materialized_idea_ids = list(dict.fromkeys(
        row.idea_id
        for row in observations
        if row.materialization_status == "MATERIALIZED"
    ))
    observed_idea_ids = list(dict.fromkeys(row.idea_id for row in observations))

    generation_states = [
        IdeaGenerationState(
            generation_index=0,
            population_idea_ids=[row.idea_id for row in g0],
            evaluated_idea_ids=[
                idea_id for idea_id in evaluated_idea_ids
                if next(node for node in nodes if node.idea_id == idea_id).generation_index == 0
            ],
            materialized_idea_ids=[
                idea_id for idea_id in materialized_idea_ids
                if next(node for node in nodes if node.idea_id == idea_id).generation_index == 0
            ],
            observed_idea_ids=[
                idea_id for idea_id in observed_idea_ids
                if next(node for node in nodes if node.idea_id == idea_id).generation_index == 0
            ],
            selected_parent_idea_ids=[],
            max_parent_budget=0,
            selected_parent_count=0,
        ),
        IdeaGenerationState(
            generation_index=1,
            population_idea_ids=[row.idea_id for row in g1],
            evaluated_idea_ids=[
                idea_id for idea_id in evaluated_idea_ids
                if next(node for node in nodes if node.idea_id == idea_id).generation_index == 1
            ],
            materialized_idea_ids=[
                idea_id for idea_id in materialized_idea_ids
                if next(node for node in nodes if node.idea_id == idea_id).generation_index == 1
            ],
            observed_idea_ids=[
                idea_id for idea_id in observed_idea_ids
                if next(node for node in nodes if node.idea_id == idea_id).generation_index == 1
            ],
            selected_parent_idea_ids=[],
            max_parent_budget=0,
            selected_parent_count=0,
        ),
        IdeaGenerationState(
            generation_index=2,
            population_idea_ids=[],
            evaluated_idea_ids=[],
            materialized_idea_ids=[],
            observed_idea_ids=[],
            selected_parent_idea_ids=g2_parent_ids,
            max_parent_budget=max_g2_parents,
            selected_parent_count=len(g2_parent_ids),
        ),
    ]

    identity_counts = Counter(row.identity_relation for row in transitions)
    genealogy_counts = Counter(row.genealogy_relation for row in transitions)
    operator_counts = Counter(str(row.operator_id or "NONE") for row in transitions)
    materialization_counts = Counter(
        str(row.materialization_status)
        for row in observations
        if row.materialization_status is not None
    )
    residual_counts = Counter(
        str(row.residual_epistemic_state)
        for row in observations
        if row.residual_epistemic_state is not None
    )
    prospective_counts = Counter(
        str(row.prospective_identifiability)
        for row in observations
        if row.prospective_identifiability is not None
    )
    inconsistent_count = sum(
        row.operator_expectation_consistent is False
        for row in transitions
    )
    semantic_noop_count = sum(
        "EXPECTED_IDEA_MUTATION_BUT_SEMANTIC_NOOP" in row.diagnostic_codes
        or "MULTI_PARENT_COMPOSITION_SEMANTIC_NOOP" in row.diagnostic_codes
        for row in transitions
    )
    local_boundary_cross_count = sum(
        "LOCAL_REPAIR_ESCALATED_TO_IDEA_MUTATION" in row.diagnostic_codes
        or "LOCAL_REPAIR_CROSSED_IDEA_BOUNDARY" in row.diagnostic_codes
        for row in transitions
    )
    unique_kernels = len({node.kernel_sha256 for node in nodes})
    legacy_overlap = sum(
        row.selected_for_g2 and row.legacy_portfolio_retained
        for row in candidate_states
    )
    baseline_priority_counts = Counter(
        row.priority_band for row in baseline_directives
    )
    feedback_priority_counts = Counter(row.priority_band for row in directives)
    baseline_parent_set = set(baseline_g2_parent_ids)
    feedback_parent_set = set(g2_parent_ids)
    feedback_added = [
        idea_id for idea_id in g2_parent_ids if idea_id not in baseline_parent_set
    ]
    feedback_dropped = [
        idea_id
        for idea_id in baseline_g2_parent_ids
        if idea_id not in feedback_parent_set
    ]

    provisional = GenerationalIdeaSearchShadowReport(
        report_id="pending",
        report_sha256="pending",
        source_population_id=population.population_id,
        source_evolution_report_id=evolution_report.report_id,
        source_pool_id=pool.pool_id,
        source_evaluation_report_id=evaluation.report_id,
        source_selection_id=selection.selection_id,
        source_materialization_report_id=materialization.report_id,
        research_ideas=list(nodes),
        research_idea_count=len(nodes),
        generation0_idea_count=len(g0),
        generation1_idea_count=len(g1),
        exact_kernel_unique_count=unique_kernels,
        exact_kernel_duplicate_group_count=len(duplicate_groups),
        exact_kernel_duplicate_groups=duplicate_groups,
        transitions=transitions,
        transition_count=len(transitions),
        identity_relation_counts=dict(sorted(identity_counts.items())),
        genealogy_relation_counts=dict(sorted(genealogy_counts.items())),
        transition_counts_by_operator=dict(sorted(operator_counts.items())),
        operator_expectation_inconsistent_count=inconsistent_count,
        semantic_noop_count=semantic_noop_count,
        local_repair_boundary_cross_count=local_boundary_cross_count,
        observations=observations,
        observation_count=len(observations),
        observation_count_by_materialization_status=dict(
            sorted(materialization_counts.items())
        ),
        observation_count_by_residual_state=dict(sorted(residual_counts.items())),
        observation_count_by_prospective_status=dict(
            sorted(prospective_counts.items())
        ),
        feedback_coverage=feedback_coverage,
        directives=directives,
        search_states=search_states,
        candidate_states=candidate_states,
        generation_states=generation_states,
        g2_parent_idea_ids=g2_parent_ids,
        g2_parent_count=len(g2_parent_ids),
        g2_parent_profile_counts=g2_profile_counts,
        legacy_portfolio_overlap_count=legacy_overlap,
        baseline_g2_parent_idea_ids=baseline_g2_parent_ids,
        baseline_g2_parent_count=len(baseline_g2_parent_ids),
        baseline_g2_parent_profile_counts=baseline_g2_profile_counts,
        baseline_priority_band_counts=dict(sorted(baseline_priority_counts.items())),
        feedback_priority_band_counts=dict(sorted(feedback_priority_counts.items())),
        feedback_changed_parent_set=(
            baseline_parent_set != feedback_parent_set
        ),
        feedback_parent_added_idea_ids=feedback_added,
        feedback_parent_dropped_idea_ids=feedback_dropped,
        feedback_parent_overlap_count=len(
            baseline_parent_set & feedback_parent_set
        ),
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"generational_idea_search_shadow:{digest[:20]}",
            "report_sha256": digest,
        }
    )


__all__ = [
    "build_generational_idea_search_shadow",
]
