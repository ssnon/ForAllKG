from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_canonical(parts).encode('utf-8')).hexdigest()[:20]}"


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


FertilityDisposition = Literal[
    "HOLD_STABLE",
    "CONTINUE_SAME_IDEA",
    "EVOLVE_CHILD",
    "DIVERSIFY",
    "EVIDENCE_PROBE_ONLY",
    "DROP_INACTIVE",
]

EvolutionChannel = Literal["TRANSFORM", "EXPLORE", "WILDCARD"]

_REPRODUCTIVE = {"EVOLVE_CHILD", "DIVERSIFY"}
_NON_GROUNDED = {
    "IDEA_ONLY",
    "PARTIALLY_GROUNDED",
    "EVIDENCE_SEEKING",
    "SPECULATIVE_BUT_FALSIFIABLE",
}


class ResearchIdeaFertilityDecision(StrictModel):
    schema_version: Literal[
        "sis-v3-1-research-idea-fertility-decision-v1"
    ] = "sis-v3-1-research-idea-fertility-decision-v1"
    decision_id: str
    idea_id: str
    cycle_generation_index: int = Field(ge=2)
    idea_birth_generation_index: int = Field(ge=0)

    disposition: FertilityDisposition
    remains_active: bool
    fertile_for_child_generation: bool
    carry_forward_if_not_replaced: bool

    source_handoff_id: str | None = None
    source_member_ids: list[str] = Field(default_factory=list)
    source_epistemic_realization_ids: list[str] = Field(default_factory=list)
    maturity_counts: dict[str, int] = Field(default_factory=dict)
    active_non_grounded_member_count: int = Field(ge=0)
    active_grounded_member_count: int = Field(ge=0)
    speculative_member_count: int = Field(ge=0)

    realization_count: int = Field(ge=0)
    usable_grounded_realization_count: int = Field(ge=0)
    failed_or_abstained_count: int = Field(ge=0)
    not_operationalizable_count: int = Field(ge=0)
    local_search_budget: int = Field(ge=0)
    local_search_budget_used: int = Field(ge=0)
    local_search_exhausted: bool
    rescued_within_same_idea: bool

    epistemic_debt_ids: list[str] = Field(default_factory=list)
    epistemic_debt_kinds: list[str] = Field(default_factory=list)
    acquisition_escalation_eligible_debt_count: int = Field(ge=0)

    recommended_channels: list[EvolutionChannel] = Field(default_factory=list)
    nonbinding_operator_hints: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    active_status_is_not_fertility_authority: Literal[True] = True
    grounding_status_is_not_fertility_authority: Literal[True] = True
    lack_of_grounding_does_not_force_reproduction: Literal[True] = True
    usable_grounding_does_not_force_reproduction: Literal[True] = True
    one_realization_failure_does_not_force_reproduction: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_semantics(self) -> "ResearchIdeaFertilityDecision":
        expected = self.disposition in _REPRODUCTIVE
        if self.fertile_for_child_generation != expected:
            raise ValueError("fertility flag / disposition mismatch")
        if self.disposition == "DROP_INACTIVE" and self.remains_active:
            raise ValueError("DROP_INACTIVE cannot remain active")
        if self.fertile_for_child_generation and not self.remains_active:
            raise ValueError("inactive idea cannot be fertile")
        if self.fertile_for_child_generation and self.source_handoff_id is None:
            raise ValueError("fertile decision requires a source evolution handoff")
        return self


class AdaptiveFertilityReport(StrictModel):
    schema_version: Literal[
        "sis-v3-1-adaptive-research-idea-fertility-shadow-v1"
    ] = "sis-v3-1-adaptive-research-idea-fertility-shadow-v1"
    report_id: str
    report_sha256: str
    source_parallel_report_id: str
    source_lifecycle_report_id: str
    cycle_generation_index: int = Field(ge=2)

    decisions: list[ResearchIdeaFertilityDecision] = Field(default_factory=list)
    decision_count: int = Field(ge=0)
    active_idea_count: int = Field(ge=0)
    fertile_idea_count: int = Field(ge=0)
    persistent_nonfertile_idea_count: int = Field(ge=0)
    dropped_inactive_idea_count: int = Field(ge=0)
    source_evolution_handoff_count: int = Field(ge=0)
    fertile_handoff_count: int = Field(ge=0)
    disposition_counts: dict[str, int] = Field(default_factory=dict)

    active_idea_ids: list[str] = Field(default_factory=list)
    fertile_idea_ids: list[str] = Field(default_factory=list)
    persistent_nonfertile_idea_ids: list[str] = Field(default_factory=list)

    active_is_not_equal_to_fertile: Literal[True] = True
    persistent_idea_does_not_require_child_generation: Literal[True] = True
    local_search_precedes_idea_evolution: Literal[True] = True
    evidence_acquisition_is_not_fertility_gate: Literal[True] = True
    single_scalar_fitness_used: Literal[False] = False
    automatic_child_for_every_active_idea: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_authority: Literal[False] = False
    production_generation_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "AdaptiveFertilityReport":
        if self.decision_count != len(self.decisions):
            raise ValueError("decision_count mismatch")
        if self.active_idea_count != len(self.active_idea_ids):
            raise ValueError("active_idea_count mismatch")
        if self.fertile_idea_count != len(self.fertile_idea_ids):
            raise ValueError("fertile_idea_count mismatch")
        if self.persistent_nonfertile_idea_count != len(self.persistent_nonfertile_idea_ids):
            raise ValueError("persistent_nonfertile_idea_count mismatch")
        if set(self.fertile_idea_ids) - set(self.active_idea_ids):
            raise ValueError("fertile ideas must be active")
        if set(self.persistent_nonfertile_idea_ids) - set(self.active_idea_ids):
            raise ValueError("persistent nonfertile ideas must be active")
        if set(self.fertile_idea_ids) & set(self.persistent_nonfertile_idea_ids):
            raise ValueError("fertile and persistent-nonfertile sets must be disjoint")
        return self


@dataclass(frozen=True)
class FertilityGatedParallelView:
    report_id: str
    source_parallel_report_id: str
    members: list[Any]
    epistemic_debts: list[Any]
    evolution_handoffs: list[Any]


class PopulationPersistenceReport(StrictModel):
    schema_version: Literal[
        "sis-v3-1-population-persistence-report-v1"
    ] = "sis-v3-1-population-persistence-report-v1"
    report_id: str
    report_sha256: str
    cycle_generation_index: int = Field(ge=2)
    next_generation_index: int = Field(ge=3)
    source_fertility_report_id: str
    source_generation_execution_report_id: str
    source_next_raw_execution_report_id: str | None = None

    source_active_population_count: int = Field(ge=0)
    carried_forward_idea_ids: list[str] = Field(default_factory=list)
    replaced_parent_idea_ids: list[str] = Field(default_factory=list)
    generated_child_idea_ids: list[str] = Field(default_factory=list)
    final_population_idea_ids: list[str] = Field(default_factory=list)
    carried_forward_count: int = Field(ge=0)
    replaced_parent_count: int = Field(ge=0)
    generated_child_count: int = Field(ge=0)
    final_population_count: int = Field(ge=0)
    population_growth_budget: int = Field(ge=0)
    population_growth_count: int = Field(ge=0)

    active_nonfertile_ideas_persist_without_reproduction: Literal[True] = True
    selected_parent_without_retained_child_is_carried_forward: Literal[True] = True
    parent_replacement_requires_retained_child: Literal[True] = True
    same_research_idea_identity_preserved_on_carry_forward: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False

    @model_validator(mode="after")
    def _validate_counts(self) -> "PopulationPersistenceReport":
        if self.next_generation_index != self.cycle_generation_index + 1:
            raise ValueError("next_generation_index mismatch")
        if self.carried_forward_count != len(self.carried_forward_idea_ids):
            raise ValueError("carried_forward_count mismatch")
        if self.replaced_parent_count != len(self.replaced_parent_idea_ids):
            raise ValueError("replaced_parent_count mismatch")
        if self.generated_child_count != len(self.generated_child_idea_ids):
            raise ValueError("generated_child_count mismatch")
        if self.final_population_count != len(self.final_population_idea_ids):
            raise ValueError("final_population_count mismatch")
        if set(self.carried_forward_idea_ids) & set(self.replaced_parent_idea_ids):
            raise ValueError("carried-forward and replaced parent sets overlap")
        expected_growth = max(0, self.final_population_count - self.source_active_population_count)
        if self.population_growth_count != expected_growth:
            raise ValueError("population_growth_count mismatch")
        if self.population_growth_count > self.population_growth_budget:
            raise ValueError("population growth exceeds budget")
        return self


def _copy_handoff(handoff: Any, *, channels: Sequence[str], hints: Sequence[str], reasons: Sequence[str]) -> Any:
    updates = {
        "recommended_channels": _dedupe(channels),
        "nonbinding_operator_hints": _dedupe(hints),
        "reason_codes": _dedupe(reasons),
    }
    if hasattr(handoff, "model_copy"):
        return handoff.model_copy(update=updates)
    payload = dict(handoff) if isinstance(handoff, Mapping) else dict(vars(handoff))
    payload.update(updates)
    try:
        from pipeline_core.discovery.research_idea_parallel_partial_search import (
            ParallelIdeaEvolutionHandoff,
        )

        return ParallelIdeaEvolutionHandoff.model_validate(payload)
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(**payload)


def _maturity_summary(members: Sequence[Any]) -> tuple[dict[str, int], int, int, int]:
    counts = Counter(str(_get(row, "epistemic_maturity", "UNKNOWN")) for row in members)
    non_grounded = sum(counts[value] for value in _NON_GROUNDED)
    grounded = counts.get("STRICT_GROUNDED", 0) + counts.get("OPERATIONAL_GROUNDED", 0)
    speculative = counts.get("SPECULATIVE_BUT_FALSIFIABLE", 0)
    return dict(sorted(counts.items())), non_grounded, grounded, speculative


def build_adaptive_fertility_report(
    *,
    parallel_report: Any,
    lifecycle: Any,
    cycle_generation_index: int,
) -> AdaptiveFertilityReport:
    if cycle_generation_index < 2:
        raise ValueError("cycle_generation_index must be >= 2")

    members_by_idea: dict[str, list[Any]] = defaultdict(list)
    for member in (_get(parallel_report, "members", []) or []):
        members_by_idea[str(_get(member, "idea_id"))].append(member)

    debts_by_idea: dict[str, list[Any]] = defaultdict(list)
    for debt in (_get(parallel_report, "epistemic_debts", []) or []):
        debts_by_idea[str(_get(debt, "idea_id"))].append(debt)

    handoff_by_idea = {
        str(_get(row, "idea_id")): row
        for row in (_get(parallel_report, "evolution_handoffs", []) or [])
    }
    local_by_idea = {
        str(_get(row, "idea_id")): row
        for row in (_get(lifecycle, "local_states", []) or [])
    }

    decisions: list[ResearchIdeaFertilityDecision] = []
    for idea_state in (_get(parallel_report, "idea_states", []) or []):
        idea_id = str(_get(idea_state, "idea_id"))
        members = members_by_idea.get(idea_id, [])
        active_member_ids = set(_get(idea_state, "active_member_ids", []) or [])
        active_members = [
            row for row in members if str(_get(row, "member_id")) in active_member_ids
        ]
        maturity_counts, non_grounded_count, grounded_count, speculative_count = (
            _maturity_summary(active_members)
        )
        debts = debts_by_idea.get(idea_id, [])
        handoff = handoff_by_idea.get(idea_id)
        local = local_by_idea.get(idea_id)

        remains_active = bool(_get(idea_state, "remains_in_search_population", False))
        realization_count = int(_get(local, "realization_count", 0) or 0)
        usable = int(_get(local, "usable_grounded_realization_count", 0) or 0)
        failed = int(_get(local, "failed_or_abstained_count", 0) or 0)
        not_op = int(_get(local, "not_operationalizable_count", 0) or 0)
        budget = int(_get(local, "local_search_budget", 0) or 0)
        budget_used = int(_get(local, "local_search_budget_used", 0) or 0)
        exhausted = bool(_get(local, "local_search_exhausted", False))
        rescued = bool(_get(local, "rescued_within_same_idea", False))
        debt_kinds = _dedupe([str(_get(row, "requirement_kind", "")) for row in debts])
        acquisition_eligible = sum(
            str(_get(row, "disposition", "")) == "ACQUISITION_ESCALATION_ELIGIBLE"
            for row in debts
        )

        reasons: list[str] = []
        channels: list[EvolutionChannel] = []
        hints: list[str] = []

        if not remains_active:
            disposition: FertilityDisposition = "DROP_INACTIVE"
            reasons.append("IDEA_NOT_IN_ACTIVE_SEARCH_POPULATION")
        elif local is None:
            disposition = "CONTINUE_SAME_IDEA"
            reasons.extend(
                [
                    "LOCAL_SEARCH_STATE_MISSING_CONSERVATIVE_NO_MUTATION",
                    "ACTIVE_IDEA_PERSISTS_WITHOUT_AUTOMATIC_CHILD",
                ]
            )
        elif usable > 0:
            # A successful grounded realization is evidence that the current idea can
            # still be worked as itself. It is not a reason to mutate the idea.
            if acquisition_eligible > 0:
                disposition = "EVIDENCE_PROBE_ONLY"
                reasons.extend(
                    [
                        "USABLE_GROUNDED_REALIZATION_EXISTS",
                        "PERSISTENT_EPISTEMIC_DEBT_MAY_BE_PROBED_WITHOUT_CHILD_GENERATION",
                    ]
                )
            else:
                disposition = "HOLD_STABLE"
                reasons.extend(
                    [
                        "USABLE_GROUNDED_REALIZATION_EXISTS",
                        "NO_CHILD_EVOLUTION_PRESSURE_OBSERVED",
                    ]
                )
            if rescued:
                reasons.append("SAME_IDEA_LOCAL_RESCUE_SUCCEEDED")
            if not_op:
                reasons.append("MIXED_REALIZATION_OUTCOMES_DO_NOT_OVERRIDE_USABLE_SAME_IDEA_PATH")
        elif not exhausted:
            disposition = "CONTINUE_SAME_IDEA"
            reasons.extend(
                [
                    "NO_USABLE_GROUNDED_REALIZATION_YET",
                    "LOCAL_REALIZATION_SPACE_NOT_EXHAUSTED",
                    "SAME_IDEA_SEARCH_PRECEDES_IDEA_MUTATION",
                ]
            )
            if acquisition_eligible:
                reasons.append("EVIDENCE_ESCALATION_REMAINS_OPTIONAL_WHILE_LOCAL_SEARCH_IS_OPEN")
        elif handoff is None:
            disposition = "CONTINUE_SAME_IDEA"
            reasons.extend(
                [
                    "LOCAL_REALIZATION_SPACE_EXHAUSTED",
                    "NO_VALID_IDEA_EVOLUTION_HANDOFF_AVAILABLE",
                    "CONSERVATIVE_PERSISTENCE_WITHOUT_AUTOMATIC_MUTATION",
                ]
            )
        elif not_op > 0 or speculative_count > 0 or len(debt_kinds) >= 2:
            disposition = "DIVERSIFY"
            channels = ["EXPLORE", "WILDCARD"]
            hints = [
                "LATENT_VARIABLE",
                "REGIME_BOUNDARY",
                "PROXY_CHALLENGE",
                "CROSS_SOURCE_BRIDGE",
            ]
            reasons.extend(
                [
                    "LOCAL_REALIZATION_SPACE_EXHAUSTED",
                    "UNRESOLVED_STATE_SUPPORTS_DIVERSIFICATION",
                ]
            )
            if not_op:
                reasons.append("REPEATED_OR_PRESENT_NOT_OPERATIONALIZABLE_SIGNAL")
            if speculative_count:
                reasons.append("SPECULATIVE_FALSIFIABLE_MEMBER_REQUIRES_BROADER_SEARCH")
            if len(debt_kinds) >= 2:
                reasons.append("MULTI_AXIS_EPISTEMIC_DEBT_SUPPORTS_DIVERSIFICATION")
        else:
            disposition = "EVOLVE_CHILD"
            channels = ["TRANSFORM", "EXPLORE"]
            hints = list(_get(handoff, "nonbinding_operator_hints", []) or [])
            reasons.extend(
                [
                    "LOCAL_REALIZATION_SPACE_EXHAUSTED",
                    "UNRESOLVED_IDEA_REQUIRES_CONCEPTUAL_EVOLUTION",
                ]
            )

        fertile = disposition in _REPRODUCTIVE
        if fertile and handoff is not None:
            original_channels = [
                str(value)
                for value in (_get(handoff, "recommended_channels", []) or [])
                if str(value) in {"TRANSFORM", "EXPLORE", "WILDCARD"}
            ]
            if disposition == "EVOLVE_CHILD":
                channels = _dedupe([*channels, *original_channels])
                hints = _dedupe(
                    [*hints, *list(_get(handoff, "nonbinding_operator_hints", []) or [])]
                )
            elif disposition == "DIVERSIFY":
                channels = _dedupe([*channels, *[c for c in original_channels if c != "TRANSFORM"]])
            reasons.extend(list(_get(handoff, "reason_codes", []) or []))

        decision = ResearchIdeaFertilityDecision(
            decision_id=_stable_id(
                "research_idea_fertility_decision",
                cycle_generation_index,
                idea_id,
                disposition,
                [str(_get(row, "member_id")) for row in active_members],
                [str(_get(row, "debt_id")) for row in debts],
            ),
            idea_id=idea_id,
            cycle_generation_index=cycle_generation_index,
            idea_birth_generation_index=int(_get(idea_state, "generation_index", 0) or 0),
            disposition=disposition,
            remains_active=remains_active,
            fertile_for_child_generation=fertile,
            carry_forward_if_not_replaced=remains_active,
            source_handoff_id=(str(_get(handoff, "handoff_id")) if handoff is not None else None),
            source_member_ids=[str(_get(row, "member_id")) for row in active_members],
            source_epistemic_realization_ids=[
                str(_get(row, "epistemic_realization_id")) for row in active_members
            ],
            maturity_counts=maturity_counts,
            active_non_grounded_member_count=non_grounded_count,
            active_grounded_member_count=grounded_count,
            speculative_member_count=speculative_count,
            realization_count=realization_count,
            usable_grounded_realization_count=usable,
            failed_or_abstained_count=failed,
            not_operationalizable_count=not_op,
            local_search_budget=budget,
            local_search_budget_used=budget_used,
            local_search_exhausted=exhausted,
            rescued_within_same_idea=rescued,
            epistemic_debt_ids=[str(_get(row, "debt_id")) for row in debts],
            epistemic_debt_kinds=debt_kinds,
            acquisition_escalation_eligible_debt_count=acquisition_eligible,
            recommended_channels=channels,
            nonbinding_operator_hints=_dedupe(hints),
            reason_codes=_dedupe(reasons),
        )
        decisions.append(decision)

    counts = Counter(row.disposition for row in decisions)
    active_ids = [row.idea_id for row in decisions if row.remains_active]
    fertile_ids = [row.idea_id for row in decisions if row.fertile_for_child_generation]
    persistent_nonfertile = [
        row.idea_id
        for row in decisions
        if row.remains_active and not row.fertile_for_child_generation
    ]
    provisional = AdaptiveFertilityReport(
        report_id="pending",
        report_sha256="pending",
        source_parallel_report_id=str(_get(parallel_report, "report_id")),
        source_lifecycle_report_id=str(_get(lifecycle, "report_id")),
        cycle_generation_index=cycle_generation_index,
        decisions=decisions,
        decision_count=len(decisions),
        active_idea_count=len(active_ids),
        fertile_idea_count=len(fertile_ids),
        persistent_nonfertile_idea_count=len(persistent_nonfertile),
        dropped_inactive_idea_count=counts.get("DROP_INACTIVE", 0),
        source_evolution_handoff_count=len(_get(parallel_report, "evolution_handoffs", []) or []),
        fertile_handoff_count=len(fertile_ids),
        disposition_counts=dict(sorted(counts.items())),
        active_idea_ids=active_ids,
        fertile_idea_ids=fertile_ids,
        persistent_nonfertile_idea_ids=persistent_nonfertile,
    )
    payload = provisional.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    return provisional.model_copy(
        update={
            "report_id": f"adaptive_fertility:g{cycle_generation_index}:{digest[:20]}",
            "report_sha256": digest,
        }
    )


def build_fertility_gated_parallel_view(
    *,
    parallel_report: Any,
    fertility_report: AdaptiveFertilityReport,
) -> FertilityGatedParallelView:
    if fertility_report.source_parallel_report_id != str(_get(parallel_report, "report_id")):
        raise ValueError("fertility report / parallel report lineage mismatch")
    handoff_by_id = {
        str(_get(row, "handoff_id")): row
        for row in (_get(parallel_report, "evolution_handoffs", []) or [])
    }
    gated: list[Any] = []
    for decision in fertility_report.decisions:
        if not decision.fertile_for_child_generation:
            continue
        if decision.source_handoff_id is None:
            continue
        handoff = handoff_by_id.get(decision.source_handoff_id)
        if handoff is None:
            raise ValueError(f"fertile handoff missing: {decision.source_handoff_id}")
        gated.append(
            _copy_handoff(
                handoff,
                channels=decision.recommended_channels,
                hints=decision.nonbinding_operator_hints,
                reasons=[
                    *list(_get(handoff, "reason_codes", []) or []),
                    f"FERTILITY_DISPOSITION_{decision.disposition}",
                    "ACTIVE_STATUS_DOES_NOT_IMPLY_REPRODUCTION",
                    *decision.reason_codes,
                ],
            )
        )
    report_id = _stable_id(
        "fertility_gated_parallel_view",
        fertility_report.report_id,
        [str(_get(row, "handoff_id")) for row in gated],
    )
    return FertilityGatedParallelView(
        report_id=report_id,
        source_parallel_report_id=str(_get(parallel_report, "report_id")),
        members=list(_get(parallel_report, "members", []) or []),
        epistemic_debts=list(_get(parallel_report, "epistemic_debts", []) or []),
        evolution_handoffs=gated,
    )


def compose_persistent_population_execution(
    *,
    current_execution: Any,
    fertility_report: AdaptiveFertilityReport,
    next_generation_index: int,
    next_plan: Any,
    raw_next_execution: Any | None,
    population_growth_budget: int = 0,
) -> tuple[Any, PopulationPersistenceReport]:
    if population_growth_budget < 0:
        raise ValueError("population_growth_budget must be >= 0")
    if next_generation_index != fertility_report.cycle_generation_index + 1:
        raise ValueError("next generation / fertility cycle mismatch")

    from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
        EpistemicG4ExecutionReport,
    )
    from pipeline_core.discovery.research_idea_closed_generation_cycle import population_nodes

    current_nodes = list(population_nodes(current_execution))
    current_by_id = {str(_get(row, "idea_id")): row for row in current_nodes}
    active_ids = set(fertility_report.active_idea_ids)
    active_nodes = {
        idea_id: row for idea_id, row in current_by_id.items() if idea_id in active_ids
    }

    raw_population = list(_get(raw_next_execution, "g4_population_nodes", []) or [])
    semantic_by_child = {
        str(_get(row, "idea_id")): row
        for row in (_get(raw_next_execution, "semantic_records", []) or [])
    }
    children_by_parent: dict[str, list[Any]] = defaultdict(list)
    for child in raw_population:
        child_id = str(_get(child, "idea_id"))
        semantic = semantic_by_child.get(child_id)
        if semantic is None:
            continue
        parent_id = str(_get(semantic, "primary_parent_idea_id", ""))
        if parent_id:
            children_by_parent[parent_id].append(child)
    for rows in children_by_parent.values():
        rows.sort(key=lambda row: str(_get(row, "idea_id")))

    final_by_id = dict(active_nodes)
    replaced: list[str] = []
    generated_added: list[str] = []
    source_active_count = len(active_nodes)
    max_population = source_active_count + population_growth_budget

    selected_parent_ids = list(_get(next_plan, "selected_parent_idea_ids", []) or [])
    for parent_id in selected_parent_ids:
        if parent_id not in active_nodes:
            continue
        children = children_by_parent.get(parent_id, [])
        if not children:
            continue
        # The first retained child replaces its fertile parent. Extra siblings are
        # admitted only when an explicit population-growth budget exists.
        first = children[0]
        final_by_id.pop(parent_id, None)
        first_id = str(_get(first, "idea_id"))
        final_by_id[first_id] = first
        replaced.append(parent_id)
        generated_added.append(first_id)
        for child in children[1:]:
            if len(final_by_id) >= max_population:
                break
            child_id = str(_get(child, "idea_id"))
            final_by_id[child_id] = child
            generated_added.append(child_id)

    carried = sorted(idea_id for idea_id in active_nodes if idea_id in final_by_id)
    final_nodes = [final_by_id[key] for key in sorted(final_by_id)]
    final_ids = [str(_get(row, "idea_id")) for row in final_nodes]

    if raw_next_execution is None:
        base = EpistemicG4ExecutionReport(
            report_id="pending",
            report_sha256="pending",
            source_parallel_report_id=str(_get(next_plan, "source_parallel_report_id")),
            source_plan_id=str(_get(next_plan, "plan_id")),
            generation_index=next_generation_index,
            tasks=list(_get(next_plan, "tasks", []) or []),
            run_records=[],
            offspring_nodes=[],
            semantic_records=[],
            g4_population_nodes=final_nodes,
            raw_offspring_count=0,
            g4_population_count=len(final_nodes),
            genuine_child_count=0,
            indeterminate_probe_count=0,
            same_idea_refinement_count=0,
            exact_kernel_duplicate_suppressed_count=0,
            identity_relation_counts={},
            disposition_counts={},
            generated_count_by_channel={},
            generated_count_by_operator={},
            genuine_child_from_non_grounded_feedback_count=0,
            genuine_child_from_speculative_feedback_count=0,
            genuine_child_from_evidence_seeking_feedback_count=0,
            llm_call_count=0,
            input_tokens=0,
            output_tokens=0,
            semantic_retry_count=0,
            offspring_generation_executed=False,
            carried_forward_idea_ids=carried,
            carried_forward_count=len(carried),
            replaced_parent_idea_ids=replaced,
            replaced_parent_count=len(replaced),
            population_composed_with_persistence=True,
            population_growth_budget=population_growth_budget,
        )
    else:
        base = raw_next_execution.model_copy(
            update={
                "g4_population_nodes": final_nodes,
                "g4_population_count": len(final_nodes),
                "carried_forward_idea_ids": carried,
                "carried_forward_count": len(carried),
                "replaced_parent_idea_ids": sorted(replaced),
                "replaced_parent_count": len(replaced),
                "population_composed_with_persistence": True,
                "population_growth_budget": population_growth_budget,
            }
        )

    payload = base.model_dump(mode="json")
    payload.pop("report_id", None)
    payload.pop("report_sha256", None)
    digest = _sha(payload)
    composed = base.model_copy(
        update={
            "report_id": f"g{next_generation_index}_adaptive_population:{digest[:20]}",
            "report_sha256": digest,
        }
    )

    persistence = PopulationPersistenceReport(
        report_id="pending",
        report_sha256="pending",
        cycle_generation_index=fertility_report.cycle_generation_index,
        next_generation_index=next_generation_index,
        source_fertility_report_id=fertility_report.report_id,
        source_generation_execution_report_id=str(_get(current_execution, "report_id")),
        source_next_raw_execution_report_id=(
            str(_get(raw_next_execution, "report_id")) if raw_next_execution is not None else None
        ),
        source_active_population_count=source_active_count,
        carried_forward_idea_ids=carried,
        replaced_parent_idea_ids=sorted(replaced),
        generated_child_idea_ids=sorted(generated_added),
        final_population_idea_ids=final_ids,
        carried_forward_count=len(carried),
        replaced_parent_count=len(replaced),
        generated_child_count=len(generated_added),
        final_population_count=len(final_nodes),
        population_growth_budget=population_growth_budget,
        population_growth_count=max(0, len(final_nodes) - source_active_count),
    )
    persistence_payload = persistence.model_dump(mode="json")
    persistence_payload.pop("report_id", None)
    persistence_payload.pop("report_sha256", None)
    persistence_digest = _sha(persistence_payload)
    persistence = persistence.model_copy(
        update={
            "report_id": f"population_persistence:g{fertility_report.cycle_generation_index}:{persistence_digest[:20]}",
            "report_sha256": persistence_digest,
        }
    )
    return composed, persistence


__all__ = [
    "AdaptiveFertilityReport",
    "FertilityDisposition",
    "FertilityGatedParallelView",
    "PopulationPersistenceReport",
    "ResearchIdeaFertilityDecision",
    "build_adaptive_fertility_report",
    "build_fertility_gated_parallel_view",
    "compose_persistent_population_execution",
]
