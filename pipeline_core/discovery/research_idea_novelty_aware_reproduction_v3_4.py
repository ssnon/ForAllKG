from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline_core.discovery.external_novelty_contracts import (
    ExternalNoveltyReport,
    LiteratureQueryPlan,
)
from pipeline_core.discovery.novelty_residue import extract_novelty_residue
from pipeline_core.discovery.research_idea_adaptive_fertility import (
    AdaptiveFertilityReport,
    ResearchIdeaFertilityDecision,
)
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4Prompt,
)
from pipeline_core.discovery.research_idea_search_pressure_v3_3_2 import (
    BoundedResearchIdeaParentSchedule,
    ResearchIdeaNoveltyDepthSignal,
    ResearchIdeaSearchPressureReport,
)


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
    return list(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    )


class PriorArtSearchClaimContext(StrictModel):
    claim_id: str
    claim_text: str
    importance: str
    disposition: Literal[
        "SATURATED",
        "UNRESOLVED_PARTIAL",
        "RESIDUAL",
        "UNRESOLVED",
    ]
    prior_art_status: str
    required_bridge: str = ""
    predicted_observation: str = ""
    falsification_condition: str = ""

    search_feedback_only: Literal[True] = True
    positive_premise_authority: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_certification_authority: Literal[False] = False


class ResearchIdeaMutationSearchContext(StrictModel):
    schema_version: Literal[
        "sis-v3-4-research-idea-mutation-search-context-v1"
    ] = "sis-v3-4-research-idea-mutation-search-context-v1"

    idea_id: str
    novelty_shape: str
    external_status_counts: dict[str, int] = Field(default_factory=dict)
    search_coverage_sufficient: bool = False

    saturated_claims: list[PriorArtSearchClaimContext] = Field(default_factory=list)
    residual_claims: list[PriorArtSearchClaimContext] = Field(default_factory=list)
    partial_claims: list[PriorArtSearchClaimContext] = Field(default_factory=list)
    unresolved_claims: list[PriorArtSearchClaimContext] = Field(default_factory=list)

    preferred_action: str
    search_priority: str
    terminal_worthy: bool
    epistemic_debt_kinds: list[str] = Field(default_factory=list)
    scheduler_reason_codes: list[str] = Field(default_factory=list)

    contemporaneous_prior_art_search: Literal[True] = True
    prior_art_is_search_feedback_only: Literal[True] = True
    external_prior_art_promoted_to_positive_premise: Literal[False] = False
    absence_based_novelty_is_not_truth_authority: Literal[True] = True
    scientific_truth_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


class NoveltyAwareReproductionControlReport(StrictModel):
    schema_version: Literal[
        "sis-v3-4-novelty-aware-bounded-reproduction-control-v1"
    ] = "sis-v3-4-novelty-aware-bounded-reproduction-control-v1"

    report_id: str
    report_sha256: str
    generation_index: int
    source_pressure_report_id: str
    source_schedule_id: str
    source_base_fertility_report_id: str
    compatibility_fertility_report_id: str

    selected_parent_idea_ids: list[str] = Field(default_factory=list)
    deferred_search_worthy_idea_ids: list[str] = Field(default_factory=list)
    probe_only_idea_ids: list[str] = Field(default_factory=list)
    stable_retained_idea_ids: list[str] = Field(default_factory=list)

    selected_parent_count: int = 0
    deferred_search_worthy_count: int = 0
    probe_only_count: int = 0
    stable_retained_count: int = 0

    reproduction_selection_is_not_survival_selection: Literal[True] = True
    nonselected_active_parent_is_retained: Literal[True] = True
    prior_art_is_search_feedback_only: Literal[True] = True
    generated_children_require_later_grounding: Literal[True] = True
    canonical_graph_mutated: Literal[False] = False
    scientific_truth_authority: Literal[False] = False
    novelty_certification_authority: Literal[False] = False
    production_selection_authority: Literal[False] = False


def build_mutation_search_contexts(
    *,
    parallel_report: Any,
    query_plan: LiteratureQueryPlan,
    external_novelty_report: ExternalNoveltyReport,
    novelty_by_idea: Mapping[str, ResearchIdeaNoveltyDepthSignal],
    pressure_report: ResearchIdeaSearchPressureReport,
    schedule: BoundedResearchIdeaParentSchedule,
    max_claims_per_bucket: int = 4,
) -> dict[str, ResearchIdeaMutationSearchContext]:
    if max_claims_per_bucket < 1:
        raise ValueError("max_claims_per_bucket must be >= 1")

    hypothesis_ids_by_idea: dict[str, list[str]] = defaultdict(list)
    debt_kinds_by_idea: dict[str, list[str]] = defaultdict(list)
    for member in (_get(parallel_report, "members", []) or []):
        idea_id = str(_get(member, "idea_id", "") or "").strip()
        hypothesis_id = str(_get(member, "hypothesis_id", "") or "").strip()
        if idea_id and hypothesis_id:
            hypothesis_ids_by_idea[idea_id].append(hypothesis_id)
    for debt in (_get(parallel_report, "epistemic_debts", []) or []):
        idea_id = str(_get(debt, "idea_id", "") or "").strip()
        kind = str(_get(debt, "requirement_kind", "") or "").strip()
        if idea_id and kind:
            debt_kinds_by_idea[idea_id].append(kind)

    residue_by_hypothesis = {
        row.hypothesis_id: row
        for row in extract_novelty_residue(query_plan, external_novelty_report)
    }
    pressure_by_idea = {row.idea_id: row for row in pressure_report.pressures}
    selected_by_idea = {row.idea_id: row for row in schedule.selected_parents}

    result: dict[str, ResearchIdeaMutationSearchContext] = {}
    for idea_id, pressure in sorted(pressure_by_idea.items()):
        buckets: dict[str, list[PriorArtSearchClaimContext]] = {
            "SATURATED": [],
            "UNRESOLVED_PARTIAL": [],
            "RESIDUAL": [],
            "UNRESOLVED": [],
        }
        for hypothesis_id in _dedupe(hypothesis_ids_by_idea.get(idea_id, [])):
            residue = residue_by_hypothesis.get(hypothesis_id)
            if residue is None:
                continue
            for claim in residue.claims:
                row = PriorArtSearchClaimContext(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    importance=claim.importance,
                    disposition=claim.disposition,
                    prior_art_status=claim.prior_art_status,
                    required_bridge=claim.required_bridge,
                    predicted_observation=claim.predicted_observation,
                    falsification_condition=claim.falsification_condition,
                )
                buckets[claim.disposition].append(row)

        signal = novelty_by_idea.get(idea_id)
        scheduled = selected_by_idea.get(idea_id)
        scheduler_reasons = (
            list(scheduled.selection_reason_codes)
            if scheduled is not None
            else ["NOT_SELECTED_BY_BOUNDED_PARENT_SCHEDULER"]
        )
        result[idea_id] = ResearchIdeaMutationSearchContext(
            idea_id=idea_id,
            novelty_shape=(signal.novelty_shape if signal is not None else "NOT_ASSESSED"),
            external_status_counts=(
                dict(signal.external_status_counts) if signal is not None else {}
            ),
            search_coverage_sufficient=(
                bool(signal.search_coverage_sufficient) if signal is not None else False
            ),
            saturated_claims=buckets["SATURATED"][:max_claims_per_bucket],
            residual_claims=buckets["RESIDUAL"][:max_claims_per_bucket],
            partial_claims=buckets["UNRESOLVED_PARTIAL"][:max_claims_per_bucket],
            unresolved_claims=buckets["UNRESOLVED"][:max_claims_per_bucket],
            preferred_action=pressure.preferred_action,
            search_priority=pressure.search_priority,
            terminal_worthy=pressure.terminal_worthy,
            epistemic_debt_kinds=_dedupe(debt_kinds_by_idea.get(idea_id, [])),
            scheduler_reason_codes=_dedupe(scheduler_reasons),
        )
    return result


def adapt_fertility_report_v3_4(
    *,
    base_fertility: AdaptiveFertilityReport,
    pressure_report: ResearchIdeaSearchPressureReport,
    schedule: BoundedResearchIdeaParentSchedule,
) -> tuple[AdaptiveFertilityReport, NoveltyAwareReproductionControlReport]:
    if pressure_report.generation_index != base_fertility.cycle_generation_index:
        raise ValueError("pressure/fertility generation mismatch")
    if schedule.generation_index != base_fertility.cycle_generation_index:
        raise ValueError("schedule/fertility generation mismatch")
    if schedule.source_pressure_report_id != pressure_report.report_id:
        raise ValueError("schedule/pressure lineage mismatch")

    pressure_by_idea = {row.idea_id: row for row in pressure_report.pressures}
    selected_by_idea = {row.idea_id: row for row in schedule.selected_parents}

    decisions: list[ResearchIdeaFertilityDecision] = []
    deferred: list[str] = []
    probes: list[str] = []
    stable: list[str] = []

    for decision in base_fertility.decisions:
        pressure = pressure_by_idea.get(decision.idea_id)
        if pressure is None:
            decisions.append(decision)
            continue
        if not decision.remains_active:
            decisions.append(decision)
            continue

        scheduled = selected_by_idea.get(decision.idea_id)
        if scheduled is not None:
            if decision.source_handoff_id is None:
                raise ValueError(
                    f"v3.4 selected parent has no evolution handoff: {decision.idea_id}"
                )
            disposition = (
                "DIVERSIFY"
                if scheduled.preferred_action == "DIVERSIFY"
                else "EVOLVE_CHILD"
            )
            channels = list(scheduled.recommended_channels)
            hints = list(scheduled.nonbinding_operator_hints)
            reasons = _dedupe(
                [
                    *decision.reason_codes,
                    "V3_4_SELECTED_BY_NOVELTY_AWARE_BOUNDED_SCHEDULER",
                    f"V3_4_SEARCH_PRIORITY_{scheduled.search_priority}",
                    f"V3_4_PREFERRED_ACTION_{scheduled.preferred_action}",
                    "REPRODUCTION_SELECTION_IS_NOT_SURVIVAL_SELECTION",
                ]
            )
            decisions.append(
                decision.model_copy(
                    update={
                        "disposition": disposition,
                        "fertile_for_child_generation": True,
                        "carry_forward_if_not_replaced": True,
                        "recommended_channels": channels,
                        "nonbinding_operator_hints": hints,
                        "reason_codes": reasons,
                    }
                )
            )
            continue

        if pressure.search_worthy:
            deferred.append(decision.idea_id)
            reasons = _dedupe(
                [
                    *decision.reason_codes,
                    "V3_4_SEARCH_WORTHY_BUT_DEFERRED_BY_BOUNDED_SCHEDULER",
                    "NONSELECTED_ACTIVE_PARENT_RETAINED_WITHOUT_REPRODUCTION",
                ]
            )
            decisions.append(
                decision.model_copy(
                    update={
                        "disposition": "HOLD_STABLE",
                        "fertile_for_child_generation": False,
                        "carry_forward_if_not_replaced": True,
                        "recommended_channels": [],
                        "nonbinding_operator_hints": [],
                        "reason_codes": reasons,
                    }
                )
            )
        elif pressure.preferred_action == "PROBE":
            probes.append(decision.idea_id)
            decisions.append(
                decision.model_copy(
                    update={
                        "disposition": "EVIDENCE_PROBE_ONLY",
                        "fertile_for_child_generation": False,
                        "carry_forward_if_not_replaced": True,
                        "recommended_channels": [],
                        "nonbinding_operator_hints": [],
                        "reason_codes": _dedupe(
                            [
                                *decision.reason_codes,
                                "V3_4_CONTEMPORANEOUS_PRIOR_ART_REMAINS_UNRESOLVED",
                                "PROBE_DOES_NOT_CREATE_RESEARCH_IDEA_CHILD",
                            ]
                        ),
                    }
                )
            )
        elif pressure.preferred_action == "REALIZE":
            stable.append(decision.idea_id)
            decisions.append(
                decision.model_copy(
                    update={
                        "disposition": "CONTINUE_SAME_IDEA",
                        "fertile_for_child_generation": False,
                        "carry_forward_if_not_replaced": True,
                        "recommended_channels": [],
                        "nonbinding_operator_hints": [],
                        "reason_codes": _dedupe(
                            [
                                *decision.reason_codes,
                                "V3_4_CONTINUE_SAME_IDEA_REALIZATION_BEFORE_CHILD_SEARCH",
                            ]
                        ),
                    }
                )
            )
        else:
            stable.append(decision.idea_id)
            decisions.append(
                decision.model_copy(
                    update={
                        "disposition": "HOLD_STABLE",
                        "fertile_for_child_generation": False,
                        "carry_forward_if_not_replaced": True,
                        "recommended_channels": [],
                        "nonbinding_operator_hints": [],
                        "reason_codes": _dedupe(
                            [
                                *decision.reason_codes,
                                "V3_4_RETAINED_WITHOUT_CURRENT_REPRODUCTION_PRESSURE",
                            ]
                        ),
                    }
                )
            )

    counts = Counter(row.disposition for row in decisions)
    active_ids = [row.idea_id for row in decisions if row.remains_active]
    fertile_ids = [row.idea_id for row in decisions if row.fertile_for_child_generation]
    persistent_nonfertile = [
        row.idea_id
        for row in decisions
        if row.remains_active and not row.fertile_for_child_generation
    ]

    provisional = base_fertility.model_copy(
        update={
            "report_id": "pending",
            "report_sha256": "pending",
            "decisions": decisions,
            "decision_count": len(decisions),
            "active_idea_count": len(active_ids),
            "fertile_idea_count": len(fertile_ids),
            "persistent_nonfertile_idea_count": len(persistent_nonfertile),
            "dropped_inactive_idea_count": counts.get("DROP_INACTIVE", 0),
            "fertile_handoff_count": len(fertile_ids),
            "disposition_counts": dict(sorted(counts.items())),
            "active_idea_ids": active_ids,
            "fertile_idea_ids": fertile_ids,
            "persistent_nonfertile_idea_ids": persistent_nonfertile,
        }
    )
    body = provisional.model_dump(mode="json")
    body.pop("report_id", None)
    body.pop("report_sha256", None)
    digest = _sha(body)
    compatibility = provisional.model_copy(
        update={
            "report_id": f"sis_v3_4_compat_fertility:g{base_fertility.cycle_generation_index}:{digest[:20]}",
            "report_sha256": digest,
        }
    )

    selected_parent_ids = [row.idea_id for row in schedule.selected_parents]

    control_provisional = NoveltyAwareReproductionControlReport(
        report_id="pending",
        report_sha256="pending",
        generation_index=base_fertility.cycle_generation_index,
        source_pressure_report_id=pressure_report.report_id,
        source_schedule_id=schedule.schedule_id,
        source_base_fertility_report_id=base_fertility.report_id,
        compatibility_fertility_report_id=compatibility.report_id,
        selected_parent_idea_ids=selected_parent_ids,
        deferred_search_worthy_idea_ids=sorted(deferred),
        probe_only_idea_ids=sorted(probes),
        stable_retained_idea_ids=sorted(stable),
        selected_parent_count=len(selected_parent_ids),
        deferred_search_worthy_count=len(deferred),
        probe_only_count=len(probes),
        stable_retained_count=len(stable),
    )
    control_body = control_provisional.model_dump(mode="json")
    control_body.pop("report_id", None)
    control_body.pop("report_sha256", None)
    control_digest = _sha(control_body)
    control = control_provisional.model_copy(
        update={
            "report_id": f"sis_v3_4_reproduction_control:g{base_fertility.cycle_generation_index}:{control_digest[:20]}",
            "report_sha256": control_digest,
        }
    )
    return compatibility, control


_NOVELTY_PROMPT_ADDENDUM = r"""

SIS-v3.4 NOVELTY-AWARE SEARCH CONTEXT
=====================================
The caller may append bounded external prior-art and novelty-residue information.
This information is SEARCH FEEDBACK ONLY.

- Prior-art matches are not positive premises for the child ResearchIdea.
- A residual claim is not an established fact and does not prove novelty.
- Absence of a direct match is not literature-wide novelty proof.
- Do not invent citations or claim that retrieval established scientific truth.
- Do not merely paraphrase saturated/known relations as if they were novel.
- Prefer a child that sharpens, challenges, escapes, conditions, or experimentally
  discriminates the supplied unresolved residue while preserving task relevance.
- A known relation may still be used as background inspiration, but novelty must not
  be claimed from wording changes around that known relation.
"""


class NoveltyAwareOffspringBackendAdapter:
    """Inject v3.4 prior-art residue into the existing offspring transport.

    The wrapped backend and existing semantic identity/duplicate machinery remain
    unchanged. Only the prompt surface is augmented. This keeps novelty steering
    separate from evidence and truth authority.
    """

    def __init__(
        self,
        *,
        backend: Any,
        context_by_idea: Mapping[str, ResearchIdeaMutationSearchContext],
        task_to_idea: Mapping[str, str],
    ) -> None:
        self.backend = backend
        self.context_by_idea = dict(context_by_idea)
        self.task_to_idea = dict(task_to_idea)
        self.augmented_prompts: dict[str, EpistemicG4Prompt] = {}

    def _augment(self, prompt: EpistemicG4Prompt) -> EpistemicG4Prompt:
        idea_id = self.task_to_idea.get(prompt.task_id)
        context = self.context_by_idea.get(str(idea_id or ""))
        if context is None:
            return prompt
        try:
            payload = json.loads(prompt.user_prompt)
            if not isinstance(payload, dict):
                payload = {"base_generation_payload": payload}
        except Exception:
            payload = {"base_generation_prompt": prompt.user_prompt}
        payload["prior_art_and_novelty_residue_search_context_only"] = context.model_dump(
            mode="json"
        )
        payload["v3_4_generation_rules"] = [
            "Preserve saturated prior-art claims as search context, not positive premises.",
            "Do not emit a local rephrase of saturated claims as the main conceptual change.",
            "Use residual/partial/unresolved claims to motivate a substantive scientific move.",
            "Prefer a child with a new mediator, regime, proxy challenge, causal contrast, or decisive discriminator when supported by the parent search context.",
            "Abstain rather than fabricating evidence or novelty.",
        ]
        system = prompt.system_prompt + _NOVELTY_PROMPT_ADDENDUM
        user = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        augmented = EpistemicG4Prompt(
            task_id=prompt.task_id,
            system_prompt=system,
            user_prompt=user,
            prompt_sha256=_sha({"system": system, "user": user}),
        )
        self.augmented_prompts[prompt.task_id] = augmented
        return augmented

    def generate(self, prompt: EpistemicG4Prompt) -> Any:
        return self.backend.generate(self._augment(prompt))

    def repair(self, prompt: EpistemicG4Prompt, previous_draft: Any, feedback: str) -> Any:
        augmented = self.augmented_prompts.get(prompt.task_id) or self._augment(prompt)
        return self.backend.repair(augmented, previous_draft, feedback)


__all__ = [
    "PriorArtSearchClaimContext",
    "ResearchIdeaMutationSearchContext",
    "NoveltyAwareReproductionControlReport",
    "NoveltyAwareOffspringBackendAdapter",
    "build_mutation_search_contexts",
    "adapt_fertility_report_v3_4",
]
