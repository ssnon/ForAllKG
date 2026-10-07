from __future__ import annotations

import copy
from types import SimpleNamespace

from pipeline_core.discovery.research_idea_adaptive_fertility import (
    build_adaptive_fertility_report,
    build_fertility_gated_parallel_view,
    compose_persistent_population_execution,
)
from pipeline_core.discovery.research_idea_closed_generation_cycle import (
    build_legacy_offspring_execution_payload,
)
from pipeline_core.discovery.research_idea_epistemic_generational_evolution import (
    EpistemicG4ExecutionReport,
)


def member(
    idea_id: str,
    maturity: str,
    *,
    active: bool = True,
    speculative: bool = False,
):
    mid = f"m:{idea_id}:{maturity}"
    return SimpleNamespace(
        member_id=mid,
        idea_id=idea_id,
        epistemic_realization_id=f"e:{idea_id}:{maturity}",
        epistemic_maturity=maturity,
        archive_retained=active,
    )


def handoff(idea_id: str):
    return SimpleNamespace(
        handoff_id=f"h:{idea_id}",
        idea_id=idea_id,
        generation_index=5,
        source_member_ids=[],
        source_epistemic_realization_ids=[],
        recommended_channels=["TRANSFORM", "EXPLORE"],
        nonbinding_operator_hints=["AXIS_MUTATION", "REGIME_BOUNDARY"],
        reason_codes=["SOURCE_HANDOFF"],
    )


def debt(idea_id: str, kind: str = "RELATION_EVIDENCE", *, acquisition: bool = False):
    return SimpleNamespace(
        debt_id=f"d:{idea_id}:{kind}",
        idea_id=idea_id,
        requirement_kind=kind,
        disposition=(
            "ACQUISITION_ESCALATION_ELIGIBLE" if acquisition else "PROBE_ELIGIBLE"
        ),
    )


def parallel_fixture(specs):
    members = []
    idea_states = []
    handoffs = []
    debts = []
    for spec in specs:
        idea_id = spec["idea_id"]
        rows = [member(idea_id, maturity) for maturity in spec.get("maturities", [])]
        members.extend(rows)
        mids = [row.member_id for row in rows]
        idea_states.append(
            SimpleNamespace(
                idea_id=idea_id,
                generation_index=spec.get("birth_generation", 5),
                active_member_ids=mids,
                remains_in_search_population=spec.get("active", True),
            )
        )
        if spec.get("handoff", True):
            h = handoff(idea_id)
            h.source_member_ids = mids
            h.source_epistemic_realization_ids = [row.epistemic_realization_id for row in rows]
            handoffs.append(h)
        debts.extend(spec.get("debts", []))
    return SimpleNamespace(
        report_id="parallel:g5",
        members=members,
        idea_states=idea_states,
        epistemic_debts=debts,
        evolution_handoffs=handoffs,
    )


def local_state(
    idea_id: str,
    *,
    usable: int = 0,
    exhausted: bool = False,
    realization_count: int = 1,
    failed: int = 1,
    not_op: int = 0,
    rescued: bool = False,
    budget: int = 2,
    used: int = 1,
):
    return SimpleNamespace(
        idea_id=idea_id,
        realization_count=realization_count,
        usable_grounded_realization_count=usable,
        failed_or_abstained_count=failed,
        not_operationalizable_count=not_op,
        local_search_budget=budget,
        local_search_budget_used=used,
        local_search_exhausted=exhausted,
        rescued_within_same_idea=rescued,
    )


def lifecycle_fixture(*states):
    return SimpleNamespace(report_id="lifecycle:g5", local_states=list(states))


def test_grounded_debt_free_active_idea_persists_without_becoming_fertile():
    p = parallel_fixture(
        [{"idea_id": "stable", "maturities": ["STRICT_GROUNDED"]}]
    )
    lifecycle = lifecycle_fixture(
        local_state("stable", usable=1, failed=0, exhausted=False)
    )
    report = build_adaptive_fertility_report(
        parallel_report=p,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    decision = report.decisions[0]
    assert decision.disposition == "HOLD_STABLE"
    assert decision.remains_active is True
    assert decision.fertile_for_child_generation is False
    assert report.active_idea_count == 1
    assert report.fertile_idea_count == 0
    assert report.persistent_nonfertile_idea_ids == ["stable"]


def test_non_grounded_open_local_space_continues_same_idea_before_mutation():
    p = parallel_fixture(
        [{"idea_id": "open", "maturities": ["EVIDENCE_SEEKING"]}]
    )
    lifecycle = lifecycle_fixture(
        local_state("open", usable=0, exhausted=False, realization_count=1, used=1)
    )
    report = build_adaptive_fertility_report(
        parallel_report=p,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    decision = report.decisions[0]
    assert decision.disposition == "CONTINUE_SAME_IDEA"
    assert decision.fertile_for_child_generation is False
    assert "SAME_IDEA_SEARCH_PRECEDES_IDEA_MUTATION" in decision.reason_codes


def test_exhausted_not_operationalizable_idea_diversifies_and_gates_handoff():
    p = parallel_fixture(
        [{"idea_id": "div", "maturities": ["EVIDENCE_SEEKING"]}]
    )
    lifecycle = lifecycle_fixture(
        local_state(
            "div",
            usable=0,
            exhausted=True,
            realization_count=2,
            failed=2,
            not_op=1,
            used=2,
        )
    )
    report = build_adaptive_fertility_report(
        parallel_report=p,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    decision = report.decisions[0]
    assert decision.disposition == "DIVERSIFY"
    assert decision.fertile_for_child_generation is True
    assert decision.recommended_channels == ["EXPLORE", "WILDCARD"]

    gated = build_fertility_gated_parallel_view(
        parallel_report=p,
        fertility_report=report,
    )
    assert len(gated.evolution_handoffs) == 1
    assert gated.evolution_handoffs[0].recommended_channels == ["EXPLORE", "WILDCARD"]


def test_exhausted_single_axis_gap_evolves_child_but_stable_sibling_does_not():
    p = parallel_fixture(
        [
            {
                "idea_id": "evolve",
                "maturities": ["EVIDENCE_SEEKING"],
                "debts": [debt("evolve", "RELATION_EVIDENCE")],
            },
            {"idea_id": "stable", "maturities": ["STRICT_GROUNDED"]},
        ]
    )
    lifecycle = lifecycle_fixture(
        local_state(
            "evolve",
            usable=0,
            exhausted=True,
            realization_count=2,
            failed=2,
            used=2,
        ),
        local_state("stable", usable=1, failed=0),
    )
    report = build_adaptive_fertility_report(
        parallel_report=p,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    by_id = {row.idea_id: row for row in report.decisions}
    assert by_id["evolve"].disposition == "EVOLVE_CHILD"
    assert by_id["evolve"].fertile_for_child_generation is True
    assert by_id["stable"].disposition == "HOLD_STABLE"
    assert by_id["stable"].fertile_for_child_generation is False
    assert report.fertile_idea_count == 1
    assert report.active_idea_count == 2


def node(idea_id: str, generation: int = 5):
    return SimpleNamespace(
        idea_id=idea_id,
        generation_index=generation,
        parent_idea_ids=[],
        source_context_id="ctx",
        source_context_sha256="sha",
    )


class FakeExecution:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_copy(self, *, update):
        values = copy.copy(self.__dict__)
        values.update(update)
        return FakeExecution(**values)

    def model_dump(self, mode="json"):
        def dump(value):
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if isinstance(value, SimpleNamespace):
                return {k: dump(v) for k, v in vars(value).items()}
            if isinstance(value, list):
                return [dump(v) for v in value]
            if isinstance(value, dict):
                return {k: dump(v) for k, v in value.items()}
            return value

        return {k: dump(v) for k, v in self.__dict__.items()}


def test_population_persistence_carries_stable_and_replaces_only_productive_fertile_parent():
    p = parallel_fixture(
        [
            {"idea_id": "stable", "maturities": ["STRICT_GROUNDED"]},
            {"idea_id": "evolve", "maturities": ["EVIDENCE_SEEKING"]},
        ]
    )
    lifecycle = lifecycle_fixture(
        local_state("stable", usable=1, failed=0),
        local_state("evolve", exhausted=True, realization_count=2, failed=2, used=2),
    )
    fertility = build_adaptive_fertility_report(
        parallel_report=p,
        lifecycle=lifecycle,
        cycle_generation_index=5,
    )
    current = SimpleNamespace(
        report_id="g5:current",
        generation_index=5,
        g4_population_nodes=[node("stable"), node("evolve")],
    )
    child = node("child", generation=6)
    raw = FakeExecution(
        report_id="g6:raw",
        report_sha256="x",
        source_parallel_report_id="gated",
        source_plan_id="plan:g6",
        generation_index=6,
        tasks=[],
        run_records=[],
        offspring_nodes=[child],
        semantic_records=[
            SimpleNamespace(idea_id="child", primary_parent_idea_id="evolve")
        ],
        g4_population_nodes=[child],
        raw_offspring_count=1,
        g4_population_count=1,
        genuine_child_count=1,
        indeterminate_probe_count=0,
        same_idea_refinement_count=0,
        exact_kernel_duplicate_suppressed_count=0,
        identity_relation_counts={"DIFFERENT_IDEA": 1},
        disposition_counts={"GENUINE_CHILD": 1},
        generated_count_by_channel={"TRANSFORM": 1},
        generated_count_by_operator={"AXIS_MUTATION": 1},
        genuine_child_from_non_grounded_feedback_count=1,
        genuine_child_from_speculative_feedback_count=0,
        genuine_child_from_evidence_seeking_feedback_count=1,
        llm_call_count=1,
        input_tokens=1,
        output_tokens=1,
        semantic_retry_count=0,
        offspring_generation_executed=True,
        carried_forward_idea_ids=[],
        carried_forward_count=0,
        replaced_parent_idea_ids=[],
        replaced_parent_count=0,
        population_composed_with_persistence=False,
        population_growth_budget=0,
    )
    plan = SimpleNamespace(
        source_parallel_report_id="gated",
        plan_id="plan:g6",
        tasks=[],
        selected_parent_idea_ids=["evolve"],
    )
    composed, persistence = compose_persistent_population_execution(
        current_execution=current,
        fertility_report=fertility,
        next_generation_index=6,
        next_plan=plan,
        raw_next_execution=raw,
        population_growth_budget=0,
    )
    assert set(persistence.final_population_idea_ids) == {"stable", "child"}
    assert persistence.carried_forward_idea_ids == ["stable"]
    assert persistence.replaced_parent_idea_ids == ["evolve"]
    assert persistence.population_growth_count == 0
    assert set(row.idea_id for row in composed.g4_population_nodes) == {"stable", "child"}


def test_carryover_only_execution_is_valid_and_realization_adapter_keeps_persistent_idea():
    execution = EpistemicG4ExecutionReport(
        report_id="g6:carry",
        report_sha256="x",
        source_parallel_report_id="parallel:g5",
        source_plan_id="plan:g6",
        generation_index=6,
        tasks=[],
        run_records=[],
        offspring_nodes=[],
        semantic_records=[],
        g4_population_nodes=[node("stable", generation=5)],
        raw_offspring_count=0,
        g4_population_count=1,
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
        carried_forward_idea_ids=["stable"],
        carried_forward_count=1,
        population_composed_with_persistence=True,
    )
    payload = build_legacy_offspring_execution_payload(execution)
    assert payload["raw_offspring_count"] == 1
    assert payload["accepted_for_realization_count"] == 1
    assert payload["semantic_records"][0]["idea_id"] == "stable"
    assert payload["semantic_records"][0]["channel"] == "EXPLOIT"
    assert payload["semantic_records"][0]["disposition"] == "ACCEPTED_REFINEMENT"
