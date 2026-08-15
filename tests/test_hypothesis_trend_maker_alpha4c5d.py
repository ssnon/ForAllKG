from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import dac_her.hypothesis_trend_maker_exposure as exposure_module
import dac_her.hypothesis_trend_runtime as runtime_module
from dac_her.hypothesis_trend_compiler import (
    TrendHypothesisCompileError,
    TrendHypothesisCompileIssue,
)
from dac_her.hypothesis_trend_contracts import (
    TrendAwareHypothesisPortfolioDraft,
)
from dac_her.hypothesis_trend_input import (
    HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID,
)
from dac_her.hypothesis_trend_llm import (
    TrendAwareHypothesisDraftGeneration,
)
from dac_her.hypothesis_trend_maker_exposure import (
    build_trend_maker_exposure,
)
from dac_her.hypothesis_trend_prompt import (
    TrendAwareHypothesisPromptAssembler,
)
from dac_her.hypothesis_trend_runtime import (
    TrendAwareHypothesisMakerAgentRuntime,
)


class FakePortfolio(BaseModel):
    portfolio_id: str = "portfolio:test"
    hypotheses: list[object] = []


class FakeValidation:
    semantics_id = "validator:test"

    def __init__(self, passes: bool = True) -> None:
        self.passes = passes
        self.errors = 0 if passes else 1
        self.warnings = 0
        self.issues = []


class FakeCompiler:
    semantics_id = "compiler:test"

    def __init__(self, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.calls = 0

    def compile(self, source, draft):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise TrendHypothesisCompileError([
                TrendHypothesisCompileIssue(
                    code="MISSING_REPLICATION_GAP_COMPANION",
                    location="draft.hypotheses[0].trend_references",
                    message="gap required",
                )
            ])
        return FakePortfolio()


class FakeValidator:
    semantics_id = "validator:test"

    def validate(self, source, portfolio):
        return FakeValidation(True)


class FakeBackend:
    backend_name = "fake_alpha4c5d"
    model_name = "none"
    temperature = 0.0
    instructor_mode = "none"
    base_url = None
    parse_retries = 0

    def __init__(self) -> None:
        self.generate_calls = 0
        self.repair_calls = 0
        self.draft = TrendAwareHypothesisPortfolioDraft(
            hypotheses=[],
            abstention_reason="deterministic test abstention",
        )

    def generate(self, prompt):
        self.generate_calls += 1
        return TrendAwareHypothesisDraftGeneration(draft=self.draft)

    def repair(self, prompt, previous_draft, feedback):
        self.repair_calls += 1
        return TrendAwareHypothesisDraftGeneration(draft=self.draft)


def _policy():
    return SimpleNamespace(
        maker_consumption_enabled=False,
        prompt_modified=False,
        compiler_modified=False,
        validator_modified=False,
        runtime_modified=False,
        llm_calls_allowed=False,
        causality_authorized_by_trend_input=False,
        universal_relation_authorized_by_trend_input=False,
        unknown_context_fill_allowed=False,
        majority_direction_vote_allowed=False,
    )


def _view(
    lane: str,
    *,
    status: str = "insufficient",
    view_id: str,
    grounding_id: str = "grounding:1",
):
    return SimpleNamespace(
        view_id=view_id,
        grounding_id=grounding_id,
        relation_id="relation:1",
        lane=lane,
        cross_context_status=status,
        support_role="local_support_with_replication_gap",
        independent_variable_key="particle_size",
        dependent_observable_key="sers_performance",
        control_family="structural",
        observable_semantics="qualitative_sers_performance",
        paper_ids=["p1"],
        local_result_ids=["local:1"],
        member_trend_ids=["trend:1"],
        directions=["positive"],
        shapes=["monotonic"],
        evidence_kinds=["reported_claim"],
        evidence_bases=["reported_directional_claim"],
        source_claim_ids=["claim:1"],
        source_measurement_ids=[],
        source_measurement_result_ids=[],
        source_calculation_ids=[],
        source_node_ids=["claim:1"],
        differentiating_dimensions=[],
        unresolved_dimensions=["excitation_wavelength"],
        association_only_result_ids=[],
        source_asserted_causal_trend_ids=[],
        source_requires_verification_trend_ids=[],
        requires_context_qualification=False,
        requires_verification=False,
        directional_cross_paper_premise_allowed=False,
        maker_selectable=False,
        causal_use_allowed=False,
        universal_use_allowed=False,
    )


def _source(*, views=None):
    context = SimpleNamespace(
        task_id="task:test",
        question="test question",
        corpus_id="corpus:test",
        domain_profile_id="sers_au_ag",
        context_id="context:test",
        context_sha256="context-sha",
        source_packet_id="packet:test",
        source_packet_sha256="packet-sha",
        source_report_id="report:test",
        source_report_sha256="report-sha",
        evidence_statements=[],
        mechanism_routes=[],
        mechanistic_motifs=[],
        reported_design_levers=[],
        partial_absence_blocked_paper_ids=[],
    )
    return SimpleNamespace(
        input_id="trend_input:test",
        input_sha256="input-sha",
        contract_semantics_id=
            HYPOTHESIS_TREND_INPUT_CONTRACT_SEMANTICS_ID,
        input_semantics_id=(
            "sers_au_ag_hypothesis_trend_input_v1_alpha4c5b"
        ),
        domain_profile_id="sers_au_ag",
        corpus_id="corpus:test",
        grounded_context=context,
        trend_views=(
            views
            if views is not None
            else [
                _view(
                    "local_empirical_support",
                    view_id="view:local",
                ),
                _view(
                    "replication_gap",
                    view_id="view:gap",
                ),
            ]
        ),
        policy=_policy(),
    )


@pytest.fixture(autouse=True)
def _disable_external_source_verification(monkeypatch):
    monkeypatch.setattr(
        exposure_module,
        "verify_trend_aware_input_sources",
        lambda source: None,
    )
    monkeypatch.setattr(
        runtime_module,
        "verify_trend_aware_input_sources",
        lambda source: None,
    )


def test_5d_exposure_activates_without_mutating_5b_flags():
    source = _source()
    exposure = build_trend_maker_exposure(source)
    assert all(row.selectable_by_maker for row in exposure.views)
    assert all(not row.maker_selectable for row in source.trend_views)
    assert all(not row.causal_use_allowed for row in source.trend_views)
    assert all(not row.universal_use_allowed for row in source.trend_views)


def test_insufficient_local_support_exposes_gap_companion():
    exposure = build_trend_maker_exposure(_source())
    local = next(
        row for row in exposure.views
        if row.lane == "local_empirical_support"
    )
    gap = next(
        row for row in exposure.views
        if row.lane == "replication_gap"
    )
    assert local.allowed_use_role == "positive_empirical_support"
    assert gap.allowed_use_role == "replication_gap"
    assert [
        (row.use_role, row.view_id)
        for row in local.required_companions
    ] == [("replication_gap", gap.view_id)]



def test_reversed_exposure_preserves_both_companion_role_bindings():
    source = _source(views=[
        _view(
            "local_empirical_support",
            status="reversed",
            view_id="view:local",
        ),
        _view(
            "context_dependency_signal",
            status="reversed",
            view_id="view:context",
        ),
        _view(
            "reversal_boundary",
            status="reversed",
            view_id="view:reversal",
        ),
    ])
    exposure = build_trend_maker_exposure(source)
    local = next(
        row for row in exposure.views
        if row.lane == "local_empirical_support"
    )
    assert [
        (row.use_role, row.view_id)
        for row in local.required_companions
    ] == [
        ("context_qualification", "view:context"),
        ("counterevidence_boundary", "view:reversal"),
    ]


def test_prompt_preserves_separate_namespaces_and_exact_roles():
    source = _source()
    prompt = TrendAwareHypothesisPromptAssembler().build(source)
    assert "premise_statement_ids" in prompt.system_prompt
    assert "trend_references[].view_id" in prompt.system_prompt
    assert "view:local" in prompt.user_prompt
    assert "view:gap" in prompt.user_prompt
    assert "allowed_use_role=positive_empirical_support" in prompt.user_prompt
    assert "allowed_use_role=replication_gap" in prompt.user_prompt
    assert "REQUIRED COMPANIONS IF SELECTED AS POSITIVE SUPPORT" in prompt.user_prompt


def test_runtime_accepts_with_zero_repair():
    source = _source()
    backend = FakeBackend()
    compiler = FakeCompiler()
    runtime = TrendAwareHypothesisMakerAgentRuntime(
        backend,
        compiler=compiler,
        validator=FakeValidator(),
        max_repairs=1,
    )
    outcome = runtime.run(source)
    assert outcome.accepted
    assert backend.generate_calls == 1
    assert backend.repair_calls == 0
    assert compiler.calls == 1
    assert outcome.run_record.generation_attempts == 1
    assert outcome.run_record.repair_attempts == 0
    assert outcome.run_record.source_trend_input_id == source.input_id
    assert outcome.run_record.trend_exposure_id == outcome.exposure.exposure_id


def test_runtime_uses_at_most_one_contract_repair():
    source = _source()
    backend = FakeBackend()
    compiler = FakeCompiler(fail_first=True)
    runtime = TrendAwareHypothesisMakerAgentRuntime(
        backend,
        compiler=compiler,
        validator=FakeValidator(),
        max_repairs=1,
    )
    outcome = runtime.run(source)
    assert outcome.accepted
    assert backend.generate_calls == 1
    assert backend.repair_calls == 1
    assert compiler.calls == 2
    assert outcome.run_record.generation_attempts == 2
    assert outcome.run_record.repair_attempts == 1


def test_source_view_activation_drift_fails_closed():
    source = _source()
    source.trend_views[0].maker_selectable = True
    with pytest.raises(ValueError, match="must not mutate 5b maker_selectable"):
        build_trend_maker_exposure(source)


def test_zero_trend_views_remain_empty_and_prompt_allows_abstention():
    source = _source(views=[])
    exposure = build_trend_maker_exposure(source)
    assert exposure.views == []
    assert exposure.lane_counts == {}
    prompt = TrendAwareHypothesisPromptAssembler().build(
        source,
        exposure=exposure,
    )
    assert "TREND LOCAL EMPIRICAL SUPPORT" in prompt.user_prompt
    assert "- NONE" in prompt.user_prompt
    assert "abstain" in prompt.user_prompt.lower()
