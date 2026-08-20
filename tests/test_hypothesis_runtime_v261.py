from __future__ import annotations

from pipeline_core.discovery.hypothesis_contracts import HypothesisPortfolioDraft
from dac_her.hypothesis_llm import HypothesisDraftGeneration
from dac_her.hypothesis_runtime import HypothesisMakerAgentRuntime

from _hypothesis_v261_fixtures import (
    make_bad_novelty_draft,
    make_context,
    make_valid_draft,
)


class FakeBackend:
    backend_name = "fake_hypothesis_backend"
    model_name = "fake-model"
    temperature = 0.0
    instructor_mode = "FAKE"
    base_url = "fake://local"
    parse_retries = 0

    def __init__(
        self,
        initial: HypothesisPortfolioDraft,
        repairs: list[HypothesisPortfolioDraft] | None = None,
    ) -> None:
        self.initial = initial
        self.repairs = list(repairs or [])
        self.feedback: list[str] = []

    def generate(self, prompt):
        return HypothesisDraftGeneration(
            draft=self.initial,
            input_tokens=10,
            output_tokens=5,
        )

    def repair(self, prompt, previous_draft, feedback):
        self.feedback.append(feedback)
        if not self.repairs:
            raise AssertionError("unexpected repair call")
        return HypothesisDraftGeneration(
            draft=self.repairs.pop(0),
            input_tokens=12,
            output_tokens=6,
        )


def test_runtime_accepts_valid_first_generation():
    context = make_context()
    backend = FakeBackend(make_valid_draft())
    outcome = HypothesisMakerAgentRuntime(backend).run(context)

    assert outcome.accepted
    assert outcome.accepted_portfolio is not None
    assert outcome.run_record.final_validation_passed
    assert outcome.run_record.failure_stage == "none"
    assert outcome.run_record.generation_attempts == 1
    assert outcome.run_record.repair_attempts == 0
    assert outcome.run_record.context_sha256 == context.context_sha256
    assert outcome.run_record.portfolio_sha256
    assert outcome.run_record.input_tokens == 10
    assert outcome.run_record.output_tokens == 5


def test_runtime_repairs_compile_failure_once():
    context = make_context()
    bad = make_valid_draft(premise_id="s:gap")
    backend = FakeBackend(bad, repairs=[make_valid_draft()])

    outcome = HypothesisMakerAgentRuntime(backend, max_repairs=1).run(context)

    assert outcome.accepted
    assert outcome.run_record.generation_attempts == 2
    assert outcome.run_record.repair_attempts == 1
    assert len(outcome.draft_history) == 2
    assert backend.feedback
    assert "INELIGIBLE_POSITIVE_PREMISE" in backend.feedback[0]


def test_runtime_repairs_validation_failure_once():
    context = make_context()
    backend = FakeBackend(make_bad_novelty_draft(), repairs=[make_valid_draft()])

    outcome = HypothesisMakerAgentRuntime(backend, max_repairs=1).run(context)

    assert outcome.accepted
    assert outcome.run_record.generation_attempts == 2
    assert outcome.run_record.repair_attempts == 1
    assert "EXTERNAL_NOVELTY_CLAIM" in backend.feedback[0]


def test_runtime_fails_closed_when_repair_disabled():
    context = make_context()
    bad = make_valid_draft(premise_id="s:gap")
    backend = FakeBackend(bad)

    outcome = HypothesisMakerAgentRuntime(backend, max_repairs=0).run(context)

    assert not outcome.accepted
    assert outcome.accepted_portfolio is None
    assert outcome.last_portfolio is None
    assert outcome.validation is None
    assert outcome.run_record.failure_stage == "compile"
    assert outcome.run_record.final_validation_passed is False
    assert outcome.run_record.compile_issue_count == 1
    assert outcome.compile_issues[0].code == "INELIGIBLE_POSITIVE_PREMISE"


def test_candidate_dependency_is_deterministically_propagated():
    context = make_context()
    backend = FakeBackend(make_valid_draft(premise_id="s:candidate"))

    outcome = HypothesisMakerAgentRuntime(backend).run(context)

    assert outcome.accepted
    assert outcome.accepted_portfolio is not None
    card = outcome.accepted_portfolio.hypotheses[0]
    assert card.candidate_dependency == "essential"
    assert card.evidence_profile.candidate_premise_count == 1
