from __future__ import annotations

import sys

from pipeline_core.discovery.hypothesis_contracts import (
    HypothesisContext,
    HypothesisEvidenceStatement,
)
from pipeline_core.discovery.hypothesis_prompt import HypothesisPromptAssembler
from scripts.discovery import run_discovery_axis_hypothesis_maker as axis_runner
from scripts.discovery import run_dac_discovery_e2e as e2e_runner


def _context() -> HypothesisContext:
    return HypothesisContext(
        context_id="ctx:test",
        context_sha256="0" * 64,
        source_packet_id="packet:test",
        source_packet_sha256="1" * 64,
        source_report_id="report:test",
        source_report_sha256="2" * 64,
        task_id="task:test",
        question="test question",
        corpus_id="test-corpus",
        domain_profile_id="dac_her",
        evidence_statements=[
            HypothesisEvidenceStatement(
                statement_id="stmt:test",
                text="A reported observation.",
                epistemic_role="reported",
                claim_kind="observation",
                paper_ids=["P1"],
                eligible_as_premise=True,
            )
        ],
    )


def test_prompt_explicitly_enumerates_expected_direction_contract():
    prompt = HypothesisPromptAssembler().build(_context())
    assert (
        "expected_direction MUST be exactly one of: "
        "increase, decrease, shift, non_monotonic, "
        "qualitative_change, unspecified."
        in prompt.user_prompt
    )
    assert "Do not use conditional" in prompt.user_prompt


def test_axis_runner_default_parse_retries_is_three(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_discovery_axis_hypothesis_maker",
            "--dual-context",
            "dummy.json",
            "--output-prefix",
            "dummy-output",
        ],
    )
    args = axis_runner.parse_args()
    assert args.parse_retries == 3


def test_e2e_default_hypothesis_parse_retries_is_three(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dac_discovery_e2e",
            "--run-dir", "dummy-run",
            "--source", "s",
            "--target", "t",
            "--question", "q",
        ],
    )
    args = e2e_runner.parse_args()
    assert args.hypothesis_parse_retries == 3
