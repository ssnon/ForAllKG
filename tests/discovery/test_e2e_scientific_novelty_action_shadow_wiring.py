from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.discovery.run_dac_discovery_e2e import (
    _run_scientific_novelty_action_shadow_chain,
)


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run_stage(
        self,
        name,
        module,
        argv,
        *,
        expected=None,
    ):
        self.calls.append(
            {
                "name": name,
                "module": module,
                "argv": list(argv),
                "expected": list(
                    expected or []
                ),
            }
        )


def _value(argv, option):
    index = argv.index(
        option
    )
    return argv[index + 1]


def test_shadow_chain_materializes_two_pass_reviews_and_action_batch(
    tmp_path,
):
    external = (
        tmp_path
        / "external.report.json"
    )

    external.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "hypothesis_id":
                            "hypothesis:h1",
                        "status":
                            "LITERATURE_SUPPORTED_EXTENSION",
                    },
                    {
                        "hypothesis_id":
                            "hypothesis:h2",
                        "status":
                            "KNOWN_COMPONENTS_WITH_RELATIONAL_GAP",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = (
        tmp_path
        / "external.plan.json"
    )

    prior = (
        tmp_path
        / "external.prior.json"
    )

    plan.write_text(
        "{}",
        encoding="utf-8",
    )

    prior.write_text(
        "{}",
        encoding="utf-8",
    )

    runner = FakeRunner()

    args = Namespace(
        critic_model="openai/gpt-5.6-luna",
        model="fallback-model",
    )

    action_path = (
        _run_scientific_novelty_action_shadow_chain(
            runner=runner,
            args=args,
            run=tmp_path,
            external_report=external,
            external_plan=plan,
            external_prior=prior,
        )
    )

    # 1 scientific report + 2 hypotheses * 2 passes + 1 action batch.
    assert len(
        runner.calls
    ) == 6

    assert runner.calls[0][
        "module"
    ] == (
        "scripts.discovery."
        "run_scientific_distinctiveness_diagnostic"
    )

    semantic_calls = (
        runner.calls[1:5]
    )

    assert all(
        call["module"]
        == (
            "scripts.discovery."
            "run_semantic_distinctiveness_review"
        )
        for call in semantic_calls
    )

    assert [
        _value(
            call["argv"],
            "--hypothesis-id",
        )
        for call in semantic_calls
    ] == [
        "hypothesis:h1",
        "hypothesis:h1",
        "hypothesis:h2",
        "hypothesis:h2",
    ]

    assert [
        _value(
            call["argv"],
            "--review-pass-index",
        )
        for call in semantic_calls
    ] == [
        "1",
        "2",
        "1",
        "2",
    ]

    assert all(
        _value(
            call["argv"],
            "--model",
        )
        == "openai/gpt-5.6-luna"
        for call in semantic_calls
    )

    final_call = runner.calls[-1]

    assert final_call[
        "module"
    ] == (
        "scripts.discovery."
        "build_scientific_novelty_action_shadow"
    )

    assert (
        final_call["argv"].count(
            "--semantic-review"
        )
        == 4
    )

    assert _value(
        final_call["argv"],
        "--output",
    ) == str(
        action_path
    )

    # No production portfolio is an output of this chain.
    joined = " ".join(
        arg
        for call in runner.calls
        for arg in call["argv"]
    )

    assert (
        "novelty_refinement_a6.portfolio.json"
        not in joined
    )


def test_shadow_chain_requires_semantic_model(
    tmp_path,
):
    external = (
        tmp_path
        / "external.report.json"
    )

    external.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "hypothesis_id":
                            "hypothesis:h1",
                        "status":
                            "LITERATURE_SUPPORTED_EXTENSION",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = FakeRunner()

    args = Namespace(
        critic_model=None,
        model=None,
    )

    try:
        _run_scientific_novelty_action_shadow_chain(
            runner=runner,
            args=args,
            run=tmp_path,
            external_report=external,
            external_plan=tmp_path / "plan.json",
            external_prior=tmp_path / "prior.json",
        )
    except RuntimeError as exc:
        assert (
            "requires --critic-model or --model"
            in str(exc)
        )
    else:
        raise AssertionError(
            "missing semantic model must fail closed"
        )

    # Scientific report is deterministic and may have been scheduled first,
    # but semantic/action stages must not be scheduled without a model.
    assert len(
        runner.calls
    ) == 1
