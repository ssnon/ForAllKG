from argparse import Namespace
from pathlib import Path

import pytest

from scripts.discovery.run_dac_discovery_e2e import (
    _run_question_task_preservation_shadow_chain,
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


def option_value(argv, option):
    index = argv.index(option)

    return argv[index + 1]


def test_shadow_chain_wires_both_child_stages_without_selection_mutation():
    runner = FakeRunner()

    run = Path(
        "/tmp/q02_shadow_integration"
    )

    raw = (
        run
        / "question_task_preservation."
          "semantic_conflicts.shadow.json"
    )

    final = (
        run
        / "traversal.json"
    )

    candidate = (
        run
        / "candidate_unit.traversal.a3.json"
    )

    args = Namespace(
        question=(
            "Which structural properties of SERS architectures "
            "relate jointly to electromagnetic enhancement and "
            "measurement reproducibility?"
        ),
        critic_model="openai/gpt-5.6-luna",
        model="fallback-model",
    )

    responsiveness, proposals = (
        _run_question_task_preservation_shadow_chain(
            runner=runner,
            args=args,
            run=run,
            semantic_conflict_shadow=raw,
            final_traversal=final,
            candidate_traversal=candidate,
        )
    )

    assert len(
        runner.calls
    ) == 2

    first = runner.calls[0]
    second = runner.calls[1]

    assert first["module"] == (
        "scripts.discovery."
        "run_question_task_conflict_responsiveness"
    )

    assert second["module"] == (
        "scripts.discovery."
        "build_question_task_preservation_shadow"
    )

    assert option_value(
        first["argv"],
        "--raw-conflicts",
    ) == str(raw)

    traversal_indexes = [
        i
        for i, value in enumerate(
            first["argv"]
        )
        if value == "--traversal"
    ]

    assert len(
        traversal_indexes
    ) == 2

    assert first["argv"][
        traversal_indexes[0] + 1
    ] == str(final)

    assert first["argv"][
        traversal_indexes[1] + 1
    ] == str(candidate)

    assert option_value(
        first["argv"],
        "--question",
    ) == args.question

    assert option_value(
        first["argv"],
        "--model",
    ) == "openai/gpt-5.6-luna"

    assert option_value(
        second["argv"],
        "--raw-conflicts",
    ) == str(raw)

    assert option_value(
        second["argv"],
        "--responsiveness-audit",
    ) == str(
        responsiveness
    )

    assert option_value(
        second["argv"],
        "--output",
    ) == str(
        proposals
    )

    assert first[
        "expected"
    ] == [
        responsiveness
    ]

    assert second[
        "expected"
    ] == [
        proposals
    ]

    # The helper has no Bundle/axis/hypothesis output argument.
    joined = " ".join(
        first["argv"]
        + second["argv"]
    )

    assert (
        "discovery.bundle.a3.json"
        not in joined
    )

    assert (
        "hypothesis_axis"
        not in joined
    )


def test_shadow_chain_fails_closed_without_model():
    runner = FakeRunner()

    args = Namespace(
        question="q",
        critic_model=None,
        model=None,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "requires --critic-model "
            "or --model"
        ),
    ):
        _run_question_task_preservation_shadow_chain(
            runner=runner,
            args=args,
            run=Path("/tmp/x"),
            semantic_conflict_shadow=Path(
                "/tmp/x/raw.json"
            ),
            final_traversal=Path(
                "/tmp/x/final.json"
            ),
            candidate_traversal=Path(
                "/tmp/x/candidate.json"
            ),
        )

    assert runner.calls == []
