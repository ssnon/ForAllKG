from __future__ import annotations

from pathlib import Path
import re

import pytest

from scripts.discovery.build_task_conditioned_axis_plan import (
    _endpoint_resolution_observability,
    _resolve_task_endpoints,
)


E2E = Path(
    "scripts/discovery/"
    "run_dac_discovery_e2e.py"
)


def test_explicit_pair_wins_over_conflicting_parseable_question():
    resolved = _resolve_task_endpoints(
        question=(
            "How does legacy source "
            "relate to legacy target?"
        ),
        requested_source="authoritative source",
        requested_target="authoritative target",
    )

    assert resolved == (
        "authoritative source",
        "authoritative target",
    )


def test_explicit_pair_does_not_require_legacy_question_grammar():
    resolved = _resolve_task_endpoints(
        question="Generate a scientific hypothesis.",
        requested_source="source concept",
        requested_target="target concept",
    )

    assert resolved == (
        "source concept",
        "target concept",
    )


def test_no_explicit_pair_preserves_legacy_grammar_fallback():
    resolved = _resolve_task_endpoints(
        question=(
            "How does legacy source "
            "relate to legacy target?"
        ),
        requested_source=None,
        requested_target=None,
    )

    assert resolved == (
        "legacy source",
        "legacy target",
    )


def test_no_explicit_pair_and_unparseable_question_remains_not_applicable():
    resolved = _resolve_task_endpoints(
        question="Generate a scientific hypothesis.",
        requested_source=None,
        requested_target=None,
    )

    assert resolved is None


@pytest.mark.parametrize(
    (
        "requested_source",
        "requested_target",
    ),
    [
        ("source only", None),
        (None, "target only"),
    ],
)
def test_one_sided_explicit_endpoint_fails_closed(
    requested_source: str | None,
    requested_target: str | None,
):
    with pytest.raises(
        ValueError,
        match="provided together",
    ):
        _resolve_task_endpoints(
            question=(
                "How does legacy source "
                "relate to legacy target?"
            ),
            requested_source=requested_source,
            requested_target=requested_target,
        )


@pytest.mark.parametrize(
    (
        "requested_source",
        "requested_target",
    ),
    [
        ("", "target"),
        ("source", ""),
        ("   ", "target"),
        ("source", "   "),
    ],
)
def test_blank_explicit_endpoint_fails_closed(
    requested_source: str,
    requested_target: str,
):
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        _resolve_task_endpoints(
            question=(
                "How does legacy source "
                "relate to legacy target?"
            ),
            requested_source=requested_source,
            requested_target=requested_target,
        )


def test_explicit_endpoints_are_trimmed_without_rewriting_content():
    resolved = _resolve_task_endpoints(
        question="irrelevant",
        requested_source=(
            "  HaB binding sites structural motif  "
        ),
        requested_target=(
            "  SERS plasmonic behavior  "
        ),
    )

    assert resolved == (
        "HaB binding sites structural motif",
        "SERS plasmonic behavior",
    )


def test_parent_stage75_forwards_source_and_target_authoritatively():
    source = E2E.read_text(
        encoding="utf-8"
    )

    start_marker = (
        '    runner.run_stage(\n'
        '        "[7.5/13] Task-conditioned '
        'discovery-axis plan",'
    )
    end_marker = (
        "    dual_context = "
        "task_conditioned_dual_context"
    )

    start = source.index(start_marker)
    end = source.index(
        end_marker,
        start,
    )
    block = source[start:end]

    assert (
        '"--question", str(args.question),'
        in block
    )
    assert (
        '"--requested-source", '
        'str(args.source),'
        in block
    )
    assert (
        '"--requested-target", '
        'str(args.target),'
        in block
    )


def test_s18a_does_not_wire_explicit_endpoints_into_other_stages():
    source = E2E.read_text(
        encoding="utf-8"
    )

    assert source.count(
        '"--requested-source"'
    ) == 1
    assert source.count(
        '"--requested-target"'
    ) == 1

def test_explicit_resolution_mode_is_recorded_even_when_question_parses():
    mode, matched = _endpoint_resolution_observability(
        question=(
            "How does legacy source "
            "relate to legacy target?"
        ),
        requested_source="explicit source",
        requested_target="explicit target",
    )

    assert mode == "EXPLICIT_RUNNER_ENDPOINTS_V1"
    assert matched is True


def test_explicit_resolution_mode_records_unparseable_question():
    mode, matched = _endpoint_resolution_observability(
        question="Which architecture is most favorable?",
        requested_source="explicit source",
        requested_target="explicit target",
    )

    assert mode == "EXPLICIT_RUNNER_ENDPOINTS_V1"
    assert matched is False


def test_legacy_resolution_mode_and_match_state_are_separate():
    mode, matched = _endpoint_resolution_observability(
        question=(
            "How does legacy source "
            "relate to legacy target?"
        ),
        requested_source=None,
        requested_target=None,
    )

    assert mode == "LEGACY_QUESTION_GRAMMAR_V1"
    assert matched is True


def test_stage75_report_wires_endpoint_resolution_provenance():
    source = Path(
        "scripts/discovery/"
        "build_task_conditioned_axis_plan.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        '"endpoint_resolution_mode":'
        in source
    )
    assert (
        '"question_grammar_matched":'
        in source
    )


def test_every_no_task_composite_report_preserves_endpoint_provenance():
    source = Path(
        "scripts/discovery/"
        "build_task_conditioned_axis_plan.py"
    ).read_text(
        encoding="utf-8"
    )

    header = re.compile(
        r'(?P<indent>[ \t]+)"status":\n'
        r'(?P=indent)[ \t]+"NO_TASK_COMPOSITE",\n'
        r'(?P=indent)"grammar":\n'
        r'(?P=indent)[ \t]+'
        r'"HOW_DOES_RELATE_TO_GRAMMAR_V1",\n'
    )

    matches = list(
        header.finditer(source)
    )

    assert matches

    for match in matches:
        tail = source[
            match.end():
            match.end() + 700
        ]

        before_generic = tail.split(
            '"generic_bundle_changed":',
            1,
        )[0]

        assert (
            '"endpoint_resolution_mode":'
            in before_generic
        )
        assert (
            '"requested_source":'
            in before_generic
        )
        assert (
            '"requested_target":'
            in before_generic
        )
        assert (
            '"question_grammar_matched":'
            in before_generic
        )
