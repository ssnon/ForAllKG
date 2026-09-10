from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.discovery import run_dac_discovery_e2e as e2e


def argv(
    *extra,
):
    return [
        "run_dac_discovery_e2e",
        "--run-dir",
        "/tmp/run",
        "--source",
        "source",
        "--target",
        "target",
        "--question",
        "question",
        *extra,
    ]


def test_public_nonobviousness_switch_now_requires_data_root(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        argv(
            "--nonobviousness-enforce"
        ),
    )

    with pytest.raises(
        SystemExit,
    ):
        e2e.parse_args()


def test_public_nonobviousness_switch_enables_complete_n10_path(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        argv(
            "--data-root",
            "/tmp/data",
            "--nonobviousness-enforce",
        ),
    )

    args = e2e.parse_args()

    assert (
        args.nonobviousness_original_fallback_enforce
        is True
    )

    assert (
        args.nonobviousness_post_generation_enforce
        is True
    )

    assert (
        args.nonobviousness_bounded_continuation_enforce
        is True
    )


def test_bounded_flag_requires_post_generation_n10(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        argv(
            "--data-root",
            "/tmp/data",
            "--nonobviousness-bounded-continuation-enforce",
        ),
    )

    with pytest.raises(
        SystemExit,
    ):
        e2e.parse_args()


def test_bounded_flag_requires_explicit_data_root(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        argv(
            "--nonobviousness-enforce",
            "--nonobviousness-bounded-continuation-enforce",
        ),
    )

    with pytest.raises(
        SystemExit,
    ):
        e2e.parse_args()


def test_bounded_flag_accepts_public_n10_plus_explicit_data_root(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        argv(
            "--data-root",
            "/tmp/data",
            "--nonobviousness-enforce",
            "--nonobviousness-bounded-continuation-enforce",
        ),
    )

    args = e2e.parse_args()

    assert (
        args.nonobviousness_original_fallback_enforce
        is True
    )

    assert (
        args.nonobviousness_post_generation_enforce
        is True
    )

    assert (
        args.nonobviousness_bounded_continuation_enforce
        is True
    )

    assert args.data_root == "/tmp/data"


def test_main_e2e_contains_bounded_execution_between_h1_n10_and_final_count():
    path = Path(
        e2e.__file__
    )

    text = path.read_text(
        encoding="utf-8"
    )

    first_n10 = text.index(
        '"[11N10/13] Fresh Alpha6 candidate '
        'non-obviousness enforcement"'
    )

    bounded = text.index(
        '"[11N10-R/13] Bounded diagnostic "'
    )

    merge = text.index(
        '"[11N10-M/13] Merge authoritative "'
    )

    final_count = text.index(
        "final_hypotheses = _hypothesis_count(refined_portfolio)"
    )

    assert (
        first_n10
        < bounded
        < merge
        < final_count
    )


def test_bounded_stage_receives_explicit_data_root_derived_index():
    text = Path(
        e2e.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert (
        'bounded_index_dir = ('
        in text
    )

    assert (
        '/ "corpus"\n'
        '                / args.corpus_id\n'
        '                / "mechanism"\n'
        '                / "navigation"\n'
        '                / "node_index"'
        in text
    )

    assert (
        '"--index-dir",\n'
        '                    str(\n'
        '                        bounded_index_dir'
        in text
    )


def test_disabled_bounded_path_still_consumes_first_post_n10_portfolio():
    text = Path(
        e2e.__file__
    ).read_text(
        encoding="utf-8"
    )

    first_n10 = text.index(
        '"[11N10/13] Fresh Alpha6 candidate '
        'non-obviousness enforcement"'
    )

    default_binding = text.index(
        "refined_portfolio = post_n10_portfolio",
        first_n10,
    )

    bounded_branch = text.index(
        "if args.nonobviousness_bounded_continuation_enforce:",
        default_binding,
    )

    bounded_override = text.index(
        "refined_portfolio = (\n"
        "                bounded_final_portfolio",
        bounded_branch,
    )

    final_count = text.index(
        "final_hypotheses = _hypothesis_count(refined_portfolio)",
        bounded_override,
    )

    assert (
        first_n10
        < default_binding
        < bounded_branch
        < bounded_override
        < final_count
    )


def test_bounded_merge_output_becomes_stage12_input():
    text = Path(
        e2e.__file__
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "refined_portfolio = (\n"
        "                bounded_final_portfolio\n"
        "            )"
        in text
    )

    assert (
        "--first-post-n10-portfolio"
        in text
    )

    assert (
        "--continuation-report"
        in text
    )

    assert (
        "merge_n10_bounded_continuation_portfolio"
        in text
    )
