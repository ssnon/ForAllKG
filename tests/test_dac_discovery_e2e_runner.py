import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts.discovery import run_novelty_refinement as novelty_refinement_runner

from scripts.discovery.run_dac_discovery_e2e import (
    _data_root_args,
    _mechanism_index_args,
    _returned_path_count,
    parse_args,
    run_pipeline,
)


def test_returned_path_count_supports_current_and_fallback_shapes():
    assert _returned_path_count({"returned_path_count": 4}) == 4
    assert _returned_path_count({"paths": [{}, {}]}) == 2
    assert _returned_path_count({"summary": {"returned_path_count": 3}}) == 3
    assert _returned_path_count({}) == 0


def test_data_root_args_preserve_default_behavior():
    assert (
        _data_root_args(
            SimpleNamespace(
                data_root=None,
            )
        )
        == []
    )

    assert (
        _data_root_args(
            SimpleNamespace(
                data_root="",
            )
        )
        == []
    )


def test_data_root_args_forward_explicit_override():
    assert _data_root_args(
        SimpleNamespace(
            data_root="/tmp/sers-scratch",
        )
    ) == [
        "--data-root",
        "/tmp/sers-scratch",
    ]


def test_e2e_parser_accepts_data_root_override(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dac_discovery_e2e",
            "--run-dir",
            "/tmp/e2e-run",
            "--source",
            "source",
            "--target",
            "target",
            "--question",
            "question",
            "--data-root",
            "/tmp/sers-scratch",
        ],
    )

    args = parse_args()

    assert (
        args.data_root
        == "/tmp/sers-scratch"
    )


def test_e2e_forwards_data_root_to_all_traversal_lanes():
    source = inspect.getsource(
        run_pipeline
    )

    assert (
        source.count(
            "*_data_root_args(args)"
        )
        == 3
    )

    assert (
        source.count(
            '"scripts.discovery.run_graph_traversal"'
        )
        == 2
    )

    assert (
        source.count(
            '"scripts.discovery.run_candidate_unit_traversal"'
        )
        == 1
    )


def test_mechanism_index_args_preserve_default_behavior():
    assert (
        _mechanism_index_args(
            SimpleNamespace(
                data_root=None,
                corpus_id="corpus-a",
            )
        )
        == []
    )


def test_mechanism_index_args_follow_custom_data_root():
    assert _mechanism_index_args(
        SimpleNamespace(
            data_root="/tmp/sers-root",
            corpus_id="sers-corpus",
        )
    ) == [
        "--index-dir",
        (
            "/tmp/sers-root/corpus/"
            "sers-corpus/mechanism/"
            "navigation/node_index"
        ),
    ]


def test_e2e_forwards_custom_mechanism_index_to_embedding_consumers():
    source = inspect.getsource(
        run_pipeline
    )

    assert (
        source.count(
            "*_mechanism_index_args(args)"
        )
        == 2
    )


def test_novelty_refinement_parser_accepts_explicit_index_dir(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_novelty_refinement",
            "--dual-context", "dual.json",
            "--domain-profile", "sers_au_ag",
            "--axis-plan", "axis.json",
            "--portfolio", "portfolio.json",
            "--lineage", "lineage.json",
            "--external-report", "external.json",
            "--external-query-plan", "queries.json",
            "--external-prior-art", "prior.json",
            "--index-dir", "/tmp/sers-index",
            "--output-prefix", "/tmp/refinement",
        ],
    )

    args = novelty_refinement_runner.parse_args()

    assert args.index_dir == Path(
        "/tmp/sers-index"
    )
