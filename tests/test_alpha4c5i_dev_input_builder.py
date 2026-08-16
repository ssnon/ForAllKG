from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import dac_her.alpha4c5i_dev_input_builder as builder


def test_closed_reserve_path_guard():
    assert builder.path_is_closed_reserve(
        Path("evaluation/x/reserve_b_v1/file.json")
    )
    assert builder.path_is_closed_reserve(
        Path("evaluation/x/reserve-a/file.json")
    )
    assert not builder.path_is_closed_reserve(
        Path("evaluation/x/dev_compat_v1/file.json")
    )


def test_existing_context_deduplicates_by_context_sha(
    tmp_path: Path,
    monkeypatch,
):
    binding = SimpleNamespace(
        domain_profile_id="sers_au_ag",
        corpus_id="dev_corpus",
    )
    fake_context = SimpleNamespace(
        domain_profile_id="sers_au_ag",
        corpus_id="dev_corpus",
        context_sha256="a" * 64,
    )

    p1 = tmp_path / "evaluation/a/hypothesis_context.json"
    p2 = tmp_path / "evaluation/b/hypothesis_context_copy.json"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_text("{}\n", encoding="utf-8")
    p2.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        builder,
        "_parse_context",
        lambda path: fake_context,
    )
    monkeypatch.setattr(
        builder,
        "DISCOVERY_ROOTS",
        (Path("evaluation"),),
    )

    candidates = builder.discover_existing_contexts(
        root=tmp_path,
        binding=binding,
    )
    assert len(candidates) == 2


def test_packet_exact_dev_requires_full_partition():
    binding = SimpleNamespace(
        domain_profile_id="sers_au_ag",
        corpus_id="dev_corpus",
    )
    packet = SimpleNamespace(
        domain_profile_id="sers_au_ag",
        corpus=SimpleNamespace(
            corpus_id="dev_corpus",
            papers=[
                SimpleNamespace(paper_id=f"P{i:02d}")
                for i in range(53)
            ],
        ),
    )
    exact = [f"P{i:02d}" for i in range(53)]
    assert builder._packet_exact_dev(
        packet=packet,
        binding=binding,
        exact_dev=exact,
    )
    assert not builder._packet_exact_dev(
        packet=packet,
        binding=binding,
        exact_dev=exact[:-1],
    )


def test_explicit_packet_report_must_be_paired(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="must be supplied together"):
        builder.select_or_build_context(
            root=tmp_path,
            binding=SimpleNamespace(),
            exact_dev=[],
            explicit_packet=Path("packet.json"),
            explicit_report=None,
        )


def test_no_context_or_pair_fails_closed(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        builder,
        "discover_existing_contexts",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        builder,
        "discover_packet_report_pairs",
        lambda **kwargs: [],
    )
    with pytest.raises(ValueError, match="No matching DEV53"):
        builder.select_or_build_context(
            root=tmp_path,
            binding=SimpleNamespace(),
            exact_dev=[],
        )


def test_multiple_contexts_fail_closed(
    tmp_path: Path,
    monkeypatch,
):
    c1 = builder.ContextCandidate(
        context=SimpleNamespace(context_sha256="a" * 64),
        source_path=tmp_path / "c1.json",
        source_kind="existing_context",
    )
    c2 = builder.ContextCandidate(
        context=SimpleNamespace(context_sha256="b" * 64),
        source_path=tmp_path / "c2.json",
        source_kind="existing_context",
    )
    monkeypatch.setattr(
        builder,
        "discover_existing_contexts",
        lambda **kwargs: [c1, c2],
    )
    with pytest.raises(ValueError, match="Multiple distinct"):
        builder.select_or_build_context(
            root=tmp_path,
            binding=SimpleNamespace(),
            exact_dev=[],
        )


def test_test_source_has_no_real_reserve_identity():
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = "SERS" + "_API_"
    assert forbidden not in source
