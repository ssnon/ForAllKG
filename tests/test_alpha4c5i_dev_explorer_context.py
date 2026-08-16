from __future__ import annotations

from types import SimpleNamespace

import dac_her.alpha4c5i_dev_explorer_context as devx


def _packet(
    papers=None,
    *,
    corpus_id=devx.SOURCE_CORPUS_ID,
    domain="sers_au_ag",
):
    if papers is None:
        papers = [f"P{i:02d}" for i in range(53)]
    return SimpleNamespace(
        domain_profile_id=domain,
        packet_id="packet:1",
        packet_sha256="a" * 64,
        task=SimpleNamespace(task_id="task:1"),
        corpus=SimpleNamespace(
            corpus_id=corpus_id,
            projection_mode="evidence",
            papers=[
                SimpleNamespace(paper_id=value)
                for value in papers
            ],
        ),
    )


def test_packet_exact_dev_passes():
    exact = [f"P{i:02d}" for i in range(53)]
    assert devx.verify_packet_exact_dev(
        _packet(exact),
        exact,
    ) == []


def test_packet_wrong_partition_fails():
    exact = [f"P{i:02d}" for i in range(53)]
    issues = devx.verify_packet_exact_dev(
        _packet(exact[:-1]),
        exact,
    )
    assert any("exact 53-paper DEV" in row for row in issues)


def test_report_must_bind_packet_sha_and_task():
    packet = _packet()
    report = SimpleNamespace(
        source_packet_sha256="b" * 64,
        task_id="other",
    )
    issues = devx.verify_report_lineage(
        packet,
        report,
    )
    assert len(issues) == 2


def test_context_lineage_checks_packet_report_corpus(monkeypatch):
    monkeypatch.setattr(
        devx,
        "validate_hypothesis_context_sha",
        lambda value: None,
    )
    packet = _packet()
    report = SimpleNamespace(
        report_id="report:1",
        task_id="task:1",
        source_packet_sha256=packet.packet_sha256,
    )
    context = SimpleNamespace(
        source_packet_id=packet.packet_id,
        source_packet_sha256=packet.packet_sha256,
        source_report_id=report.report_id,
        task_id=packet.task.task_id,
        corpus_id=devx.SOURCE_CORPUS_ID,
        domain_profile_id="sers_au_ag",
    )
    assert devx.verify_context_lineage(
        packet,
        report,
        context,
    ) == []


def test_manifest_never_authorizes_reserve_use(tmp_path):
    output = tmp_path / "evaluation/dev_explorer"
    input_root = tmp_path / "evaluation/dev_input"
    output.mkdir(parents=True)
    input_root.mkdir(parents=True)

    # Minimal files used by artifact binding.
    for path in (
        output / "explorer/packet.json",
        output / "explorer/explorer.report.json",
        output / "hypothesis/hypothesis_context.json",
        input_root / "trend_aware_hypothesis_input.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    packet = SimpleNamespace(
        packet_id="packet:1",
        packet_sha256="a" * 64,
    )
    report = SimpleNamespace(report_id="report:1")
    context = SimpleNamespace(
        context_id="context:1",
        context_sha256="b" * 64,
    )
    trend_input = SimpleNamespace(
        input_id="input:1",
        input_sha256="c" * 64,
    )
    payload = devx.build_execution_manifest(
        root=tmp_path,
        output_root=output,
        input_output_root=input_root,
        packet=packet,
        report=report,
        context=context,
        trend_input=trend_input,
        model="synthetic-model",
        base_url=None,
    )
    assert payload["reserve_a_scientific_read"] is False
    assert payload["reserve_b_scientific_read"] is False
    assert payload["reserve_b_rerun"] is False
    assert payload["new_extraction"] is False


def test_test_source_has_no_real_reserve_identity():
    forbidden = "SERS" + "_API_"
    from pathlib import Path
    assert forbidden not in Path(__file__).read_text(
        encoding="utf-8"
    )
