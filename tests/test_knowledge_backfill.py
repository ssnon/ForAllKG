from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.literature.knowledge_backfill_runtime import (
    KnowledgeAwareBackfillCoordinator,
    KnowledgeBackfillOptions,
    KnowledgeBackfillPaths,
    outcome_meets_target,
    summarize_outcomes,
    write_dynamic_target_profile,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _base_outcome(paper_id: str, *, bridge="BRIDGE_USEFUL", projection="PROJECTION_USABLE", eligible=True):
    return {
        "paper_id": paper_id,
        "strict_status": "STRICT_USABLE",
        "bridge_status": bridge,
        "projection_status": projection,
        "corpus_eligible": eligible,
    }


def test_bridge_useful_requires_projection_and_corpus_eligibility():
    assert outcome_meets_target(_base_outcome("a"), "BRIDGE_USEFUL")
    assert not outcome_meets_target(
        _base_outcome("b", projection="PROJECTION_ERROR"), "BRIDGE_USEFUL"
    )
    assert not outcome_meets_target(
        _base_outcome("c", eligible=False), "BRIDGE_USEFUL"
    )


def test_summary_deduplicates_latest_paper_outcome():
    rows = [
        _base_outcome("a", bridge="BRIDGE_EMPTY"),
        _base_outcome("a"),
        _base_outcome("b"),
    ]
    summary = summarize_outcomes(rows, target_status="BRIDGE_USEFUL")
    assert summary["paper_outcome_count"] == 2
    assert summary["target_status_count"] == 2
    assert summary["target_status_paper_ids"] == ["a", "b"]


def test_dynamic_profile_scales_axis_quotas_with_total_target(tmp_path: Path):
    source = tmp_path / "profile.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "profile_id": "p",
                "domain_profile_id": "sers",
                "selection": {"target_total": 100, "include_manual_review": False},
                "axes": [{"axis_id": "a", "target_selected": 12}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "round.yaml"
    write_dynamic_target_profile(source_profile=source, output_path=output, target_total=7)
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["selection"]["target_total"] == 7
    assert payload["selection"]["include_manual_review"] is False
    # Original quota fraction is 12/100; target 7 therefore receives one
    # round-local quota slot instead of retaining the invalid quota of 12.
    assert payload["axes"][0]["target_selected"] == 1
    assert sum(axis["target_selected"] for axis in payload["axes"]) <= 7



def test_dynamic_profile_apportions_sers_like_100_to_23(tmp_path: Path):
    source = tmp_path / "profile.yaml"
    original = [12, 10, 14, 14, 12, 14, 12, 12]
    source.write_text(
        yaml.safe_dump(
            {
                "profile_id": "sers",
                "domain_profile_id": "sers",
                "selection": {"target_total": 100},
                "axes": [
                    {"axis_id": f"a{i}", "target_selected": value}
                    for i, value in enumerate(original)
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "round.yaml"
    write_dynamic_target_profile(
        source_profile=source,
        output_path=output,
        target_total=23,
    )
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    quotas = [axis["target_selected"] for axis in payload["axes"]]
    assert payload["selection"]["target_total"] == 23
    assert quotas == [3, 2, 3, 3, 3, 3, 3, 3]
    assert sum(quotas) == 23


def test_dynamic_profile_preserves_global_fill_fraction(tmp_path: Path):
    source = tmp_path / "profile.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "profile_id": "p",
                "domain_profile_id": "d",
                "selection": {"target_total": 100},
                "axes": [
                    {"axis_id": "a", "target_selected": 20},
                    {"axis_id": "b", "target_selected": 20},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "round.yaml"
    write_dynamic_target_profile(
        source_profile=source,
        output_path=output,
        target_total=25,
    )
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    # Source reserves 40% for axis quotas and 60% for global fill.
    assert [axis["target_selected"] for axis in payload["axes"]] == [5, 5]
    assert sum(axis["target_selected"] for axis in payload["axes"]) == 10


def _fixture(tmp_path: Path, outcomes) -> tuple[KnowledgeBackfillPaths, Path]:
    root = tmp_path
    configs = root / "configs"
    data = root / "data_sers"
    acquisition = root / "data_acquisition"
    m3 = acquisition / "m3_2"
    m3.mkdir(parents=True)
    _write_jsonl(
        m3 / "selected_works.jsonl",
        [{"work_id": "w1"}, {"work_id": "w2"}],
    )
    _write_json(
        m3 / "selection_report.json",
        {"profile_id": "profile", "selected_work_ids": ["w1", "w2"]},
    )
    _write_json(m3 / "acquisition_report.json", {})

    profile = configs / "profile.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(
        yaml.safe_dump(
            {
                "schema_version": "x",
                "profile_id": "profile",
                "domain_profile_id": "sers_au_ag",
                "selection": {"target_total": 100},
                "axes": [{"axis_id": "a", "target_selected": 10}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    files = {}
    for name in [
        "backfill.yaml",
        "source.yaml",
        "catalog.json",
        "m2.jsonl",
        "quality.jsonl",
        "quality_report.json",
        "materialization.yaml",
        "gate.yaml",
    ]:
        path = configs / name
        path.write_text("{}\n" if path.suffix != ".jsonl" else "", encoding="utf-8")
        files[name] = path

    m4 = acquisition / "m4"
    m4.mkdir(parents=True)
    _write_json(m4 / "materialization_report.json", {"materialization_id": "m4id"})
    m45 = acquisition / "m4_5"
    m45.mkdir(parents=True)
    (m45 / "extraction_plan.jsonl").write_text("", encoding="utf-8")
    m4config = configs / "m4.yaml"
    strictconfig = configs / "strict.yaml"
    m4config.write_text("version: 3\npapers: {}\n", encoding="utf-8")
    strictconfig.write_text("version: 3\npapers: {}\n", encoding="utf-8")

    outcomes_path = data / "pipeline_runs" / "corpus" / "strict_bridge" / "paper_outcomes.jsonl"
    _write_jsonl(outcomes_path, outcomes)

    paths = KnowledgeBackfillPaths(
        acquisition_profile=profile,
        backfill_policy=files["backfill.yaml"],
        source_policy=files["source.yaml"],
        catalog=files["catalog.json"],
        m2_assessments=files["m2.jsonl"],
        quality_assessments=files["quality.jsonl"],
        quality_gate_report=files["quality_report.json"],
        starting_m3_dir=m3,
        materialization_policy=files["materialization.yaml"],
        m4_dir=m4,
        m4_config=m4config,
        gate_policy=files["gate.yaml"],
        m4_5_dir=m45,
        strict_config=strictconfig,
        data_root=data,
        run_root=acquisition / "m6",
    )
    return paths, root


def test_target_already_satisfied_is_noop(tmp_path: Path):
    paths, root = _fixture(tmp_path, [_base_outcome("p1"), _base_outcome("p2")])
    calls = []
    coordinator = KnowledgeAwareBackfillCoordinator(
        project_root=root,
        corpus_id="corpus",
        domain_profile="sers_au_ag",
        paper_id_prefix="SERS_API",
        paths=paths,
        options=KnowledgeBackfillOptions(target_count=2),
        command_runner=lambda command, label: calls.append((label, command)) or True,
    )
    result = coordinator.run()
    assert result["status"] == "target_already_satisfied"
    assert result["extra_candidates_added"] == 0
    assert calls == []


def test_dry_run_requests_only_knowledge_deficit_and_preserves_m3_authority(tmp_path: Path):
    paths, root = _fixture(
        tmp_path,
        [_base_outcome("p1"), _base_outcome("p2", bridge="BRIDGE_EMPTY")],
    )
    calls = []
    coordinator = KnowledgeAwareBackfillCoordinator(
        project_root=root,
        corpus_id="corpus",
        domain_profile="sers_au_ag",
        paper_id_prefix="SERS_API",
        paths=paths,
        options=KnowledgeBackfillOptions(target_count=2, dry_run=True),
        command_runner=lambda command, label: calls.append((label, command)) or True,
    )
    result = coordinator.run()
    assert result["status"] == "dry_run"
    assert len(calls) == 2
    recovery_label, recovery_command = calls[0]
    assert recovery_label.endswith("access_recovery")
    assert "scripts.literature.prepare_access_recovery" in recovery_command
    label, command = calls[1]
    assert label.endswith("m3_2_backfill")
    dynamic_profile = Path(command[command.index("--profile") + 1])
    profile = yaml.safe_load(dynamic_profile.read_text(encoding="utf-8"))
    # 2 currently selected + 1 missing BRIDGE_USEFUL target.
    assert profile["selection"]["target_total"] == 3
    assert sum(axis["target_selected"] for axis in profile["axes"]) <= 3
    assert "scripts.literature.backfill_acquisition_ready_corpus" in command


def test_oversample_factor_and_budget_are_applied_to_next_round(tmp_path: Path):
    paths, root = _fixture(tmp_path, [_base_outcome("p1")])
    calls = []
    coordinator = KnowledgeAwareBackfillCoordinator(
        project_root=root,
        corpus_id="corpus",
        domain_profile="sers_au_ag",
        paper_id_prefix="SERS_API",
        paths=paths,
        options=KnowledgeBackfillOptions(
            target_count=4,
            oversample_factor=2.0,
            max_extra_candidates=4,
            dry_run=True,
        ),
        command_runner=lambda command, label: calls.append((label, command)) or True,
    )
    coordinator.run()
    command = next(
        command for label, command in calls if label.endswith("m3_2_backfill")
    )
    dynamic_profile = Path(command[command.index("--profile") + 1])
    profile = yaml.safe_load(dynamic_profile.read_text(encoding="utf-8"))
    # deficit=3, 2x would request 6, but safety budget caps at 4.
    assert profile["selection"]["target_total"] == 6  # 2 selected + 4 reserve slots
    assert sum(axis["target_selected"] for axis in profile["axes"]) <= 6


def test_full_feedback_round_adds_one_candidate_then_reaches_target(tmp_path: Path):
    paths, root = _fixture(
        tmp_path,
        [_base_outcome("p1"), _base_outcome("p2", bridge="BRIDGE_EMPTY")],
    )
    calls = []
    outcomes_path = (
        paths.data_root
        / "pipeline_runs"
        / "corpus"
        / "strict_bridge"
        / "paper_outcomes.jsonl"
    )

    def fake_run(command, label):
        calls.append(label)
        if label.endswith("m3_2_backfill"):
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            _write_jsonl(
                output / "selected_works.jsonl",
                [{"work_id": "w1"}, {"work_id": "w2"}, {"work_id": "w3"}],
            )
            _write_json(
                output / "selection_report.json",
                {
                    "profile_id": "profile",
                    "selected_work_ids": ["w1", "w2", "w3"],
                },
            )
            _write_json(output / "acquisition_report.json", {})
        elif label.endswith("strict_bridge"):
            _write_jsonl(
                outcomes_path,
                [
                    _base_outcome("p1"),
                    _base_outcome("p2", bridge="BRIDGE_EMPTY"),
                    _base_outcome("p3"),
                ],
            )
        return True

    coordinator = KnowledgeAwareBackfillCoordinator(
        project_root=root,
        corpus_id="corpus",
        domain_profile="sers_au_ag",
        paper_id_prefix="SERS_API",
        paths=paths,
        options=KnowledgeBackfillOptions(target_count=2),
        command_runner=fake_run,
    )
    result = coordinator.run()
    assert result["status"] == "target_reached"
    assert result["extra_candidates_added"] == 1
    assert result["final"]["target_status_count"] == 2
    assert calls == [
        "r001:access_recovery",
        "r001:m3_2_backfill",
        "r001:m4_materialize",
        "r001:m4_5_gate",
        "r001:strict_bridge",
    ]


def test_reserve_exhaustion_stops_before_expensive_followup(tmp_path: Path):
    paths, root = _fixture(
        tmp_path,
        [_base_outcome("p1"), _base_outcome("p2", bridge="BRIDGE_EMPTY")],
    )
    calls = []

    def fake_run(command, label):
        calls.append(label)
        if label.endswith("m3_2_backfill"):
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            # No new reserve paper selected.
            _write_jsonl(
                output / "selected_works.jsonl",
                [{"work_id": "w1"}, {"work_id": "w2"}],
            )
            _write_json(
                output / "selection_report.json",
                {"profile_id": "profile", "selected_work_ids": ["w1", "w2"]},
            )
            _write_json(output / "acquisition_report.json", {})
        return True

    coordinator = KnowledgeAwareBackfillCoordinator(
        project_root=root,
        corpus_id="corpus",
        domain_profile="sers_au_ag",
        paper_id_prefix="SERS_API",
        paths=paths,
        options=KnowledgeBackfillOptions(target_count=2),
        command_runner=fake_run,
    )
    result = coordinator.run()
    assert result["status"] == "reserve_exhausted"
    assert calls == ["r001:access_recovery", "r001:m3_2_backfill"]


def test_access_recovery_flags_propagate_to_prep_and_m3(tmp_path: Path):
    paths, root = _fixture(
        tmp_path,
        [_base_outcome("p1"), _base_outcome("p2", bridge="BRIDGE_EMPTY")],
    )
    calls = []
    coordinator = KnowledgeAwareBackfillCoordinator(
        project_root=root,
        corpus_id="corpus",
        domain_profile="sers_au_ag",
        paper_id_prefix="SERS_API",
        paths=paths,
        options=KnowledgeBackfillOptions(
            target_count=2,
            retry_failed_acquisition=True,
            retry_access_misses=True,
            dry_run=True,
        ),
        command_runner=lambda command, label: calls.append((label, command)) or True,
    )
    coordinator.run()
    recovery = next(
        command for label, command in calls if label.endswith("access_recovery")
    )
    m3 = next(
        command for label, command in calls if label.endswith("m3_2_backfill")
    )
    assert "--retry-failed" in recovery
    assert "--retry-access-misses" in recovery
    assert "--retry-failed" in m3
