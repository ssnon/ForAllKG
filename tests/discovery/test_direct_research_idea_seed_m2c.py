from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.discovery import run_direct_research_idea_seed_m2c as m2c


class Row:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, mode="json"):
        def conv(value):
            if isinstance(value, Row):
                return value.model_dump(mode=mode)
            if isinstance(value, list):
                return [conv(v) for v in value]
            if isinstance(value, dict):
                return {k: conv(v) for k, v in value.items()}
            return value
        return conv(vars(self))


def test_exact_parity_order_and_kernel_changes_are_detected():
    a = Row(offspring_nodes=[], g4_population_nodes=[{"idea_id": "A", "kernel": {"canonical_intent": "alpha"}}])
    m2c._assert_exact("p0", a, a)
    with pytest.raises(ValueError, match="exact parity FAILED"):
        m2c._assert_exact("p0", a, Row(g4_population_nodes=[{"idea_id": "A", "kernel": {"canonical_intent": "beta"}}]))
    with pytest.raises(ValueError, match="exact parity FAILED"):
        m2c._assert_exact("p0", Row(g4_population_nodes=[1, 2]), Row(g4_population_nodes=[2, 1]))


def test_pool_evaluation_requires_all_exact_candidate_ids():
    pool = Row(pool_id="p", pool_sha256="hash", candidates=[Row(candidate_id="c1"), Row(candidate_id="c2")])
    good = Row(source_pool_id="p", source_pool_sha256="hash", evaluations=[Row(candidate_id="c2"), Row(candidate_id="c1")])
    m2c._validate_pool_evaluation(pool, good)
    with pytest.raises(ValueError, match="exactly once"):
        m2c._validate_pool_evaluation(pool, Row(source_pool_id="p", source_pool_sha256="hash", evaluations=[Row(candidate_id="c1")]))
    with pytest.raises(ValueError, match="duplicate"):
        m2c._validate_pool_evaluation(pool, Row(source_pool_id="p", source_pool_sha256="hash", evaluations=[Row(candidate_id="c1"), Row(candidate_id="c1")]))
    with pytest.raises(ValueError, match="lineage SHA"):
        m2c._validate_pool_evaluation(pool, Row(source_pool_id="p", source_pool_sha256="wrong", evaluations=[]))


def test_m1_sha_pin_fails_closed(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"a":1}', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": [{"key": "stage7_candidate_pool", "present": True, "source_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}), encoding="utf-8")
    m2c._check_manifest_pin(manifest, "stage7_candidate_pool", artifact)
    artifact.write_text('{"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        m2c._check_manifest_pin(manifest, "stage7_candidate_pool", artifact)


def inputs(tmp_path, monkeypatch, eval_reuse=True):
    source = tmp_path / "source"
    source.mkdir()
    paths = {key: source / (key + ".json") for key in ("context", "population", "evolution", "pool", "evaluation", "selection", "p0")}
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")

    ctx = Row(context_id="ctx", context_sha256="sha")
    population = Row(population_id="pop", population_sha256="pop-sha", source_context_id="ctx", source_context_sha256="sha")
    evolution = Row(report_id="ev", report_sha256="ev-sha", source_population_id="pop", source_population_sha256="pop-sha")
    pool = Row(pool_id="pool", pool_sha256="pool-sha", source_population_id="pop", source_population_sha256="pop-sha", source_evolution_report_id="ev", source_evolution_report_sha256="ev-sha", source_context_id="ctx", source_context_sha256="sha", candidates=[Row(candidate_id="c1")])
    evaluation = Row(source_pool_id="pool", source_pool_sha256="pool-sha", evaluations=[Row(candidate_id="c1")])
    selection = Row(selection_id="select", retained_count=1)
    child = Row(idea_id="idea0", parent_idea_ids=[], kernel=Row(canonical_intent="seed"))
    seed = Row(seed_id="seed", seed_sha256="seed-sha", selected_candidate_count=1, research_ideas=[child])
    p0 = Row(report_id="execution", g4_population_nodes=[child.model_dump()])
    load_map = {"context": ctx, "population": population, "evolution": evolution, "pool": pool, "evaluation": evaluation, "selection": selection, "p0": p0}
    monkeypatch.setattr(m2c, "_load", lambda path, model: load_map[path.stem])
    monkeypatch.setattr(m2c, "build_scientific_portfolio_selection", lambda **kwargs: selection)
    monkeypatch.setattr(m2c, "build_e2e_research_idea_seed", lambda **kwargs: seed)
    monkeypatch.setattr(m2c, "build_e2e_seed_execution", lambda input_seed: p0)
    args = argparse.Namespace(
        context=paths["context"], population=paths["population"], evolution_report=paths["evolution"],
        candidate_pool=paths["pool"], frontier_audit=None,
        evaluation=paths["evaluation"] if eval_reuse else None,
        allow_evaluation_llm=not eval_reuse, task_source="molecular orientation", task_target="Raman intensity", model="test-model",
        api_key_env="NONE", base_url=None, instructor_mode="JSON", temperature=0, parse_retries=0, timeout=60,
        max_evaluation_candidates=48, max_retained=8, max_retained_per_profile=2,
        expected_selection=paths["selection"], expected_p0_execution=paths["p0"], m1_freeze_manifest=None,
        output_dir=tmp_path / "out", save_prompt=False,
    )
    return args, paths, load_map


def test_frozen_direct_seed_exact_parity_with_no_materialization(tmp_path, monkeypatch):
    args, paths, load_map = inputs(tmp_path, monkeypatch)
    class NoLLM:
        def __init__(self, **kwargs):
            raise AssertionError("backend should not be initialized in replay")
    monkeypatch.setattr(m2c, "InstructorOpenAICompatibleScientificPortfolioBackend", NoLLM)
    result = m2c.run(args)
    assert result["status"] == "DIRECT_P0_READY"
    assert result["early_materialization_llm_calls"] == 0
    assert result["portfolio_evaluation_llm_calls"] == 0
    assert result["old_p0_exact_parity_checked"]
    assert (args.output_dir / "p0.execution.json").exists()
    assert (args.output_dir / "p0.seed_report.json").exists()
    assert not list(args.output_dir.rglob("*materialized*"))
    original = [path.read_bytes() for path in paths.values()]
    with pytest.raises(FileExistsError, match="overwrite"):
        m2c.run(args)
    assert original == [path.read_bytes() for path in paths.values()]


def test_parity_failure_does_not_write_output(tmp_path, monkeypatch):
    args, paths, load_map = inputs(tmp_path, monkeypatch)
    load_map["p0"] = Row(report_id="changed", g4_population_nodes=[])
    # Deliberately simulate a mismatch between a generated execution and frozen baseline.
    monkeypatch.setattr(m2c, "build_e2e_seed_execution", lambda s: Row(report_id="new", g4_population_nodes=[{"idea_id": "idea0"}]))
    with pytest.raises(ValueError, match="exact parity FAILED"):
        m2c.run(args)
    assert not args.output_dir.exists()


def test_fresh_evaluation_explicit_opt_in_skips_materialization(tmp_path, monkeypatch):
    args, paths, load_map = inputs(tmp_path, monkeypatch, eval_reuse=False)
    args.expected_selection = None
    args.expected_p0_execution = None
    prompt = Row(system_prompt="sys", user_prompt="user")
    monkeypatch.setattr(m2c, "build_evaluation_prompt", lambda **kw: prompt)
    monkeypatch.setattr(m2c, "compile_scientific_portfolio_evaluation", lambda **kw: load_map["evaluation"])
    class EvaluationOnly:
        model_name = "test-model"
        def __init__(self, **kw): pass
        def evaluate(self, p):
            assert p is prompt
            return Row(draft=Row())
        def materialize(self, p):
            raise AssertionError("must never call Stage7 materialization")
    monkeypatch.setattr(m2c, "InstructorOpenAICompatibleScientificPortfolioBackend", EvaluationOnly)
    assert m2c.run(args)["portfolio_evaluation_llm_calls"] == 1
    assert not list(args.output_dir.rglob("*materialized*"))


def test_no_accidental_paid_evaluation(tmp_path, monkeypatch):
    args, _, _ = inputs(tmp_path, monkeypatch, eval_reuse=False)
    args.allow_evaluation_llm = False
    with pytest.raises(ValueError, match="disabled by default"):
        m2c.run(args)
    assert not args.output_dir.exists()


def test_fresh_candidate_projection_uses_existing_builder(tmp_path, monkeypatch):
    args, paths, load_map = inputs(tmp_path, monkeypatch)
    args.candidate_pool = None
    args.frontier_audit = paths["pool"]  # same stub path; actual type is FrontierExplorationAudit
    seen = []
    def build_pool(**kwargs):
        seen.append(kwargs)
        return load_map["pool"]
    monkeypatch.setattr(m2c, "build_scientific_portfolio_candidate_pool", build_pool)
    result = m2c.run(args)
    assert len(seen) == 1
    assert seen[0]["max_evaluation_candidates"] == 48
    assert result["old_p0_exact_parity_checked"]


def test_missing_frontier_audit_fails_before_writing(tmp_path, monkeypatch):
    args, paths, _ = inputs(tmp_path, monkeypatch)
    args.candidate_pool = None
    args.frontier_audit = None
    with pytest.raises(ValueError, match="frontier-audit"):
        m2c.run(args)
    assert not args.output_dir.exists()
