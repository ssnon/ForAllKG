from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).parents[2] / "scripts" / "discovery" / "build_scientific_program_preservation_m2f.py"
spec = importlib.util.spec_from_file_location("m2f", MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def node(name, parents=(), origin="FRONTIER", generation=0):
    return {"idea_id": name, "kernel": {"canonical_intent": name}, "generation_index": generation,
            "parent_idea_ids": list(parents), "origin_kind": origin, "source_context_id": "ctx"}


def execution(generation, nodes, report):
    return {"generation_index": generation, "g4_population_nodes": nodes, "g4_population_count": len(nodes),
            "report_id": report, "report_sha256": "mock", "population_growth_budget": 0,
            "carried_forward_idea_ids": [], "replaced_parent_idea_ids": [], "carried_forward_count": 0,
            "replaced_parent_count": 0}


def fixture(tmp_path):
    run = tmp_path / "v34"
    sp = run / "case_root" / "scientific_portfolio_shadow"
    p0 = save(tmp_path / "p0.json", execution(2, [node("a"), node("b")], "p0"))
    g3 = execution(3, [node("b"), node("c", ["a"], "GENERATIONAL_OFFSPRING", 3)], "g3")
    g3["replaced_parent_idea_ids"] = ["a"]
    g3["replaced_parent_count"] = 1
    g3["carried_forward_idea_ids"] = ["b"]
    g3["carried_forward_count"] = 1
    save(sp / "sis_v3_4.g3_adaptive_population_execution.json", g3)
    save(sp / "sis_v3_4.g2_population_persistence.json", {
        "cycle_generation_index": 2, "next_generation_index": 3,
        "replaced_parent_idea_ids": ["a"], "carried_forward_idea_ids": ["b"],
        "generated_child_idea_ids": ["c"], "final_population_idea_ids": ["b", "c"],
    })
    save(run / "cycle_01.summary.json", {"current_generation_index": 2, "next_generation_index": 3,
         "next_generation_population_count": 2, "replaced_parent_idea_count": 1})
    save(run / "arm.summary.json", {"final_execution_report_id": "g3", "child_generation_steps": 1,
           "final_portfolio": str(run / "final.materialized.portfolio.json")})
    portfolio = {"portfolio_id": "hport", "hypotheses": [{"hypothesis_id": "h1"}, {"hypothesis_id": "h2"}]}
    save(run / "final.materialized.portfolio.json", portfolio)
    save(sp / "sis_v3_4.g3_realization_lifecycle.json", {"output_portfolio_id": "hport", "links": [
        {"idea_id": "b", "hypothesis_id": "h1", "materialization_status": "MATERIALIZED"},
        {"idea_id": "c", "hypothesis_id": "h2", "materialization_status": "MATERIALIZED"},
    ]})
    args = SimpleNamespace(v34_dir=run, p0_execution=p0, output_dir=tmp_path / "m2f",
        terminal_pair_review=None, rescue_idea_id=[], expected_final_sha256=None)
    return args


def test_full_readonly_archive_and_unreviewed_view(tmp_path):
    args = fixture(tmp_path)
    before = {str(p): m.sha_file(p) for p in tmp_path.rglob("*.json")}
    state = m.run(args)
    assert state["archived_research_ideas"] == 3
    assert state["replaced_parent_ideas_preserved_in_archive"] == 1
    assert state["terminal_display_program_buckets"] == 2
    assert not state["semantic_pair_review_completed"]
    archive = m.load(args.output_dir / "scientific_program_archive.json")
    a = next(x for x in archive["ideas"] if x["idea_id"] == "a")
    assert a["historical_only"] and a["was_replaced_as_parent"] and a["direct_child_ids"] == ["c"]
    for path, fingerprint in before.items():
        assert m.sha_file(Path(path)) == fingerprint


def test_run_fails_closed_for_stale_frozen_portfolio(tmp_path):
    args = fixture(tmp_path)
    args.expected_final_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        m.run(args)
    assert not args.output_dir.exists()


def test_wrong_transition_fails_closed(tmp_path):
    args = fixture(tmp_path)
    p = args.v34_dir / "case_root/scientific_portfolio_shadow/sis_v3_4.g2_population_persistence.json"
    data = m.load(p)
    data["generated_child_idea_ids"] = ["bad"]
    save(p, data)
    with pytest.raises(ValueError, match="generated child ledger"):
        m.run(args)


def test_linkage_is_mandatory(tmp_path):
    args = fixture(tmp_path)
    p = args.v34_dir / "case_root/scientific_portfolio_shadow/sis_v3_4.g3_realization_lifecycle.json"
    data = m.load(p)
    data["links"].pop()
    save(p, data)
    with pytest.raises(ValueError, match="linkage mismatch"):
        m.run(args)


def completed_review(args):
    nodes = m.unique_nodes(m.load(args.v34_dir / "case_root/scientific_portfolio_shadow/sis_v3_4.g3_adaptive_population_execution.json"), 3)
    r = m.review_template(nodes, ["b", "c"])
    r["reviewer"] = "scientist"
    r["status"] = "COMPLETE"
    r["pairs"][0]["relation"] = "SAME_PROGRAM"
    r["pairs"][0]["rationale"] = "Same causal backbone and test class independently reviewed"
    return r


def test_reviewed_compression_is_view_only(tmp_path):
    args = fixture(tmp_path)
    args.terminal_pair_review = save(tmp_path / "reviews.json", completed_review(args))
    state = m.run(args)
    assert state["terminal_display_program_buckets"] == 1
    compressed = m.load(args.output_dir / "terminal_program_view.json")
    assert compressed["semantic_compression_applied"]
    assert len(compressed["all_hypothesis_ids_preserved"]) == 2
    assert len(compressed["program_groups"][0]["suppressed_from_display_only_hypothesis_ids"]) == 1
    assert len(m.load(args.v34_dir / "final.materialized.portfolio.json")["hypotheses"]) == 2


def test_missing_review_fails_and_does_not_create_output(tmp_path):
    args = fixture(tmp_path)
    r = completed_review(args)
    r["pairs"] = []
    args.terminal_pair_review = save(tmp_path / "reviews.json", r)
    with pytest.raises(ValueError, match="every terminal idea pair"):
        m.run(args)
    assert not args.output_dir.exists()


def test_stale_review_fails(tmp_path):
    args = fixture(tmp_path)
    r = completed_review(args)
    r["scope_sha256"] = "f" * 64
    args.terminal_pair_review = save(tmp_path / "reviews.json", r)
    with pytest.raises(ValueError, match="scope mismatch"):
        m.run(args)


def test_nonclique_chain_is_rejected():
    nodes = {x: node(x) for x in ["a", "b", "c"]}
    rev = m.review_template(nodes, ["a", "b", "c"])
    rev.update(reviewer="expert", status="COMPLETE")
    for r in rev["pairs"]:
        r["relation"] = "DISTINCT_PROGRAM" if (r["idea_id_a"], r["idea_id_b"]) == ("a", "c") else "SAME_PROGRAM"
        r["rationale"] = "Independent assessment"
    with pytest.raises(ValueError, match="non-clique"):
        m.reviewed_groups(nodes=nodes, relevant_ids=list(nodes), reviews=rev)


def test_rescue_handoff_explicit_and_nonmutating(tmp_path):
    args = fixture(tmp_path)
    args.rescue_idea_id = ["a"]
    state = m.run(args)
    assert state["terminal_rescue_execution_prepared"]
    rescued = m.load(args.output_dir / "terminal_rescue.execution.json")
    assert rescued["g4_population_count"] == 3
    assert set(x["idea_id"] for x in rescued["g4_population_nodes"]) == {"a", "b", "c"}
    handoff = m.load(args.output_dir / "terminal_rescue.handoff.json")
    assert handoff["requires_fresh_strict_realization_and_common_verification"]
    assert m.load(args.v34_dir / "case_root/scientific_portfolio_shadow/sis_v3_4.g3_adaptive_population_execution.json")["g4_population_count"] == 2


def test_refuses_existing_output_dir(tmp_path):
    args = fixture(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        m.run(args)


def test_rescue_unknown_idea_fails(tmp_path):
    args = fixture(tmp_path)
    args.rescue_idea_id = ["unseen"]
    with pytest.raises(ValueError, match="historic"):
        m.run(args)
    assert not args.output_dir.exists()


def test_no_silent_compression_for_multiple_hypotheses_same_idea():
    nodes = {"a": node("a")}
    portfolio = {"portfolio_id": "p", "hypotheses": [{"hypothesis_id": "h1"}, {"hypothesis_id": "h2"}]}
    view = m.compressed_view(portfolio=portfolio, mapping={"h1": "a", "h2": "a"}, final_nodes=nodes, reviews=None)
    assert view["display_count"] == 2
    assert not view["semantic_compression_applied"]
    assert all(len(p["hypothesis_ids"]) == 1 for p in view["program_groups"])


def test_falsifier_review_remains_unscored(tmp_path):
    args = fixture(tmp_path)
    m.run(args)
    review = m.load(args.output_dir / "terminal_falsifier_logic_review_TEMPLATE.json")
    assert len(review["reviews"]) == 2
    assert all(row["causal_necessity_of_falsifier"] == "UNREVIEWED" for row in review["reviews"])


def test_stale_family_report_rejected(tmp_path):
    args = fixture(tmp_path)
    family_path = args.v34_dir / "case_root" / "scientific_portfolio_shadow" / "sis_v3_4.g2_scientific_program_family.json"
    save(family_path, {"assignments": [{"idea_id": "not-in-g2"}], "pair_count": 0, "pair_assessments": []})
    with pytest.raises(ValueError, match="idea IDs differ"):
        m.run(args)
    assert not args.output_dir.exists()
