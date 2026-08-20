from __future__ import annotations

import json
from pathlib import Path

from dac_her.demo_viewer import (
    build_demo_viewer,
    find_feasibility_dir,
    load_core_demo_payload,
)


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _core_fixture(tmp_path: Path) -> Path:
    run = tmp_path / "sers_run"
    hid = "hypothesis:sers"
    _dump(run / "hypothesis.context.json", {
        "context_id":"context:sers", "task_id":"task:sers",
        "question":"How does Au-Ag structure affect SERS response?",
        "corpus_id":"sers_corpus", "domain_profile_id":"sers_au_ag",
        "evidence_statements":[{"statement_id":"stmt:1","text":"A reported Au-Ag structure is associated with SERS response.","epistemic_role":"reported","claim_kind":"association","paper_ids":["Paper_S1"],"requires_verification":False}]
    })
    _dump(run / "novelty_refinement_a6.portfolio.json", {
        "portfolio_id":"portfolio:sers-final", "domain_profile_id":"sers_au_ag", "abstention_reason":None,
        "hypotheses":[{
            "hypothesis_id":hid,"title":"Nanogap-conditioned SERS response","hypothesis_statement":"Au-Ag nanogap context may condition SERS response.","hypothesis_type":"context_dependency","premise_statement_ids":["stmt:1"],"inferential_bridge":"The reported association is extended into a bounded context dependency.","predicted_observations":[{"observation_id":"pred:1","observable":"SERS intensity","expected_direction":"qualitative_change","rationale":"Tests the proposed context dependency."}],"falsification_criteria":[{"criterion_id":"fal:1","observable":"SERS intensity","falsifying_outcome":"No reproducible dependence under matched conditions."}],"assumptions":["Comparable substrates can be prepared."],"source_paper_ids":["Paper_S1"],"candidate_dependency":"none","cross_paper_synthesis":False,"evidence_profile":{"premise_count":1},"novelty_status":"not_assessed"
        }]
    })
    _dump(run / "semantic_final.review.json", {
        "overall_summary":"Bounded with one directional warning.",
        "dimensions":[
            {"dimension":"directional_specificity","verdict":"warning","rationale":"Direction remains qualitative.","hypothesis_ids":[hid],"statement_ids":[]},
            {"dimension":"premise_fidelity","verdict":"pass","rationale":"Premise is grounded.","hypothesis_ids":[hid],"statement_ids":["stmt:1"]}
        ]
    })
    _dump(run / "external_novelty_a52.report.json", {
        "status_counts":{"PLAUSIBLY_NOVEL":1},
        "cards":[{"hypothesis_id":hid,"status":"PLAUSIBLY_NOVEL","claim_reviews":[],"reason_codes":["no_direct_match_found"],"interpretation":"No direct match found in the bounded search."}]
    })
    _dump(run / "novelty_refinement_a6.report.json", {
        "attempts":[{"original_hypothesis_id":hid,"final_hypothesis_id":hid,"decision":"kept_original","final_external_status":"PLAUSIBLY_NOVEL","reason_codes":["original_retained"],"interpretation":"Original bounded formulation retained."}]
    })
    _dump(run / "e2e_runner.manifest.json", {"domain_profile_id":"sers_au_ag","feasibility_status":"not_supported_for_domain"})
    return run


def test_find_feasibility_dir_returns_none_for_core_only_run(tmp_path: Path):
    run = _core_fixture(tmp_path)
    assert find_feasibility_dir(run) is None


def test_load_core_payload_preserves_domain_and_core_review_artifacts(tmp_path: Path):
    run = _core_fixture(tmp_path)
    payload = load_core_demo_payload(run)
    assert payload["viewer_mode"] == "core"
    assert payload["domain_profile_id"] == "sers_au_ag"
    assert payload["feasibility_available"] is False
    assert payload["paper_ids"] == ["Paper_S1"]
    row = payload["hypotheses"][0]
    assert row["hypothesis"]["semantic_gate_status"] == "eligible_with_warnings"
    assert row["hypothesis"]["novelty_status"] == "PLAUSIBLY_NOVEL"
    assert row["novelty"]["status"] == "PLAUSIBLY_NOVEL"
    assert row["refinement"]["decision"] == "kept_original"


def test_build_core_viewer_without_feasibility_artifacts(tmp_path: Path):
    run = _core_fixture(tmp_path)
    output = run / "demo" / "index.html"
    result = build_demo_viewer(run_dir=run, output=output, title="SERS discovery viewer")
    rendered = result.read_text(encoding="utf-8")
    assert "Core scientific pipeline view" in rendered
    assert "sers_au_ag" in rendered
    assert "Nanogap-conditioned SERS response" in rendered
    assert "PLAUSIBLY_NOVEL" in rendered
    assert "Feasibility not supported for this domain profile" in rendered
    assert "<script src=" not in rendered
    assert "<link rel=" not in rendered
