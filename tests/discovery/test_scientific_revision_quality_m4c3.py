"""M4-C3 synthetic-only source and adverse-input regression tests."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from pipeline_core.discovery.scientific_revision_quality_m4c3 import (
    evaluate_quality, render_report, sha,
)
from scripts.discovery.audit_scientific_revision_quality_m4c3 import run


def bytes_json(o):
    return (json.dumps(o, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def dig(o):
    return hashlib.sha256(bytes_json(o)).hexdigest()


def fixture(dual=False):
    hyps = [{"ref": "B03", "hypothesis_id": "h:a", "terminal_idea_id": "research_idea:a"}]
    if dual:
        hyps.append({"ref": "B06", "hypothesis_id": "h:b", "terminal_idea_id": "research_idea:b"})
    pair = "ADJACENT_NOT_EXCLUSIVE" if dual else "COMPETING_PARTIAL"
    case = {"case_id": "QA_CASE", "hypothesis_refs": [h["ref"] for h in hyps],
            "input_hypotheses": hyps,
            "proposed_experiment_UNREVIEWED": {"pair_relationship": pair}}
    a1 = {"schema_version": "m4a1-scientific-confrontation-v1",
          "status": "DRAFT_CONFRONTATIONS_NEED_REVIEW", "case_count": 1, "cases": [case]}
    parent_kernels = {}
    traces = []
    for h in hyps:
        r = h["ref"]
        k = {"schema_version": "research-idea-kernel-v1",
             "canonical_intent": "Test the mechanism " + r,
             "core_scientific_commitments": ["Spectral intensity depends on orientation " + r],
             "scope_commitments": ["Au/Ag samples matched " + r],
             "contrastive_commitments": ["EM field not equal to orientation " + r],
             "question_commitment": "What determines spectral intensity " + r + "?"}
        parent_kernels[h["terminal_idea_id"]] = k
        traces.append({"blind_id": r, "hypothesis_id": h["hypothesis_id"],
                       "terminal_idea_id": h["terminal_idea_id"],
                       "lineage_validation": "PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY",
                       "steps_earliest_to_latest": [{"idea_id": h["terminal_idea_id"], "kernel": k}]})
    m3c1 = {"status": "SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW",
             "original_data_mutated": False, "authoritative_science_judgment": False,
             "rows": traces}
    ahash, mhash = dig(a1), dig(m3c1)
    inputsha = {"m4a1": ahash, "m3c1_trajectories": mhash,
                "m4c0": "1" * 64, "m4c1": "2" * 64}
    ids = [h["terminal_idea_id"] for h in hyps]
    rows = []
    for scenario in ["A_ONLY_IN_SCOPE", "B_ONLY_IN_SCOPE", "BOTH_COMPATIBLE",
                     "NEITHER_COMPATIBLE", "NOT_IDENTIFIABLE"]:
        parent = copy.deepcopy(parent_kernels[ids[0]])
        d = None if scenario == "NOT_IDENTIFIABLE" else copy.deepcopy(parent)
        if d is not None:
            d["canonical_intent"] += " Conditional revision question " + scenario
            d["question_commitment"] = "What if " + scenario + "?"
            d["scope_commitments"].append("Hypothetical scenario only")
            d["contrastive_commitments"].extend(["Hypothetical pattern A", "Hypothetical pattern B"])
        draft_id = ("m4c2_kernel_draft:" + sha([mhash, inputsha["m4c0"], "QA_CASE", scenario, ids, d])[:24]) if d else None
        rows.append({
            "case_id": "QA_CASE", "synthetic_scenario": scenario,
            "scenario_kind": "SYNTHETIC_POLICY_FIXTURE_NOT_OBSERVED",
            "pair_relationship": pair, "source_hypothesis_refs": case["hypothesis_refs"],
            "original_terminal_idea_ids_PRESERVED": ids,
            "seed_parent_id": ids[0],
            "comparison_parent_ids_PRESERVED_SEPARATELY": ids[1:],
            "multi_parent_composition_NOT_CLAIMED": True,
            "parent_kernel_sha256_by_id": {i: sha(parent_kernels[i]) for i in ids},
            "original_parent_scientific_core_by_id": {i: parent_kernels[i]["core_scientific_commitments"] for i in ids},
            "draft_id_NOT_RESEARCH_IDEA_ID": draft_id,
            "draft_scientific_kernel_NOT_CREATED": d,
            "draft_kernel_sha256": sha(d) if d else None,
            "revision_disposition": "DRAFT_KERNEL_FOR_EXPERT_REVIEW_ONLY" if d else "NO_REVISION_UNIDENTIFIABLE",
            "preferred_policy_actions_NOT_EXECUTED": ["REQUEST_TEST"],
            "scientific_cautions": ["NONEXCLUSIVE_MECHANISMS_NO_AUTOMATIC_EXCLUSION"] if dual else ["NO_GLOBAL_REJECTION"],
            "requires_human_science_review": True, "research_idea_node_created": False,
            "original_research_ideas_modified": False, "scientific_claim_adjudicated": False,
            "real_measurement_consumed": False,
        })
    c2 = {"schema_version": "m4c2-synthetic-kernel-revision-replay-v1",
          "status": "SYNTHETIC_KERNEL_DRAFTS_ONLY_NOT_EMPIRICAL_LEARNING",
          "input_sha256": inputsha, "source_case_count": 1,
          "scenario_count": 5, "draft_kernel_count": 4, "no_revision_count": 1,
          "rows": rows, "new_llm_or_network_calls": 0}
    for name in ("empirical_scientific_learning_demonstrated", "hypothesis_cards_modified",
                 "novelty_certification_authority", "original_research_ideas_modified",
                 "production_selection_changed", "real_measurements_consumed",
                 "research_idea_nodes_created", "scientific_truth_or_falsification_authority"):
        c2[name] = False
    c2["report_id"] = "m4c2_kernel_revision_replay:" + sha(c2)[:20]
    return {"c2": c2, "a1": a1, "trajectories": m3c1,
            "sha_a1": ahash, "sha_traces": mhash, "sha_c2": dig(c2)}


def audit(f):
    return evaluate_quality(**f)


def update_checksum(f):
    f["c2"]["report_id"] = "m4c2_kernel_revision_replay:" + sha({k: v for k,v in f["c2"].items() if k != "report_id"})[:20]
    f["sha_c2"] = dig(f["c2"])


def test_single_parent_expected_classification():
    r = audit(fixture())
    assert r["status"] == "INTEGRITY_PASS_SCIENTIFIC_DELTA_UNPROVEN"
    assert r["review_required_count"] == 4
    assert r["substantive_revisions_scientifically_confirmed"] == 0
    assert r["no_revision_count"] == 1
    assert {x["scientific_delta_state"] for x in r["rows"]} == {
        "NO_REVISION_REQUIRED", "CONDITIONAL_REFRAME_ONLY_SUBSTANTIVE_DELTA_UNPROVEN"}
    assert not r["empirical_learning_certified"]
    assert "scientific" in render_report(r)


def test_separate_parent_programs_protected():
    r = audit(fixture(dual=True))
    assert all("NONEXCLUSIVE_PARENT_PROGRAMS_MUST_REMAIN_SEPARATE" in row["failure_reasons"]
               for row in r["rows"] if row["draft_id"])


def test_input_not_mutated():
    f=fixture(dual=True);copy_f=copy.deepcopy(f)
    audit(f)
    assert f==copy_f


def test_idempotent():
    f=fixture()
    assert audit(f)==audit(f)


@pytest.mark.parametrize("change", [
    lambda f: f["c2"].update(status="EMPIRICALLY_CONFIRMED"),
    lambda f: f["c2"].update(real_measurements_consumed=True),
    lambda f: f["c2"].update(production_selection_changed=True),
    lambda f: f["c2"].update(research_idea_nodes_created=True),
    lambda f: f["c2"]["rows"][0].update(scenario_kind="OBSERVED"),
    lambda f: f["c2"]["rows"][0].update(research_idea_node_created=True),
    lambda f: f["c2"]["rows"][0].update(scientific_claim_adjudicated=True),
    lambda f: f["c2"]["rows"][0].update(seed_parent_id="research_idea:fake"),
    lambda f: f["c2"]["rows"][0].update(original_terminal_idea_ids_PRESERVED=["research_idea:fake"]),
    lambda f: f["c2"]["rows"][0].update(pair_relationship="ADJACENT_NOT_EXCLUSIVE"),
    lambda f: f["c2"]["rows"][0].update(comparison_parent_ids_PRESERVED_SEPARATELY=["research_idea:fake"]),
    lambda f: f["c2"]["rows"][0].update(draft_kernel_sha256="0"*64),
    lambda f: f["c2"]["rows"][0].update(draft_id_NOT_RESEARCH_IDEA_ID="m4c2_kernel_draft:fake"),
    lambda f: f["c2"]["rows"][0]["draft_scientific_kernel_NOT_CREATED"].update(core_scientific_commitments=["unrelated"]),
    lambda f: f["c2"]["rows"][0]["draft_scientific_kernel_NOT_CREATED"].update(scope_commitments=[]),
    lambda f: f["c2"]["rows"][0]["draft_scientific_kernel_NOT_CREATED"].update(question_commitment="changed"),
    lambda f: f["c2"]["rows"][4].update(draft_scientific_kernel_NOT_CREATED={"x":1}),
    lambda f: f["c2"]["rows"].pop(),
    lambda f: f["c2"]["rows"].append(copy.deepcopy(f["c2"]["rows"][0])),
    lambda f: f["c2"].update(source_case_count=2),
    lambda f: f["c2"]["input_sha256"].update(m4a1="3"*64),
    lambda f: f["c2"]["input_sha256"].update(m3c1_trajectories="3"*64),
    lambda f: f["trajectories"]["rows"][0].update(terminal_idea_id="research_idea:fake"),
    lambda f: f["trajectories"].update(authoritative_science_judgment=True),
    lambda f: f["a1"].update(status="AUTHORITATIVE"),
])
def test_fail_closed_on_mutations(change):
    f=fixture(); change(f); update_checksum(f)
    with pytest.raises((ValueError, TypeError, KeyError), match="M4C3_INTEGRITY_FAILURE"):
        audit(f)


def test_wrong_report_checksum_fails():
    f=fixture();f["c2"]["report_id"]="fake"
    with pytest.raises(ValueError, match="checksum"):
        audit(f)


def test_nonexclusive_warning_cannot_be_removed():
    f=fixture(dual=True);f["c2"]["rows"][0]["scientific_cautions"]=[];update_checksum(f)
    with pytest.raises(ValueError, match="nonexclusive caution"):
        audit(f)


def write_input(tmp_path, f):
    paths = {}
    for name in ("c2", "a1", "trajectories"):
        p = tmp_path / (name+".json")
        p.write_bytes(bytes_json(f[name]))
        paths[name] = p
    return paths


def test_cli_safe_success(tmp_path):
    f=fixture(dual=True);paths=write_input(tmp_path,f);dest=tmp_path/"out"
    r=run(m4c2=paths["c2"],m4a1=paths["a1"],m3c1=paths["trajectories"],
          output_dir=dest,expected_m3c1_sha256=f["sha_traces"],expected_m4a1_sha256=f["sha_a1"])
    assert r["review_required_count"]==4
    assert (dest/"M4C3_REPORT_PRIVATE.md").exists()
    template=(dest/"M4C3_EXPERT_REVIEW_TEMPLATE_PRIVATE.csv").read_text()
    assert template.count("UNREVIEWED")==4


def test_cli_refuses_wrong_sha(tmp_path):
    f=fixture();paths=write_input(tmp_path,f);dest=tmp_path/"out"
    with pytest.raises(ValueError,match="source SHA pin mismatch"):
        run(m4c2=paths["c2"],m4a1=paths["a1"],m3c1=paths["trajectories"],
            output_dir=dest,expected_m3c1_sha256="x"*64,expected_m4a1_sha256=f["sha_a1"])
    assert not dest.exists()


def test_cli_refuses_existing_output(tmp_path):
    f=fixture();paths=write_input(tmp_path,f);dest=tmp_path/"out";dest.mkdir()
    (dest/"keep").write_text("KEEP")
    with pytest.raises(FileExistsError):
        run(m4c2=paths["c2"],m4a1=paths["a1"],m3c1=paths["trajectories"],
            output_dir=dest,expected_m3c1_sha256=f["sha_traces"],expected_m4a1_sha256=f["sha_a1"])
    assert (dest/"keep").read_text()=="KEEP"


def test_cli_refuses_output_in_repo(tmp_path):
    f=fixture();paths=write_input(tmp_path,f)
    root=__import__("pathlib").Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError,match="outside repository"):
        run(m4c2=paths["c2"],m4a1=paths["a1"],m3c1=paths["trajectories"],
            output_dir=root/"private",expected_m3c1_sha256=f["sha_traces"],expected_m4a1_sha256=f["sha_a1"])
