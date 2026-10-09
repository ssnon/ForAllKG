"""M4-C5: counterfactual, *human-specified* operational ResearchIdea kernel proposals.

Consumes frozen synthetic M4-C4 results and authentic M3-C1 parent kernels.
This is NOT empirical learning, autonomous hypothesis discovery, SIS generation,
or authorization to create ResearchIdeaNodes. No I/O, LLM, or network.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from pipeline_core.discovery.research_idea_contracts import ResearchIdeaKernel
from pipeline_core.discovery.research_idea_semantics import compare_idea_kernels

CASE = "QA_B03_B06_PROXY_COVARIANCE"
B03 = "research_idea:46af2e8dcdc0e3a1802a"
B06 = "research_idea:d26c4653802153c0d499"
PARENTS = frozenset((B03, B06))
FIXTURES = frozenset(("COVARIANCE_EFFECT_INJECTED", "NO_COVARIANCE_EFFECT", "EXACT_COLLINEARITY"))


def require(test: bool, reason: str) -> None:
    if not test:
        raise ValueError("M4C5_INTEGRITY_FAILURE: " + reason)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _parent_kernels(traces: Mapping[str, Any]) -> dict[str, ResearchIdeaKernel]:
    require(traces.get("status") == "SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW", "M3-C1 status")
    require(traces.get("authoritative_science_judgment") is False and
            traces.get("original_data_mutated") is False, "M3-C1 authority/mutation")
    rows = [r for r in traces.get("rows", []) if r.get("blind_id") in ("B03", "B06")]
    require(len(rows) == 2 and {r["blind_id"] for r in rows} == {"B03", "B06"}, "B03/B06 trajectory count")
    result: dict[str, ResearchIdeaKernel] = {}
    for row in rows:
        idea_id = B03 if row["blind_id"] == "B03" else B06
        require(row.get("terminal_idea_id") == idea_id and
                row.get("lineage_validation") == "PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY", "terminal lineage")
        steps = row.get("steps_earliest_to_latest", [])
        require(bool(steps) and steps[-1].get("idea_id") == idea_id, "last step terminal identity")
        kernel = ResearchIdeaKernel.model_validate(steps[-1].get("kernel"))
        result[idea_id] = kernel
    return result


def _inspect_fixtures(fixtures: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(set(fixtures) == FIXTURES, "fixture set unexpected")
    output = {}
    for name in sorted(FIXTURES):
        fixture = fixtures[name]
        require(fixture.get("effect_claim_allowed") is False, name + " scientific authority")
        if name == "EXACT_COLLINEARITY":
            require(fixture.get("status") == "UNIDENTIFIABLE_DESIGN" and
                    fixture.get("reason") == "EXACT_COVARIANCE_ACTIVE_ORIENTATION_COLLINEARITY", "collinearity fixture")
            output[name] = {"synthetic_delta_mse_ONLY": None, "condition": "NOT_IDENTIFIABLE",
                            "revision": "ABSTAIN", "inference": "No distinguishable incremental predictor"}
            continue
        require(fixture.get("status") == "SYNTHETIC_PREDICTION_COMPARISON_ONLY", "fixture scientific status")
        delta = fixture.get("delta_mse")
        require(type(delta) in (float, int) and math.isfinite(delta), "invalid delta_MSE")
        require(fixture.get("fold_count") == 4 and len(fixture.get("folds", [])) == 4,
                "four out-of-group folds required")
        require(fixture.get("measurements_independent_of_SERS_certified") is False,
                "unexpected independent measurement certification")
        require(fixture.get("n_observations") == 128 and fixture.get("group_count") == 16,
                "unexpected synthetic cohort")
        require(fixture.get("baseline_mse") is not None and fixture.get("augmented_mse") is not None and
                abs(fixture["baseline_mse"] - fixture["augmented_mse"] - delta) < 1e-9, "delta inconsistent with losses")
        folds = fixture["folds"]
        require(all(type(f.get("delta_mse")) in (float, int) and math.isfinite(f["delta_mse"])
                    and f.get("train_group_count") == 12 and f.get("test_group_count") == 4
                    and f.get("test_n") == 32 for f in folds), "invalid held-out fold")
        if name == "COVARIANCE_EFFECT_INJECTED":
            require(delta > 0.05 and all(f["delta_mse"] > 0 for f in folds), "positive fixture inconsistent")
            condition = "SYNTHETIC_INCREMENT_DETECTED"
        else:
            require(abs(delta) < 0.01 and fixture.get("positive_delta_all_folds") is False,
                    "negative fixture inconsistent")
            condition = "SYNTHETIC_NO_INCREMENT_DETECTED"
        output[name] = {"synthetic_delta_mse_ONLY": float(delta), "condition": condition,
                        "revision": "CONDITIONAL_DRAFT", "inference": "Counterfactual study design only"}
    return output


def _reframe(parent: ResearchIdeaKernel, design: Mapping[str, Any], kind: str) -> tuple[ResearchIdeaKernel, dict]:
    """Explicit, human-specified scientific protocol. Not autonomous idea synthesis."""
    estimand = design["operational_estimand"]
    baseline = design["baseline_model"]
    augmented = design["augmented_model"]
    witnesses = list(design["required_measurement_witnesses"])
    controls = list(design["required_controls"])
    question = str(design["scientific_question"])
    require("delta_MSE" in estimand and "covariance" in augmented.lower(), "operational estimand missing")
    require(len(witnesses) >= 5 and len(controls) >= 5, "independent measurement/controls missing")
    if kind == "COVARIANCE_EFFECT_INJECTED":
        scoped_question = (question + " If independent group-held-out prediction improves, in which "
                           "matched field-heterogeneity regimes does that added prediction persist beyond the B03 proxy?")
        prediction = ("In a future independently paired experiment, the predeclared augmented model "
                      "may show positive held-out delta_MSE beyond the B03 baseline in specified regimes; "
                      "this is a *testable proposal*, not an observed SERS effect.")
        failure = ("If independently measured covariance does not improve held-out error beyond "
                   "uncertainty and model-selection sensitivity in the specified regime, the claimed "
                   "conditional predictive increment is unsupported there; B06 remains unfalsified globally.")
        intent = parent.canonical_intent + " Operational follow-up: test conditional incremental predictivity under independent measurement."
        disposition = "OPERATIONAL_INCREMENT_RESEARCH_QUESTION"
    elif kind == "NO_COVARIANCE_EFFECT":
        scoped_question = ("Under what independently characterized field-heterogeneity, molecular-access or "
                           "registration regimes would orientation–field covariance add held-out predictive "
                           "information beyond the B03 active-population proxy, if it fails in one matched regime?")
        prediction = ("In a future sufficiently sensitive, independently registered experiment, "
                      "held-out delta_MSE may be indistinguishable from zero in a specified restricted regime "
                      "but differs under deliberate variation in field/access heterogeneity; not a general rejection.")
        failure = ("Failure to observe the predicted regime-specific change, after confirming variation, "
                   "independent measurement and uncertainty, challenges that *regime-bound* proposal only.")
        intent = parent.canonical_intent + " Operational follow-up: locate boundaries of incremental predictivity without discarding B06."
        disposition = "REGIME_BOUNDARY_RESEARCH_QUESTION"
    else:
        raise ValueError("M4C5_INTEGRITY_FAILURE: cannot reframe unidentified fixture")
    scope = [*parent.scope_commitments,
             "Conditional synthetic-design exercise only; no empirical observation has been consumed",
             "Unit of prospective validation: matched sample/condition group; no group shared between train/test",
             "Independent measurement plan: " + "; ".join(witnesses),
             "Matched controls: " + "; ".join(controls)]
    contrast = [*parent.contrastive_commitments,
                "Operational comparator: " + baseline + " versus " + augmented,
                "Operational estimand: " + estimand,
                "Prospective differential prediction: " + prediction,
                "Regime-scoped non-confirmation condition: " + failure]
    kernel = ResearchIdeaKernel(
        canonical_intent=intent,
        core_scientific_commitments=list(parent.core_scientific_commitments),
        scope_commitments=scope,
        contrastive_commitments=contrast,
        question_commitment=scoped_question,
    )
    new_contrasts = [x for x in kernel.contrastive_commitments if x not in parent.contrastive_commitments]
    require(len(new_contrasts) == 4 and "Operational estimand:" in new_contrasts[1], "not an operational contrast")
    return kernel, {"question":scoped_question, "prediction":prediction, "scoped_non_confirmation":failure,
                    "estimand":estimand, "baseline":baseline, "augmented":augmented,
                    "required_measurement_witnesses":witnesses, "required_controls":controls,
                    "revision_dimension":disposition,
                    "distinguishing_added_contrastive_commitments":new_contrasts}


def execute_shadow(*, m4c4: Mapping[str, Any], trajectories: Mapping[str, Any],
                   sha_m4c4: str, sha_traces: str) -> dict[str, Any]:
    require(m4c4.get("schema_version") == "m4c4-conditional-predictive-increment-shadow-v1" and
            m4c4.get("status") == "SYNTHETIC_DISCRIMINATION_DEMO_ONLY_NO_SCIENTIFIC_REVISION", "M4-C4 identity/status")
    require(m4c4.get("case_id") == CASE and m4c4.get("source_files_mutated") is False and
            m4c4.get("research_idea_nodes_created") is False and
            m4c4.get("production_selection_changed") is False and
            m4c4.get("empirical_data_consumed") is False and
            m4c4.get("scientific_improvement_certified") is False and
            m4c4.get("scientific_truth_or_falsification_authority") is False and
            m4c4.get("llm_or_network_calls") == 0, "unexpected input promotion")
    require(set(m4c4.get("design", {}).get("parent_idea_ids_preserved_separately", [])) == PARENTS,
            "parent set altered")
    require(m4c4["design"].get("comparison_type") == "NESTED_NONEXCLUSIVE_PREDICTIVE_INCREMENT" and
            m4c4["design"].get("causal_claim") is False and
            m4c4["design"].get("mechanisms_mutually_exclusive") is False, "comparison semantics changed")
    lineage = m4c4.get("lineage", {})
    require(lineage.get("b03_idea_id") == B03 and lineage.get("b06_idea_id") == B06,
            "M4-C4 source lineage changed")
    require(m4c4.get("source_sha256", {}).get("m4a1") ==
            "cabf847646a23eb292aa9cc1deb5097a07cf4c927605d9f6f4631da5e29732f7", "M4-A1 anchor mismatch")
    parents = _parent_kernels(trajectories)
    findings = _inspect_fixtures(m4c4.get("synthetic_fixtures", {}))
    source = parents[B06]
    rows = []
    for name in ("COVARIANCE_EFFECT_INJECTED", "NO_COVARIANCE_EFFECT", "EXACT_COLLINEARITY"):
        outcome = findings[name]
        if outcome["revision"] == "ABSTAIN":
            rows.append({"synthetic_fixture":name,"fixture_result":outcome,
                         "draft_id_NOT_RESEARCH_IDEA_ID":None,"kernel_draft_NOT_CREATED":None,
                         "status":"ABSTAIN_NON_IDENTIFIABLE", "scientific_review_required":True,
                         "comparison_parent_idea_id_PRESERVED":B03,"seed_parent_idea_id_PRESERVED":B06})
            continue
        kernel, protocol = _reframe(source, m4c4["design"], name)
        diagnostic, facets = compare_idea_kernels(source, kernel)
        payload = kernel.model_dump(mode="json")
        rows.append({"synthetic_fixture":name,"fixture_result":outcome,
                     "draft_id_NOT_RESEARCH_IDEA_ID":"m4c5_operational_draft:" + stable_hash([name,B06,payload])[:24],
                     "kernel_draft_NOT_CREATED":payload, "kernel_sha256":stable_hash(payload),
                     "status":"HUMAN_SPECIFIED_OPERATIONAL_REFRAME_DRAFT_ONLY",
                     "protocol":protocol, "core_original_preserved":kernel.core_scientific_commitments == source.core_scientific_commitments,
                     "sis_identity_diagnostic_ONLY":diagnostic,
                     "sis_facet_diagnostics_ONLY":[r.model_dump(mode="json") for r in facets],
                     "scientific_review_required":True,
                     "comparison_parent_idea_id_PRESERVED":B03,"seed_parent_idea_id_PRESERVED":B06,
                     "semantic_or_scientific_improvement_certified":False})
    require(sum(r["kernel_draft_NOT_CREATED"] is not None for r in rows) == 2,
            "unexpected conditional proposal count")
    require(all(r.get("core_original_preserved") is not False for r in rows), "parent core lost")
    return {"schema_version":"m4c5-counterfactual-operational-revision-shadow-v1",
            "status":"COUNTERFACTUAL_OPERATIONAL_DRAFTS_ONLY_NO_AUTONOMOUS_SCIENTIFIC_LEARNING",
            "source_sha256":{"m4c4":sha_m4c4,"m3c1":sha_traces},
            "source_case":CASE,"original_parent_idea_ids_PRESERVED_SEPARATELY":[B03,B06],
            "seed_parent_idea_id":B06,"comparison_parent_idea_id":B03,
            "candidate_generation_method":"HUMAN_SPECIFIED_OPERATIONAL_PROTOCOL_NOT_LLM_OR_SIS_GENERATION",
            "new_research_idea_nodes_created":False,"scientific_semantic_improvement_certified":False,
            "empirical_measurements_consumed":False,"llm_or_network_calls":0,
            "production_selection_changed":False,"original_idea_nodes_mutated":False,
            "rows":rows, "kernel_drafts":2,"abstentions":1}


def render_report(result: Mapping[str, Any]) -> str:
    lines = ["# M4-C5 — Counterfactual Operational ResearchIdea Revision (PRIVATE)", "",
             "Status: `" + result["status"] + "`", "",
             "Two explicitly human-specified ResearchIdeaKernel drafts were produced from synthetic M4-C4 fixtures.",
             "**No ResearchIdeaNode was generated; no SIS reproduction, empirical learning or scientific superiority was certified.**", "",
             "| Synthetic fixture | Revision response | Operational change | SIS identity diagnostic |",
             "|---|---|---|---|"]
    for row in result["rows"]:
        details = row.get("protocol") or {}
        lines.append("| " + row["synthetic_fixture"] + " | " + row["status"] + " | " +
                     details.get("revision_dimension", "ABSTAIN") + " | " +
                     str(row.get("sis_identity_diagnostic_ONLY", "N/A")) + " |")
    lines.extend(["", "**Operational delta:** held-out group-level ΔMSE, added covariance predictor, independent orientation/field measurements, and explicitly bounded failure criteria.",
                  "**Scientific limitation:** the generator is hand-specified and synthetic. SIS similarity is a semantic diagnostic, not proof of a new idea.",
                  "**Next:** expert evaluation of whether the conditional tests are mechanistically informative, then opt-in isolated SIS shadow reproduction with frozen costs and parent protection.", ""])
    return "\n".join(lines)
