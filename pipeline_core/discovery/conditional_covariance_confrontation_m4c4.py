"""M4-C4: operational B03/B06 confrontation on *synthetic* group-held-out data.

This is an algorithm/estimand demonstration, not physical/empirical SERS evidence.
No models, network, archive, ResearchIdea or production API are invoked.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping

CASE = "QA_B03_B06_PROXY_COVARIANCE"
PARENTS = frozenset({
    "research_idea:46af2e8dcdc0e3a1802a",  # B03
    "research_idea:d26c4653802153c0d499",  # B06
})
FEATURES = ("active_orientation", "mean_orientation", "mean_enhancement", "substrate_ag")
COVARIANCE = "orientation_field_covariance"


def fail(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("M4C4_INTEGRITY_FAILURE: " + reason)


def _stable(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _origins(a1: Mapping[str, Any], c2: Mapping[str, Any], c3: Mapping[str, Any],
             sha_a1: str, sha_c2: str) -> dict[str, Any]:
    fail(a1.get("status") == "DRAFT_CONFRONTATIONS_NEED_REVIEW", "M4-A1 status")
    fail(c2.get("status") == "SYNTHETIC_KERNEL_DRAFTS_ONLY_NOT_EMPIRICAL_LEARNING", "M4-C2 status")
    fail(c3.get("status") == "INTEGRITY_PASS_SCIENTIFIC_DELTA_UNPROVEN", "M4-C3 status")
    fail(c2.get("input_sha256", {}).get("m4a1") == sha_a1, "M4-C2/M4-A1 lineage")
    fail(c3.get("input_sha256", {}).get("m4a1") == sha_a1, "M4-C3/M4-A1 lineage")
    fail(c3.get("input_sha256", {}).get("m4c2") == sha_c2, "M4-C3/M4-C2 lineage")
    fail(c3.get("substantive_revisions_scientifically_confirmed") == 0, "unexpected M4-C3 science promotion")
    case = [r for r in a1.get("cases", []) if r.get("case_id") == CASE]
    fail(len(case) == 1, "B03/B06 source case missing or duplicated")
    origin = case[0]
    fail(origin.get("status") == "REVIEW_REQUIRED_NOT_SCIENCE_CERTIFIED", "source case science status")
    fail(origin.get("proposed_experiment_UNREVIEWED", {}).get("pair_relationship") == "ADJACENT_NOT_EXCLUSIVE", "B03/B06 must remain nonexclusive")
    refs = {x.get("ref"): x.get("terminal_idea_id") for x in origin.get("input_hypotheses", [])}
    fail(set(refs) == {"B03", "B06"} and frozenset(refs.values()) == PARENTS, "source parent identity mismatch")
    drafts = [r for r in c2.get("rows", []) if r.get("case_id") == CASE]
    reviews = [r for r in c3.get("rows", []) if r.get("case_id") == CASE]
    fail(len(drafts) == len(reviews) == 5, "B03/B06 synthetic cohort count")
    scenarios = {"A_ONLY_IN_SCOPE", "B_ONLY_IN_SCOPE", "BOTH_COMPATIBLE",
                 "NEITHER_COMPATIBLE", "NOT_IDENTIFIABLE"}
    fail({r.get("synthetic_scenario") for r in drafts} == scenarios and
         {r.get("scenario") for r in reviews} == scenarios,
         "B03/B06 scenario identity mismatch")
    fail(all(r.get("scenario_kind") == "SYNTHETIC_POLICY_FIXTURE_NOT_OBSERVED"
             and r.get("research_idea_node_created") is False
             for r in drafts), "synthetic status or node creation inconsistent")
    fail(all(r.get("structural_integrity") == "PASS" and
             r.get("independent_science_confirmed") is False
             for r in reviews), "M4-C3 integrity or scientific authorization")
    fail(sum(r.get("draft_scientific_kernel_NOT_CREATED") is not None for r in drafts) == 4, "four draft kernels required")
    fail(all(set(r.get("original_terminal_idea_ids_PRESERVED", [])) == PARENTS for r in drafts), "parent preservation")
    fail(all(r.get("scientific_delta_state") != "SUBSTANTIVE_IMPROVEMENT_CONFIRMED" for r in reviews), "unjustified science promotion")
    return {"b03_idea_id": refs["B03"], "b06_idea_id": refs["B06"],
            "m4c2_synthetic_scenarios": len(drafts), "m4c3_review_count": len(reviews)}


def fixture(*, effect: float, collinear: bool = False, leakage: bool = False,
            groups: int = 16, per_group: int = 8) -> list[dict[str, Any]]:
    """Deterministic fixed DGP; effect is *injected*, not discovered."""
    fail(groups >= 8 and per_group >= 5, "fixture design too small")
    rows: list[dict[str, Any]] = []
    for g in range(groups):
        ag = float(g % 2)
        for i in range(per_group):
            t = g * per_group + i
            active = 0.48 * math.sin(t * 0.61) + 0.16 * math.cos(g * 0.34) + 0.15 * (i - 3.5)
            mean_ori = 0.35 * math.cos(t * 0.29) + 0.24 * active
            enhancement = 0.33 * math.sin(t * 0.17 + g * .31) + 0.12 * ag
            extra = 0.64 * math.sin(t * 0.97 + g * .19) + 0.30 * math.cos(t * .45)
            cov = 2.0 * active if collinear else 0.4 * active + extra
            noise = .04 * math.cos(t * 2.1) + .025 * math.sin(t * 1.67)
            y = 1.1 * active + .45 * mean_ori + .26 * enhancement + .18 * ag + effect * cov + noise
            rows.append({"group_id": f"matched-group-{g:02d}",
                         "target_band_ratio": y,
                         "features": {"active_orientation": active,
                                      "mean_orientation": mean_ori,
                                      "mean_enhancement": enhancement,
                                      "substrate_ag": ag,
                                      COVARIANCE: y if leakage else cov},
                         "measurement_source": ("DERIVED_FROM_TARGET" if leakage else "SYNTHETIC_INDEPENDENT_CHANNELS")})
    return rows


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Small ridge-normal-equation solver with deterministic pivoting."""
    size = len(rhs)
    a = [list(matrix[i]) + [rhs[i]] for i in range(size)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda i: abs(a[i][col]))
        fail(abs(a[pivot][col]) > 1e-12, "singular design")
        a[col], a[pivot] = a[pivot], a[col]
        d = a[col][col]
        for j in range(col, size + 1):
            a[col][j] /= d
        for row in range(size):
            if row == col:
                continue
            scale = a[row][col]
            for j in range(col, size + 1):
                a[row][j] -= scale * a[col][j]
    return [a[i][-1] for i in range(size)]


def _train_predict(train: list[dict[str, Any]], test: list[dict[str, Any]],
                   keys: tuple[str, ...]) -> list[float]:
    means = [sum(x["features"][k] for x in train) / len(train) for k in keys]
    stds = [math.sqrt(sum((x["features"][k]-mu)**2 for x in train) / len(train))
            for k, mu in zip(keys, means)]
    stds = [s if s > 1e-10 else 1. for s in stds]
    def design(x: dict[str, Any]) -> list[float]:
        return [1.] + [(x["features"][k] - mu) / s for k, mu, s in zip(keys, means, stds)]
    n = 1 + len(keys)
    gram = [[0.] * n for _ in range(n)]
    cross = [0.] * n
    for row in train:
        x = design(row)
        for i in range(n):
            cross[i] += x[i] * row["target_band_ratio"]
            for j in range(n):
                gram[i][j] += x[i] * x[j]
    for i in range(1, n):
        gram[i][i] += 1e-6
    weights = _solve(gram, cross)
    return [sum(v * w for v,w in zip(design(x), weights)) for x in test]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fail(len(rows) >= 40, "insufficient synthetic fixture")
    fail(all(isinstance(r.get("group_id"), str) and r["group_id"].strip() for r in rows), "missing matched group")
    groups = sorted({r["group_id"] for r in rows})
    fail(len(groups) >= 8, "insufficient distinct matched groups")
    for row in rows:
        fail(row.get("measurement_source") == "SYNTHETIC_INDEPENDENT_CHANNELS", "leaky or unapproved data source")
        f = row.get("features")
        fail(isinstance(f, dict) and set(FEATURES + (COVARIANCE,)).issubset(f), "missing features")
        vals = [row.get("target_band_ratio")] + [f[k] for k in FEATURES + (COVARIANCE,)]
        fail(all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in vals), "nonfinite/non-numeric input")
    fail(len({(r["group_id"], r["features"][COVARIANCE]) for r in rows}) > len(groups), "insufficient covariance variation")
    x = [r["features"][COVARIANCE] for r in rows]
    # Exact collinearity with baseline controls is not identifiable; this check
    # catches the targeted pathological synthetic fixture, not general dependence.
    if all(abs(a["features"][COVARIANCE] - 2. * a["features"]["active_orientation"]) < 1e-10 for a in rows):
        return {"status": "UNIDENTIFIABLE_DESIGN", "reason": "EXACT_COVARIANCE_ACTIVE_ORIENTATION_COLLINEARITY",
                "group_count": len(groups), "n_observations": len(rows), "effect_claim_allowed": False}
    fold_by_group = {group: i % 4 for i, group in enumerate(groups)}
    folds = []
    base_squared: list[float] = []
    add_squared: list[float] = []
    for fold in range(4):
        train = [r for r in rows if fold_by_group[r["group_id"]] != fold]
        test = [r for r in rows if fold_by_group[r["group_id"]] == fold]
        fail(bool(train) and bool(test), "empty group-held-out fold")
        p0 = _train_predict(train, test, FEATURES)
        p1 = _train_predict(train, test, FEATURES + (COVARIANCE,))
        l0 = [(r["target_band_ratio"] - p)**2 for r,p in zip(test,p0)]
        l1 = [(r["target_band_ratio"] - p)**2 for r,p in zip(test,p1)]
        base_squared.extend(l0)
        add_squared.extend(l1)
        folds.append({"fold":fold,"train_group_count":len(set(r['group_id'] for r in train)),
                      "test_group_count": len(set(r['group_id'] for r in test)),
                      "test_n":len(test), "baseline_mse": sum(l0)/len(l0),
                      "augmented_mse":sum(l1)/len(l1), "delta_mse": (sum(l0)-sum(l1))/len(l0)})
    m0 = sum(base_squared) / len(base_squared)
    m1 = sum(add_squared) / len(add_squared)
    # Point estimates only; threshold is a synthetic diagnostic, not a hypothesis test.
    return {"status":"SYNTHETIC_PREDICTION_COMPARISON_ONLY", "n_observations":len(rows),
            "group_count":len(groups), "fold_count":4, "baseline_mse":m0,
            "augmented_mse":m1, "delta_mse":m0-m1, "folds":folds,
            "positive_delta_all_folds":all(f["delta_mse"] > 0 for f in folds),
            "effect_claim_allowed":False,
            "measurements_independent_of_SERS_certified":False}


def run(*, a1: dict, c2: dict, c3: dict, sha_a1: str, sha_c2: str,
        sha_c3: str) -> dict:
    lineage = _origins(a1,c2,c3,sha_a1,sha_c2)
    positive = _metrics(fixture(effect=1.15))
    null = _metrics(fixture(effect=0.))
    unidentifiable = _metrics(fixture(effect=1.15, collinear=True))
    fail(positive["delta_mse"] > 0.05 and positive["positive_delta_all_folds"], "positive fixture not separated")
    fail(null["delta_mse"] < 0.01, "null fixture spuriously separated")
    fail(unidentifiable["status"] == "UNIDENTIFIABLE_DESIGN", "collinearity not guarded")
    design = {
        "comparison_type":"NESTED_NONEXCLUSIVE_PREDICTIVE_INCREMENT",
        "scientific_question":"Does independently measured orientation–local-field covariance improve out-of-group prediction of relative SERS band ratios beyond active-subpopulation orientation and matched controls?",
        "baseline_model":"f(active-subpopulation orientation, mean orientation, mean enhancement, substrate indicator)",
        "augmented_model":"baseline + registered orientation–field covariance",
        "operational_estimand":"mean squared prediction error of baseline minus that of augmented model on held-out independent sample/condition groups (delta_MSE)",
        "positive_delta_means":"predictive increment in this specified model/sampling regime, not unique mechanism identification or causation",
        "null_or_negative_delta_means":"no demonstrated incremental utility in this design, NOT universal falsification of B06",
        "observational_unit":"sample/condition group with paired independent orientation, local-field and SERS readouts",
        "validation_split":"groups never shared across train and test; train-only feature scaling",
        "required_measurement_witnesses":["independently acquired orientation states", "independently calibrated spatial EM/field map", "registered orientation–field covariance derived without the target SERS ratio", "SERS band-ratio target and uncertainty", "same specimen/time/condition linkage"],
        "required_controls":["substrate and reporter chemistry", "adsorbate coverage/access", "measurement geometry and polarization", "nanostructure morphology and average enhancement", "group registration uncertainty"],
        "causal_claim":False,
        "mechanisms_mutually_exclusive":False,
        "potential_nonidentifiability":["covariance may encode the active-subpopulation proxy mathematically", "feature collinearity", "correlated instrumentation artefacts", "unmodeled nonlinear interactions", "group and substrate distribution shift"],
        "parent_idea_ids_preserved_separately":sorted(PARENTS),
    }
    return {"schema_version":"m4c4-conditional-predictive-increment-shadow-v1",
            "status":"SYNTHETIC_DISCRIMINATION_DEMO_ONLY_NO_SCIENTIFIC_REVISION",
            "source_sha256":{"m4a1":sha_a1,"m4c2":sha_c2,"m4c3":sha_c3},
            "case_id":CASE, "lineage":lineage, "design":design,
            "synthetic_fixtures":{"COVARIANCE_EFFECT_INJECTED":positive,
                                  "NO_COVARIANCE_EFFECT":null,
                                  "EXACT_COLLINEARITY":unidentifiable},
            "source_files_mutated":False,"research_idea_nodes_created":False,
            "empirical_data_consumed":False,"scientific_improvement_certified":False,
            "scientific_truth_or_falsification_authority":False,"llm_or_network_calls":0,
            "production_selection_changed":False,
            "report_id":"m4c4_confrontation:"+_stable([sha_a1,sha_c2,sha_c3,design])[:24]}


def render_report(report: Mapping[str,Any]) -> str:
    f=report["synthetic_fixtures"]
    def delta(x: dict) -> str:
        return f'{x["delta_mse"]:.6f}' if "delta_mse" in x else "NOT_IDENTIFIABLE"
    return f'''# M4-C4 — B03/B06 Operational Confrontation (PRIVATE)

Status: `{report['status']}`

**Scientific question:** {report['design']['scientific_question']}

**Estimand:** {report['design']['operational_estimand']}

Original parent ResearchIdea IDs: `{report['lineage']['b03_idea_id']}` (B03), `{report['lineage']['b06_idea_id']}` (B06). Both preserved; not composed or revised.

| Synthetic fixture | Delta held-out MSE (baseline - augmented) | Interpretation |
|---|---:|---|
| Injected covariance contribution | {delta(f['COVARIANCE_EFFECT_INJECTED'])} | Controlled simulation detects its own injected predictive effect |
| No covariance contribution | {delta(f['NO_COVARIANCE_EFFECT'])} | Negative-control check; no science inference |
| Covariance exactly collinear with active orientation | {delta(f['EXACT_COLLINEARITY'])} | Unidentifiable, no effect estimate |

All fixtures have synthetic observations only; **zero independent empirical measurements**. A positive delta_MSE is not confirmation of B06, a zero/negative delta_MSE is not falsification, and **no ResearchIdeaNode is created**.

**Scientific next gate:** independently acquired, paired measurements; shared sample/time/condition identities; pre-registered controls and out-of-group holdout; uncertainty and collinearity analysis; expert review of actual conditional mechanism separation.
'''
