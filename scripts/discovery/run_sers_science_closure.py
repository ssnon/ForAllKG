"""SERS-only scientific closure: synthetic identifiability stress test; opt-in."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from pipeline_core.discovery.sers_science_closure import closure_report,data_contract_summary,_save_json


def _write_csv(path:Path, rows:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore")
            writer.writeheader();writer.writerows(rows)


def write_report(report:dict,out:Path):
    _save_json(out/"SERS_CLOSURE_SYNTHETIC_RESULTS.json",report)
    kinetic=report["kinetic_model_benchmark"]["cases"]
    _write_csv(out/"SERS_KINETIC_MODEL_NLL.csv",[
        {"condition":row["condition"],"truth":row["truth"],"assay_accuracy":row["assay_accuracy"],
         **{"nll_"+k:v for k,v in row["mean_nll_per_frame"].items()},
         "best":row["lowest_nll_model"],"gap_nats_per_frame":row["best_to_second_gap_nats_per_frame"],
         "status":row["synthetic_status"]} for row in kinetic])
    _write_csv(out/"SERS_DWELL_MODEL_NLL.csv",[
        {"truth":row["truth"],"nll_mix":row["mean_test_nll_per_observed_dwell"]["markov_mixture"],
         "nll_weibull":row["mean_test_nll_per_observed_dwell"]["weibull_age"],
         "best":row["best_heldout_fit"], "nll_weibull_minus_mix":row["weibull_minus_mixture_nll"]}
        for row in report["dwell_model_benchmark"]["heldout_fit"]])
    _save_json(out/"SERS_OBSERVATION_REQUIREMENTS.json",{
        "status":"EXPERIMENTAL_PROTOCOL_NOT_EXECUTED",
        "experiment_a":{"name":"Adsorption exchange vs photophysical blinking vs joint model",
          "independent_measurements":["hotspot representative occupancy/kinetic assay with spatial and time response kernels", "independent photophysical reporter", "thermal and morphology/field drift proxy"],
          "interventions":["kinetics perturbation verified by occupancy assay", "illumination history/dose perturbation under thermal control", "sham controls and time-window sweep"],
          "primary_measures":["Raman band-vector trajectories", "cross-band covariance and temporal autocovariance", "integrated intensity ratio R_T (not mean of instantaneous ratios)"],
          "critical_nulls":["exchange only", "blinking only", "joint exchange+blinking", "shared drift / EM morphology changes"],
          "identifiability_requirement":"fit calibration on disjoint data; freeze model parameters and observation kernels on heldout experiments", "negative_result":"if all eligible models within empirical prediction uncertainty, report observational equivalence, not falsification"},
        "experiment_b":{"name":"Hidden Markov/phase-type vs age-dependent adsorption dwell",
          "independent_measurements":["state / residence assays resolving short dwells", "preparation-specific hidden-state proxy and initial composition", "instrument response, dark-state detection and censoring"],
          "interventions":["age controlled preparation", "reset/sham reset with measured hidden-state composition", "time resolution sensitivity"],
          "critical_nulls":["finite-state hidden Markov with preparation-specific initialization", "phase-type family with increased hidden state capacity", "semi-Markov age dependence", "photophysical/optical gain"],
          "observation_target":"joint dwell survival and R_T = (sum intensity band a)/(sum intensity band b)",
          "negative_result":"nonexponential survival or history-dependent aggregate hazard alone is NOT memory identification"},
        "mandatory_gate":["eligible external SERS measured data", "independent measurement of hotspot sampling representation", "stable calibration and heldout groups", "predeclared scope and detection bandwidth", "external prior-art search before any novelty claim"],
        "all_protocol_elements_are_proposals_not_results":True})
    lines=[
        "# SERS Scientific Closure — Synthetic Benchmark Result", "",
        "**Result class:** synthetic model stress test; no actual SERS experiment and no scientific discovery certified.", "",
        "## Native G6 lineage", "",
        f"- Native report: `{report['provenance']['native_report_id']}`",
        f"- Discriminate feedback G6: `{report['provenance']['feedback_child_ids']['P2_DISCRIMINATE']}`",
        f"- Explore feedback G6: `{report['provenance']['feedback_child_ids']['P2_EXPLORE']}`",
        "", "## Experiment A: exchange vs blinking (calibrated oracle HMM)", "",
        "| Synthetic condition | Truth | Preferred held-out model | Second-best gap (nats/frame) | Status |",
        "|---|---|---|---:|---|",
    ]
    for row in kinetic:
        lines.append(f"| {row['condition']} | {row['truth']} | {row['lowest_nll_model']} | {row['best_to_second_gap_nats_per_frame']:.4f} | {row['synthetic_status']} |")
    lines.extend(["", "When exchange and blinking have matched transitions and Raman signatures, Raman-only HMM likelihoods are exactly symmetric. A *representative and independently calibrated* occupancy assay may break this symmetry. Its absence or poor accuracy must not be labeled exchange refutation.","",
        "## Experiment B: hidden Markov mixture vs Weibull dwell", "",
        "| Synthetic dwell truth | Best held-out fit | Mix NLL/dwell | Weibull NLL/dwell |", "|---|---|---:|---:|"])
    for row in report["dwell_model_benchmark"]["heldout_fit"]:
        a=row["mean_test_nll_per_observed_dwell"]
        lines.append(f"| {row['truth']} | {row['best_heldout_fit']} | {a['markov_mixture']:.4f} | {a['weibull_age']:.4f} |")
    lines.extend(["", "Two hidden exponential Markov substates create a non-exponential aggregate survival. Altering initial hidden mixture across preparations changes the aggregate hazard even when each state is memoryless. A 2-exponential null is not the full phase-type/HMM family.","",
        "## SERS completion gate", "", "- Executable theory-and-simulation: **completed**.",
        "- Empirical SERS validation: **not performed**.", "- Hotspot-representative independent occupancy/photophysical measurement: **not supplied**.",
        "- Novelty / physical mechanism confirmed: **no**.", "", "Next concrete action: acquire a **paired time-resolved Raman + independent occupancy/photophysics** dataset, with calibrated temporal/spatial kernels and perturbation labels, then score frozen heldout model predictions. See `SERS_OBSERVATION_REQUIREMENTS.json`." ])
    (out/"SERS_SCIENTIFIC_CLOSURE_REPORT_KO.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description="Opt-in SERS science closure benchmark, never scientific authority")
    sp=p.add_subparsers(dest="command",required=True)
    bench=sp.add_parser("benchmark")
    bench.add_argument("--g6",type=Path,required=True)
    bench.add_argument("--out",type=Path,required=True)
    bench.add_argument("--seed",type=int,default=1826)
    bench.add_argument("--fast",action="store_true")
    check=sp.add_parser("check-real-data")
    check.add_argument("--manifest",type=Path,required=True)
    check.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.command=="benchmark":
        r=closure_report(a.g6,a.seed,a.fast)
        write_report(r,a.out)
        print(json.dumps({"status":r["status"],"g6_native_children":r["provenance"]["matched_native_children"],
                          "synthetic_cases":len(r["kinetic_model_benchmark"]["cases"]),
                          "dwell_null_families":len(r["dwell_model_benchmark"]["heldout_fit"]),
                          "empirical_validation_done":False,"out":str(a.out)},indent=2))
    elif a.command=="check-real-data":
        r=data_contract_summary(a.manifest)
        if a.out: _save_json(a.out,r)
        print(json.dumps(r,indent=2))

if __name__=="__main__":main()
