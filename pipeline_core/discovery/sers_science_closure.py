"""Opt-in SERS mechanistic *synthetic* falsifiability benchmark.

Not SERS evidence. Frozen calibrated forward models and explicit observability
limits. No API, graph writes, official scheduler, or positive-premise authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, expit

STATES = np.asarray(list(itertools.product([0, 1], [0, 1])), dtype=int)
MU = np.array([10.0, 8.0])
A = np.array([1.7, -0.75])
B = np.array([1.7, -0.75])  # exactly confounded SERS signatures
INTERACTION = np.array([1.4, 1.0])
NOISE = np.array([0.65, 0.65])
Q_X = 0.10
Q_B = 0.10             # equal kinetics, maximal SERS-only ambiguity

FEEDBACK_IDEA_IDS = {
    "P2_DISCRIMINATE": "research_idea:487abeb3131a31a46c60",
    "P2_EXPLORE": "research_idea:4dde71376673638b17e8",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def check_frozen_g6(source_json: Path, expected_arm: str = "SCIENCE_FEEDBACK") -> dict[str, Any]:
    raw = source_json.read_bytes()
    payload = json.loads(raw)
    if payload.get("generation_index") != 6 or payload.get("canonical_graph_mutated") is not False:
        raise ValueError("not frozen native shadow G6 generation-6 report")
    if payload.get("scientific_truth_authority") is not False or payload.get("production_selection_authority") is not False:
        raise ValueError("G6 epistemic authority mismatch")
    nodes = payload.get("offspring_nodes", [])
    ids = {node["idea_id"] for node in nodes}
    expected = set(FEEDBACK_IDEA_IDS.values())
    if ids != expected:
        raise ValueError(f"expected {expected_arm} 2 frozen G6 children: {ids} != {expected}")
    semantic_by_id = {s["idea_id"]: s for s in payload["semantic_records"]}
    for node in nodes:
        if node["origin_kind"] != "GENERATIONAL_OFFSPRING" or node["epistemic_status"] != "INSPIRATION_ONLY":
            raise ValueError("invalid research idea authority")
        s = semantic_by_id[node["idea_id"]]
        if s["disposition"] != "GENUINE_CHILD" or not s["retained_in_g4_population"]:
            raise ValueError("not genuine retained shadow G6 offspring")
        if node["source_context_id"] != "hypothesis_context:221b1022ba7f9f6e2719":
            raise ValueError("G6 context mismatched")
    return {"source_g6_file_sha256": _sha(raw), "native_report_id": payload["report_id"], "generation_index": 6,
            "feedback_child_ids": FEEDBACK_IDEA_IDS, "matched_native_children": 2,
            "source_file": str(source_json.resolve())}


def p_two(q: float) -> np.ndarray:
    if not 0 < q < 0.5:
        raise ValueError("transition probability must be in (0,0.5)")
    return np.array([[1-q, q], [q, 1-q]])


def trans(family: str = "exchange") -> np.ndarray:
    if family == "coupled":
        # Synthetic adsorption-dependent photophysical transition rate; explicitly
        # NOT a measured physical SERS rate or an inferred causal interaction.
        rows = np.empty((4, 4))
        for i, (x,b) in enumerate(STATES):
            px = p_two(Q_X)[x]
            pb = p_two(0.035 if x == 0 else 0.22)[b]
            for j,(xn,bn) in enumerate(STATES):
                rows[i,j] = px[xn]*pb[bn]
        return rows
    return np.kron(p_two(Q_X), p_two(Q_B))


def emission_means(family: str) -> np.ndarray:
    if family not in {"exchange", "blinking", "dual", "coupled", "static"}:
        raise ValueError(f"unknown model: {family}")
    arr = np.tile(MU, (4, 1)).astype(float)
    for j, (x, b) in enumerate(STATES):
        if family in {"exchange", "dual", "coupled"}:
            arr[j] += A * (x - 0.5)
        if family in {"blinking", "dual", "coupled"}:
            arr[j] += B * (b - 0.5)
        if family == "coupled":
            arr[j] += INTERACTION * (x-0.5)*(b-0.5)
    return arr


def draw_trace(seed: int, family: str, n: int, assay_accuracy: float | None) -> tuple[np.ndarray, np.ndarray | None]:
    if n < 30:
        raise ValueError("n too small")
    if assay_accuracy is not None and not 0.5 <= assay_accuracy <= 1:
        raise ValueError("assay_accuracy must be >=0.5 and <=1")
    rng = np.random.default_rng(seed)
    P = trans(family)
    means = emission_means(family)
    hidden = np.zeros(n, dtype=int)
    hidden[0] = int(rng.integers(4))
    for i in range(1, n):
        hidden[i] = rng.choice(4, p=P[hidden[i-1]])
    y = means[hidden] + rng.normal(size=(n,2)) * NOISE
    z = None
    if assay_accuracy is not None:
        true_x = STATES[hidden, 0]
        z = np.where(rng.random(n) < assay_accuracy, true_x, 1-true_x).astype(int)
    return y, z


def hmm_nll(y: np.ndarray, z: np.ndarray | None, family: str, assay_accuracy: float | None) -> float:
    """Four latent states; fixed exact same Markov and Gaussian observation kernel.

    All parameters are known by synthetic oracle calibration. Not a practical
    model-fitting result and not evidence of real SERS identifiability.
    """
    means = emission_means(family)
    logP = np.log(trans(family))
    lp = np.full(4, -np.log(4.0))
    for t in range(len(y)):
        if t:
            lp = logsumexp(lp[:,None] + logP, axis=0)
        diff = (y[t][None,:] - means) / NOISE
        emit = -0.5*np.sum(diff*diff + np.log(2*np.pi*(NOISE**2)), axis=1)
        if z is not None:
            x = STATES[:,0]
            if assay_accuracy is None:
                raise ValueError("assay channel must be calibrated")
            prob = np.where(x == z[t], assay_accuracy, 1-assay_accuracy)
            emit += np.log(np.maximum(prob, 1e-300))
        lp += emit
        normalization = logsumexp(lp)
        lp -= normalization
        if t == 0:
            total_loglike = normalization
        else:
            total_loglike += normalization
    return -float(total_loglike) / len(y)


def exchange_blinking_benchmark(*, seed: int = 1826, n: int = 280, replicates: int = 6) -> dict[str, Any]:
    """Precalibrated held-out synthetic trajectories, not real model selection."""
    if replicates < 2:
        raise ValueError("need >=2 independent held-out traces")
    families = ["exchange", "blinking", "dual", "coupled", "static"]
    conditions = [
        ("exchange_no_occupancy_assay", "exchange", None),
        ("exchange_reliable_occupancy_assay", "exchange", 0.95),
        ("blinking_reliable_occupancy_assay", "blinking", 0.95),
        ("dual_reliable_occupancy_assay", "dual", 0.95),
        ("coupled_reliable_occupancy_assay", "coupled", 0.95),
        ("exchange_uninformative_assay", "exchange", 0.50),
    ]
    rows = []
    for j, (label, truth, accuracy) in enumerate(conditions):
        scores = []
        for i in range(replicates):
            y,z = draw_trace(seed+10000*j+11*i,truth,n,accuracy)
            scores.append({f: hmm_nll(y,z,f,accuracy) for f in families})
        means = {f: round(float(np.mean([r[f] for r in scores])),6) for f in families}
        ranked = sorted(means.items(), key=lambda kv: kv[1])
        best, runner = ranked[0],ranked[1]
        gap = round(runner[1]-best[1],6)
        # standard error of paired logscore DIFFERENCES across independent traces
        differences = np.array([r[runner[0]]-r[best[0]] for r in scores])
        se = float(np.std(differences,ddof=1)/np.sqrt(replicates))
        verdict = "RESOLVED_IN_SYNTHETIC_ORACLE" if gap > max(0.015,2*se) else "INDETERMINATE_OR_EQUIVALENT"
        rows.append({"condition":label,"truth":truth,"assay_accuracy":accuracy,
                     "n_frames_per_heldout_trace":n,"independent_heldout_traces":replicates,
                     "mean_nll_per_frame":means,"lowest_nll_model":best[0],
                     "best_to_second_gap_nats_per_frame":gap,"paired_gap_se":round(se,6),
                     "synthetic_status":verdict})
    return {"title":"SERS exchange/blinking oracle HMM observation-kernel test",
            "scope":"Synthetic oracle parameters fixed before heldout draws; no model fitting, no empirical evidence",
            "observation_equivalence":"Equal x/b rates and spectral signatures make exchange and blinking *exactly* SERS-only observationally equivalent; occupancy channel breaks symmetry only if representative",
            "parameters":{"mu":MU.tolist(),"exchange_signature":A.tolist(),"blinking_signature":B.tolist(),
                          "noise_sd":NOISE.tolist(),"qx":Q_X,"qb":Q_B,"interaction_signature":INTERACTION.tolist(),"model_family_set":families},
            "cases":rows}


def mix_sf(t: np.ndarray | float, w: float, k1: float, k2: float) -> np.ndarray:
    v = np.asarray(t)
    return w*np.exp(-k1*v)+(1-w)*np.exp(-k2*v)


def mix_pdf(t: np.ndarray, w: float, k1: float, k2: float) -> np.ndarray:
    return w*k1*np.exp(-k1*t)+(1-w)*k2*np.exp(-k2*t)


def mix_hazard(t: float, w: float, k1: float, k2: float) -> float:
    return float(mix_pdf(np.array([t]),w,k1,k2)[0]/mix_sf(t,w,k1,k2))


def sample_dwells(rng: np.random.Generator, family: str, size: int) -> np.ndarray:
    if family == "markov_mixture":
        k = np.where(rng.random(size)<0.60, 0.33, 1.8)
        return rng.exponential(1/k)
    if family == "weibull_age":
        return 2.0*rng.weibull(1.65,size)
    raise ValueError(family)


def observed_dwells(seed: int, family: str, wanted: int, cutoff: float = 0.17, cap: float = 6.0) -> tuple[np.ndarray,np.ndarray]:
    """Known left truncation at cutoff and right censoring at cap.

    This is a simplified detector observation kernel: missed sub-cutoff events
    are discarded; full missed-event dynamics and spectral ratios are NOT modeled.
    """
    rng=np.random.default_rng(seed)
    samples=[]
    while len(samples)<wanted:
        proposal=sample_dwells(rng,family,wanted)
        samples.extend(proposal[proposal>=cutoff].tolist())
    actual=np.array(samples[:wanted])
    return np.minimum(actual,cap),actual<cap


def dwell_nll_logp(params: np.ndarray, times: np.ndarray, events: np.ndarray, family: str,
                   cutoff: float) -> float:
    if family=="markov_mixture":
        logit_w, log_k1, log_k2 = params
        w=float(expit(logit_w)); k1=float(np.exp(log_k1)); k2=float(np.exp(log_k2))
        sf=np.maximum(mix_sf(times,w,k1,k2),1e-300)
        pdf=np.maximum(mix_pdf(times,w,k1,k2),1e-300)
        trunc=math.log(max(float(mix_sf(cutoff,w,k1,k2)),1e-300))
        ll=np.where(events,np.log(pdf),np.log(sf))-trunc
    elif family=="weibull_age":
        log_shape, log_scale=params
        shape=float(np.exp(log_shape)); scale=float(np.exp(log_scale))
        tt=np.maximum(times,1e-12)
        power=(tt/scale)**shape
        logsf=-power
        logpdf=np.log(shape)-np.log(scale)+(shape-1)*(np.log(tt)-np.log(scale))-power
        trunc=-(cutoff/scale)**shape
        ll=np.where(events,logpdf,logsf)-trunc
    else:
        raise ValueError(family)
    return -float(np.sum(ll))/len(times)


def fit_dwells(times: np.ndarray, events: np.ndarray, family: str, cutoff:float=0.17) -> np.ndarray:
    if family=="markov_mixture":
        starts=[[0,np.log(.3),np.log(2)],[0,np.log(1),np.log(.3)],[-1,np.log(.15),np.log(1.2)]]
        bounds=[(-4,4),(-4,3),(-4,3)]
    else:
        starts=[[np.log(1.5),np.log(2)], [0,0]]
        bounds=[(-2,2),(-3,3)]
    fitted=[minimize(dwell_nll_logp,np.array(s),args=(times,events,family,cutoff),method="L-BFGS-B",bounds=bounds)
            for s in starts]
    best=min(fitted,key=lambda r:r.fun)
    if not best.success and not np.isfinite(best.fun):
        raise RuntimeError(f"{family} parameter fit failed")
    return best.x


def dwell_competition_benchmark(*, seed:int=1826, train:int=400, test:int=400,replicates:int=5) -> dict[str,Any]:
    rows=[]
    for j,truth in enumerate(["markov_mixture","weibull_age"]):
        res=[]
        for r in range(replicates):
            t,e=observed_dwells(seed+10000*j+r*31,truth,train)
            tt,ee=observed_dwells(seed+10000*j+r*31+170000,truth,test)
            nll={}
            for alternative in ["markov_mixture","weibull_age"]:
                p=fit_dwells(t,e,alternative)
                nll[alternative]=dwell_nll_logp(p,tt,ee,alternative,0.17)
            res.append(nll)
        vals={m:round(float(np.mean([z[m] for z in res])),6) for m in res[0]}
        better=min(vals,key=vals.get)
        gaps=[v["weibull_age"]-v["markov_mixture"] for v in res]
        rows.append({"truth":truth,"best_heldout_fit":better,"mean_test_nll_per_observed_dwell":vals,
                     "replicates":replicates,"train_dwells_per_repeat":train,"test_dwells_per_repeat":test,
                     "weibull_minus_mixture_nll":round(float(np.mean(gaps)),6),
                     "mixture_wins_fraction":round(float(np.mean(np.array(gaps)>0)),4)})
    ha=[{ "t":t,"prep_A_w":0.2,"prep_A_hazard":mix_hazard(t,0.2,0.33,1.8),
          "prep_B_w":0.8,"prep_B_hazard":mix_hazard(t,0.8,0.33,1.8)} for t in [0.0,0.5,2.0,5.0]]
    ratio_integrals=(2+9)/(1+3)
    integral_instant_ratios=((2/1)+(9/3))/2
    return {"title":"Known-truncation/censoring dwell competition (2-exp versus Weibull)",
            "scope":"Synthetic finite-mixture lower-bound null; not a general hidden-Markov/phase-type versus semi-Markov proof",
            "known_left_truncation":0.17,"right_censor_time":6.0,
            "age_hazard_counterexample":{"mechanism":"two exponential Markov substates, preparation-specific hidden mixture",
                                          "rates":[0.33,1.8],"timepoint_hazards":ha},
            "ratio_observation_counterexample":{"two_equal_duration_band1":[2,9],"band2":[1,3],
                                                 "ratio_of_integrated_intensities":ratio_integrals,
                                                 "time_mean_of_instant_ratios":integral_instant_ratios},
            "heldout_fit":rows,"scientific_warning":"Nonexponential survival or preparation-dependent aggregate hazard does not imply non-Markov microscopic dynamics. Finite data can leave rich HMMs/semi-Markov models observationally equivalent."}


def data_contract_summary(path: Path) -> dict[str,Any]:
    """Validate a prospective experimental data dictionary, not numerical truth.

    Input is a JSON manifest with observed CSV paths, declared provenance and
    independent assay limitations. Fail closed if a necessary experimental
    observability requirement is missing. No fabricated measurements.
    """
    p=json.loads(path.read_text(encoding="utf-8"))
    required=["experiment_id","substrate","raman_trace_csv","occupancy_assay_csv",
              "occupancy_assay_type","occupancy_spatial_support","raman_spatial_support",
              "sampling_interval_seconds","occupancy_assay_resolution_seconds",
              "illumination_control","temperature_control","orientation_control",
              "field_proxy_method","calibration_source","experiment_is_real"]
    missing=[k for k in required if k not in p or p[k] in (None,"")]
    reasons=[]
    if missing: reasons.append("MISSING_REQUIRED_METADATA:"+",".join(missing))
    if p.get("experiment_is_real") is not True: reasons.append("NO_REAL_EXPERIMENT_DECLARED")
    for field in ["raman_trace_csv","occupancy_assay_csv"]:
        candidate=p.get(field)
        if candidate:
            f=Path(candidate)
            if not f.is_absolute(): f=path.parent/f
            if not f.is_file(): reasons.append("MISSING_FILE:"+field)
    for f in ["sampling_interval_seconds","occupancy_assay_resolution_seconds"]:
        try:
            if not float(p[f])>0: reasons.append("INVALID_POSITIVE:"+f)
        except (ValueError,TypeError,KeyError):
            reasons.append("INVALID_POSITIVE:"+f)
    if p.get("occupancy_spatial_support") != p.get("raman_spatial_support"):
        reasons.append("SPATIAL_SAMPLING_MISMATCH_REQUIRES_CALIBRATED_KERNEL")
    try:
        resolution=float(p["occupancy_assay_resolution_seconds"])
        sample=float(p["sampling_interval_seconds"])
        if resolution>5*sample:
            reasons.append("OCCUPANCY_ASSAY_BANDWIDTH_MAY_BE_INADEQUATE")
    except (ValueError, TypeError, KeyError):
        pass  # already reported by numeric metadata gate
    return {"status":"METADATA_GATE_ONLY_NOT_EMPIRICAL_VALIDATION",
            "required_metadata_complete":not missing,"ready_for_joint_analysis":not reasons,
            "blockers":reasons,"experiment_id":p.get("experiment_id"),
            "not_evidence_without_independent_qc":True}


def closure_report(g6:Path,seed:int=1826,fast:bool=False)->dict[str,Any]:
    source=check_frozen_g6(g6)
    kinetic=exchange_blinking_benchmark(seed=seed,n=140 if fast else 280,replicates=3 if fast else 8)
    dwell=dwell_competition_benchmark(seed=seed,train=150 if fast else 550,
                                      test=150 if fast else 550,replicates=2 if fast else 5)
    return {"schema_version":"sers-scientific-closure-synthetic-benchmark-v1",
            "status":"SIMULATED_SERS_MODEL_STRESS_TEST_NOT_SCIENTIFIC_CLAIM",
            "provenance":source,"seed":seed,"kinetic_model_benchmark":kinetic,
            "dwell_model_benchmark":dwell,
            "scientific_closure_gate":{
                "engineering_executable":True,
                "real_SERS_data_tested":False,
                "independent_occupancy_assay_calibrated":False,
                "real_experiment_identifiability_established":False,
                "mechanistic_novelty_certified":False,
                "next_needed":"One hotspot-representative, independently calibrated occupancy/photophysics assay paired with time-resolved SERS and controlled excitation/adsorption perturbations; compare parameter-frozen heldout predictions.",
            },
            "scope_limits":[
                "HMM benchmark has known synthetic oracle parameters; x/b are independent except in the synthetic coupled family, with constant Gaussian emissions and no realistic drift or photon-counting model",
                "Occupancy proxy is assumed to assay precisely the SERS-sampled x; spatial mismatch requires explicit observation kernel",
                "Dwell benchmark tests 2-exponential mixture versus Weibull, not universal HMM versus semi-Markov identifiability",
                "Censoring is right-censoring plus simple left truncation; missed transitions and state censoring are not included",
                "M8 G6 offspring are inspiration only; these synthetic experiments do not verify their scientific or novelty claims",
            ]}
