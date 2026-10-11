"""Opt-in, *non-certifying* SERS novelty search and exactly solvable kinetic contrast.

The finite-state time-asymmetry theorem is known prior physics. The possible
SERS-specific hypothesis and its novelty require external literature review.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any

import numpy as np
from scipy.linalg import expm

FEEDBACK_G6_IDS = {
    "P2_DISCRIMINATE": "research_idea:487abeb3131a31a46c60",
    "P2_EXPLORE": "research_idea:4dde71376673638b17e8",
}

# DOI metadata and description were manually curated from public publication pages.
# Do not promote these abstract-level summaries to full-text scientific evidence.
PRIOR_ART = [
    dict(id="PA2007_ISOTOPES", year=2007, doi="10.1021/ja077243c",
         title="A Frequency Domain Existence Proof of Single-Molecule Surface-Enhanced Raman Spectroscopy",
         known_overlap="SERS isotopologue exchange and two-dimensional band cross-correlation already demonstrated",
         url="https://doi.org/10.1021/ja077243c"),
    dict(id="PA2014_SERS_CORR", year=2014, doi="10.1021/ac403882h",
         title="Surface Enhanced Raman Correlation Spectroscopy of Particles in Solution",
         known_overlap="SERS band intensity correlation and fluctuations already studied",
         url="https://doi.org/10.1021/ac403882h"),
    dict(id="PA2023_ASYMMETRY", year=2023, doi="10.1103/PhysRevLett.131.077101",
         title="Thermodynamic Bound on the Asymmetry of Cross-Correlations",
         known_overlap="Equilibrium cross-correlation reciprocity and non-equilibrium asymmetry are established general physics",
         url="https://doi.org/10.1103/PhysRevLett.131.077101"),
    dict(id="PA2018_BLINKING", year=2018, doi="10.3791/56729",
         title="Observation and Analysis of Blinking Surface-enhanced Raman Scattering",
         known_overlap="SERS blinking dwell distributions and unresolved mechanistic inference are known",
         url="https://doi.org/10.3791/56729"),
    dict(id="PA2025_BULK_BLINKING", year=2025, doi="10.1103/4yg7-1f85",
         title="Blinking in surface-enhanced Raman spectrum beyond a single molecule",
         known_overlap="Blinking need not imply single-molecule entry/exit",
         url="https://doi.org/10.1103/4yg7-1f85"),
    dict(id="PA2026_STERS", year=2026, doi="10.1364/OE.574511",
         title="Stochastic equilibrium Raman spectroscopy (STERS)",
         known_overlap="Photon-correlation enhanced temporal Raman fluctuation measurement already proposed",
         url="https://doi.org/10.1364/OE.574511"),
]

SEARCH_TASKS = (
    ("SYMMETRY_BREAKING", "Explore a conditional new *SERS-specific* prediction from a symmetry or invariant, with a strong instrument/null challenge. The generic detailed-balance theorem itself is KNOWN."),
    ("HETEROGENEITY_INTERACTION", "Seek a falsifiable, non-obvious three-way interaction of localized field, adsorption kinetics, and photophysical state. Distinguish it from a flexible known-state-space model; do not claim that covariance alone is new."),
    ("MEASUREMENT_LIMIT", "Seek a regime boundary in information recoverable from time-binned multiband SERS with independent occupancy and controlled rate perturbations. The general sampling/variance formula and photon-correlation technique are KNOWN."),
)


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_new(path, obj):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def verify_g6(raw):
    """Native v2.9+ reports store offspring nodes as dicts, not always objects."""
    if raw.get("generation_index") != 6:
        raise ValueError("Only frozen native G6 report is admitted")
    rows = raw.get("offspring_nodes", [])
    ids = {n.get("idea_id") if isinstance(n, dict) else getattr(n, "idea_id", None) for n in rows}
    if not set(FEEDBACK_G6_IDS.values()).issubset(ids):
        raise ValueError("Requested feedback G6 children are missing")
    nodes = {}
    for n in rows:
        if not isinstance(n, dict):
            n = n.model_dump(mode="json")
        if n["idea_id"] in FEEDBACK_G6_IDS.values():
            if n.get("epistemic_status") != "INSPIRATION_ONLY" or n.get("positive_premise_authority") is not False:
                raise ValueError("G6 inspiration-only science authority boundary violated")
            nodes[n["idea_id"]] = n
    return nodes


def cyclic_generator(forward: float, backward: float):
    if not (forward >= 0 and backward >= 0 and forward + backward > 0):
        raise ValueError("Rates must be nonnegative and nonzero in total")
    q = np.zeros((3, 3), dtype=float)
    for s in range(3):
        q[s, (s + 1) % 3] = forward
        q[s, (s - 1) % 3] = backward
        q[s, s] = -(forward + backward)
    return q


def stationary_distribution(q):
    q = np.asarray(q, dtype=float)
    n = q.shape[0]
    if q.shape != (n, n) or not np.allclose(q.sum(axis=1), 0.0, atol=1e-12):
        raise ValueError("Q must be a row-stochastic CTMC generator")
    a = q.T.copy()
    a[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(a, b)
    if np.any(pi < -1e-10):
        raise ValueError("Invalid stationary solution")
    return pi


def lag_covariance(q, f, g, lag):
    """C_fg(t)=E[f(X_0)g(X_t)]-E[f]E[g], stationary CTMC."""
    if lag < 0:
        raise ValueError("Lag must be >=0")
    q = np.asarray(q, dtype=float)
    f, g = np.asarray(f, dtype=float), np.asarray(g, dtype=float)
    if f.shape != (q.shape[0],) or g.shape != f.shape:
        raise ValueError("Observation vector dimension mismatch")
    pi = stationary_distribution(q)
    return float((pi * f) @ expm(q * lag) @ g - (pi @ f) * (pi @ g))


def cross_lag_asymmetry(q, f, g, lag):
    """A_fg(t)=C_fg(t)-C_gf(t); vanishes for detailed balance."""
    return lag_covariance(q, f, g, lag) - lag_covariance(q, g, f, lag)


def benchmark():
    f = np.array([2., .5, 1.])
    g = np.array([.2, 1.7, .8])
    lags = [.1, .3, .5, 1., 2.]
    cases = {
        "equilibrium_detailed_balance": cyclic_generator(.7, .7),
        "driven_forward": cyclic_generator(1.2, .2),
        "driven_reverse": cyclic_generator(.2, 1.2),
    }
    traces = {name: [{"lag": lag, "A_fg": cross_lag_asymmetry(q, f, g, lag)} for lag in lags]
              for name, q in cases.items()}
    proportional = [cross_lag_asymmetry(cases["driven_forward"], f, 2*f+5, lag) for lag in lags]
    # Under equilibrium, unmatched channel timing can fake asymmetry.
    eq = cases["equilibrium_detailed_balance"]
    instrument_delay = lag_covariance(eq, f, g, .5+.1) - lag_covariance(eq, g, f, .5)
    result = {
        "scientific_authority": False,
        "novelty_certified": False,
        "experiment_performed": False,
        "known_thermodynamic_theorem_not_original": True,
        "model": "stationary three-state CTMC cyclic kinetics; two state-dependent Raman emission vectors; ideal noiseless detector",
        "observables": {"f": f.tolist(), "g": g.tolist()},
        "rates": {"equilibrium": [.7,.7], "forward": [1.2,.2], "reverse": [.2,1.2]},
        "lag_asymmetry": traces,
        "driven_proportional_channels": proportional,
        "equilibrium_instrument_channel_delay_false_positive": instrument_delay,
        "tests": {
            "db_zero": max(abs(x["A_fg"]) for x in traces["equilibrium_detailed_balance"]) < 1e-10,
            "driven_nonzero": max(abs(x["A_fg"]) for x in traces["driven_forward"]) > .01,
            "drive_reversal_sign": all(abs(a["A_fg"]+b["A_fg"])<1e-10 for a,b in zip(traces["driven_forward"], traces["driven_reverse"])),
            "proportional_emissions_zero": max(map(abs, proportional)) < 1e-10,
            "channel_delay_can_fake_asymmetry": abs(instrument_delay) > 1e-5,
        },
    }
    if not all(result["tests"].values()):
        raise AssertionError("Physical null and counterexample did not satisfy analytical checks")
    return result


def reference_hypothesis():
    return {
        "candidate_id": "SERS_HYPOTHESIS_SEED_001_NOT_NOVELTY_CERTIFIED",
        "source": "HUMAN_ASSISTANT_ANALYTICAL_SEED_NOT_NATIVE_G6_OFFSPRING",
        "parent_context": list(FEEDBACK_G6_IDS.values()),
        "status": "SEARCH_ONLY_LITERATURE_AND_PHYSICS_REVIEW_REQUIRED",
        "research_question": "Can antisymmetric positive-lag cross-covariance between nonproportional SERS vibrational bands distinguish an optically observed driven three-state molecular kinetic cycle from reversible equilibrium fluctuation models, after instrumental timing and photophysical nuisance controls?",
        "novelty_atom": "SERS-specific use of *drive-reversal-sensitive lag cross-covariance antisymmetry* to probe cyclic molecular-state currents, not the established generic stochastic-thermodynamics theorem itself",
        "novelty_caveat": "We have NOT established that this SERS adaptation, or its experimental signature, is unreported in the literature. Literature search is non-exhaustive.",
        "mechanism": "A stationary, actively driven cyclic three-state adsorbate / chemical-state network generates a nonzero probability current. Two non-collinear Raman band emission vectors can reveal the current as positive-lag antisymmetry; equilibrium detailed balance removes that antisymmetry.",
        "equations": [
            "Q_ij is the CTMC transition-rate generator with row sums zero and pi Q=0.",
            "C_fg(t)=sum_i,j pi_i f_i [exp(Qt)]_ij g_j -(sum_i pi_i f_i)(sum_j pi_j g_j)",
            "A_fg(t)=C_fg(t)-C_gf(t)",
            "Detailed balance pi_i Q_ij=pi_j Q_ji implies A_fg(t)=0 for stationary state observables.",
        ],
        "differential_predictions": [
            "A_fg(t)=0 under stationary detailed-balance null, including arbitrarily many hidden equilibrium Markov states, under ideal calibrated observation.",
            "A_fg(t) can be nonzero for driven cyclic kinetics with two spectrally non-collinear emission observables; this is conditional, not a universal necessity.",
            "Reversing the cyclic affinity can reverse the sign of A_fg(t) if the relevant state emissions and experimental transfer functions are unchanged.",
            "Proportional (or affine-related) state-specific band emission vectors make A_fg(t)=0 even if the hidden system breaks detailed balance.",
        ],
        "strong_competing_nulls": [
            "Equilibrium multistate hidden Markov (arbitrary stationary state mixture) with detailed balance",
            "Driven photophysical blinking / photocharging / chemical transformations without adsorption-state cycling",
            "Unequal spectrometer channel delays, response convolution, shot noise and filtering",
            "Non-stationary illumination, focus drift, morphology fluctuation or photothermal hysteresis",
            "A fully flexible general nonequilibrium hidden-state model (can mimic the signature; adsorption-specific inference needs orthogonal assays)",
        ],
        "independent_measurements": [
            "Band-dependent timing calibration using a common stable reference",
            "Independent adsorption occupancy or molecular-state reporter representative of the actual hotspot",
            "Absorbed optical power, temperature, morphology and photophysics controls",
            "Reversible drive-direction switching with same average occupancy and spectral response where feasible",
        ],
        "falsification_scope": "Disfavor the *specified kinetic-cycle parameterization* if premeasured rates and emissions predict a resolvable signed A_fg under preregistered conditions but data (after timing and nuisance correction) do not reproduce it. A_fg=0 alone never disproves non-equilibrium currents because observation vectors may be insensitive.",
        "related_known_results": [row["id"] for row in PRIOR_ART],
        "positive_premise_authority": False,
        "scientific_truth_authority": False,
        "novelty_certified": False,
    }


def make_tasks(g6_nodes):
    g6_context = [{"idea_id": n["idea_id"], "kernel": n["kernel"],
                   "differential_prediction": n.get("differential_prediction", ""),
                   "falsification_condition": n.get("falsification_condition", "")}
                  for n in g6_nodes.values()]
    tasks = []
    for name, objective in SEARCH_TASKS:
        system = ("You are generating research ideas, not literature findings. Search aggressively, claim conservatively. "
                  "Construct a physically explicit SERS hypothesis whose distinguishing claim survives strong nulls. "
                  "Never assert scientific novelty, evidence or publication absence. Return only one JSON object. "
                  "Admit that the proposal may be known; propose exact prior-art adversarial search queries. "
                  "A candidate must change a *prediction or mechanism*, not merely add variables. "
                  "The G6 ideas and the supplied physics are inspirational context, never positive evidence.")
        payload = {
            "task_id": name,
            "objective": objective,
            "research_context": g6_context,
            "known_prior_art": PRIOR_ART,
            "math_seed": reference_hypothesis() if name == "SYMMETRY_BREAKING" else {},
            "do_not_claim_novelty": True,
            "return_exact_json_keys": ["task_id", "title", "conceptual_delta", "novelty_atom", "mechanism", "assumptions", "equations", "strong_nulls", "differential_predictions", "independent_observables", "interventions", "falsification_scope", "prior_art_queries", "feasibility_limits", "risk_of_known_physics_reexpression"],
            "constraints": [
                "At least one symbolic conditional model equation, a prediction that differs under an explicit strong null, and an accessible intervention.",
                "If the candidate is implied by known physics, explicitly label the remaining novelty only as an application or new experimental combination.",
                "For each supposed unique element, search exact relation plus synonyms, inverse expressions, and adjacent spectroscopy fields.",
                "Provide 4-8 concrete prior-art queries. No fabricated references, measurements, numeric physical constants or novelty certification.",
                "Never treat paper absence in this short provided set as evidence of novelty.",
            ],
        }
        tasks.append({"task_id": name, "system": system,
                      "user": json.dumps(payload, ensure_ascii=False, indent=2),
                      "source_g6": list(g6_nodes), "positive_premise_authority": False})
    return tasks


def _has_mathematics(v):
    """Format check only. Mathematical/physical validity requires independent review."""
    import re
    strings = []
    if isinstance(v, str):
        strings = [v]
    elif isinstance(v, list):
        strings = [s for s in v if isinstance(s, str)]
    elif isinstance(v, dict):
        strings = [s for s in v.values() if isinstance(s, str)]
    return any(re.search(r"[=≈∝≤≥∂]|\\(?:frac|sum|partial|int|dot)", s) for s in strings)


def _nonempty_structured(v, *, name):
    """Accept scientific structured arrays WITHOUT flattening or inventing content."""
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, list):
        return bool(v) and all(_nonempty_structured(x, name=name) for x in v)
    if isinstance(v, dict):
        if not v or not all(isinstance(k, str) and k.strip() for k in v):
            return False
        if name == 'differential_predictions':
            if not (any(k in v for k in ('model', 'null_model', 'condition')) and 'prediction' in v):
                return False
        elif name == 'strong_nulls':
            if not (any(k in v for k in ('name', 'model', 'null_model')) and any(k in v for k in ('prediction', 'expected_result', 'distinction'))):
                return False
        return all(_nonempty_structured(x, name=name) for x in v.values())
    return False


def _valid_equations(value):
    if isinstance(value, str):
        return bool(value.strip())  # Preserve backwards-compatible legacy draft shape.
    if isinstance(value, list):
        return bool(value) and all(isinstance(s,str) and s.strip() for s in value)
    if isinstance(value, dict):
        return (bool(value) and
                all(isinstance(k,str) and k.strip() and isinstance(s,str) and s.strip()
                    for k,s in value.items()) and _has_mathematics(value))
    return False


def validate_candidate(candidate, task_id):
    """Structural validation; DOES NOT certify scientific validity or novelty.

    Native generation accepts both flat lists and richer structured model outputs;
    original text, equations, and null specifications remain untouched.
    """
    required = ['title', 'conceptual_delta', 'novelty_atom', 'mechanism',
                'assumptions', 'equations', 'strong_nulls', 'differential_predictions',
                'independent_observables', 'interventions', 'falsification_scope',
                'prior_art_queries', 'feasibility_limits',
                'risk_of_known_physics_reexpression']
    if not isinstance(candidate, dict) or candidate.get('task_id') != task_id:
        raise ValueError('Candidate task identity mismatch')
    for k in required:
        v = candidate.get(k)
        if k == 'equations':
            valid = _valid_equations(v)
        elif k == 'prior_art_queries':
            valid = isinstance(v,list) and len(v) >= 4 and all(isinstance(x,str) and x.strip() for x in v)
        else:
            valid = _nonempty_structured(v, name=k)
        if not valid:
            raise ValueError(f'Candidate missing or malformed substantive field: {k}')
    return {**candidate, 'positive_premise_authority':False,
            'novelty_certified':False,'scientific_truth_authority':False,
            'evidence_status':'UNVERIFIED_SEARCH_ONLY'}


def review_queue(candidates):
    rows = []
    for c in candidates:
        for i,q in enumerate(c.get("prior_art_queries", []),1):
            rows.append({"candidate_id":c["candidate_id"],"query_id":f"{c['candidate_id']}:{i}",
                         "query":q,"doi_or_url":"","source_relation":"NOT_REVIEWED",
                         "verbatim_support":"","reader_notes":"","novelty_authority":"NONE"})
    return rows
