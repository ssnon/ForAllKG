"""M6.1: bounded, nonauthoritative adversarial development of M6 speculative branches.

The deterministic attacks are *calibration countermodels*, not evaluations of truth.
Optional model-supplied critiques and divergent branch deltas are untrusted proposals.
No SIS mutation, automatic promotion, selection or novelty certification occurs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline_core.discovery.research_idea_scientific_development_m6 import (
    SCHEMA as M6_SCHEMA, REQUIRED_TEXT, canon, file_sha, load, save_new,
    sha, validate_proposal, check_not_under,
)

SCHEMA = "m6.1-adversarial-development-shadow-v1"
ROLES = ("REPAIR_PREDICTION", "DESIGN_DECISIVE_EXPERIMENT", "OPEN_ALTERNATE_BRANCH")
ATTACK_TYPES = ("COUNTERMODEL", "ALTERNATE_KNOWN_MECHANISM", "IDENTIFIABILITY_ATTACK")
REGIME = "research_idea:9cd359cab890711bf761"
RESIDENCE = "research_idea:b1c161e528f01392084b"
PATCHABLE = frozenset(REQUIRED_TEXT) - {"parent_idea_id", "local_id"}
ROLE_MUST_CHANGE = {
    ROLES[0]: {"quantitative_or_ordinal_prediction", "restricted_null", "local_falsifier", "heldout_test"},
    ROLES[1]: {"intervention", "observable", "heldout_test", "local_falsifier"},
    ROLES[2]: {"research_question", "mechanistic_hypothesis", "restricted_null", "quantitative_or_ordinal_prediction", "heldout_test"},
}


def assert_authority(obj: dict, label: str) -> None:
    for field in ("scientific_truth_authority", "novelty_certified"):
        if obj.get(field) is not False:
            raise ValueError(f"{label}: {field} must be false")
    if obj.get("production_selection_authority") not in (None, False):
        raise ValueError(f"{label}: production selection authority prohibited")


def _case_for_branch(branch: dict) -> dict:
    if branch.get("parent_intact") is not True or branch.get("generation_kind") != "SPECULATIVE_SHADOW_PROPOSAL_NOT_RESEARCH_IDEA_NODE":
        raise ValueError("M6 branch must be a speculative preserved-parent shadow")
    assert_authority(branch, "M6 branch")
    source = branch.get("revision") or branch.get("initial_proposal")
    if source is None or source.get("parent_idea_id") != branch.get("parent_idea_id"):
        raise ValueError("M6 source proposal missing or parent mismatch")
    validate_proposal(source, {branch["parent_idea_id"]})
    family = "REGIME_SPECTRAL" if branch["parent_idea_id"] == REGIME else ("RESIDENCE_DYNAMICS" if branch["parent_idea_id"] == RESIDENCE else "GENERAL_UNREVIEWED")
    case = {
        "parent_idea_id": branch["parent_idea_id"],
        "source_branch_id": branch["branch_id"],
        "source_phase": "REVISION" if branch.get("revision") else "INITIAL",
        "source_proposal_sha256": sha(source),
        "family": family,
        "source_proposal": source,
        "attacks": [],
        "calibration_status": "KNOWN_PHYSICS_WITNESS_NOT_SCIENTIFIC_VERDICT" if family != "GENERAL_UNREVIEWED" else "NO_DOMAIN_CALIBRATION_EXPERT_OR_MODEL_REQUIRED",
    }
    return case


def _attack(case: dict, seq: int, field: str, kind: str, observation: str,
            countermodel: str, witness: dict, repair: str, measurement: str,
            implication: str, local_scope: str) -> dict:
    source = case["source_proposal"]
    if field not in source or not isinstance(source[field], str):
        raise ValueError("attack target field invalid")
    key = f"m61_attack:{sha([case['source_proposal_sha256'], seq, kind])[:20]}"
    return {
        "attack_id": key, "target_field": field, "target_claim_exact": source[field],
        "attack_type": kind, "countermodel": countermodel,
        "witness": witness, "predicted_observation": observation,
        "implication": implication, "repair_target": repair,
        "independent_discriminator": measurement, "local_scope": local_scope,
        "evidence_status": "ANALYTIC_OR_CONSTRUCTED_COUNTERMODEL_ONLY_NOT_EMPIRICAL",
        "scientific_truth_authority": False, "novelty_certified": False,
    }


def _calibrated_attacks(case: dict) -> list[dict]:
    p=case["parent_idea_id"]
    attacks=[]
    if p == REGIME:
        # Strong mode-specific site response may cancel in the *observed ensemble ratio*.
        # (4*1.5 + 1*1)/(1*1.5 + 2*1) = 2, same as orientation-only prediction.
        attacks.append(_attack(case,1,"quantitative_or_ordinal_prediction","COUNTERMODEL",
            "Ensemble ratio 2.0 despite spectrally distinct local site spectra and differing weights; high heterogeneity need not force orientation-model error upward.",
            "Two emitting sites have band vectors (4,1) and (1,2); site weights 1.5 and 1.0. Their weighted band ratio equals an independent orientation-only prediction of 2.",
            {"site_band_vectors":[[4.,1.],[1.,2.]],"site_weights":[1.5,1.0],"orientation_only_ratio":2.,"ensemble_ratio":2.},
            "Make the high-S association a *model-conditional tendency*, not a monotonic or compulsory physical law; anticipate cancellation.",
            "Site-resolved spectral-response differences and independently characterized sampling weights; include null-cancellation cases in holdout.",
            "Disproves any universal implication from spectral/spatial heterogeneity alone to elevated observed orientation residual; does not disprove a calibrated conditional statistical effect.",
            "Two-band, two-site counterexample within tested orientation/field calibration envelope."))
        attacks.append(_attack(case,2,"heldout_test","IDENTIFIABILITY_ATTACK",
            "A fit can appear better with more model flexibility if the field proxy or nuisance terms were learned from held-out Raman outcomes.",
            "Broader known-physics model with mode-dependent transfer and hidden chemical-state contributions; near-field measurements alone may not isolate molecularly sampled transfer.",
            {"failure_mode":"heldout_target_leakage_or_nonidentifiable_optical_chemistry","numerical_result":None},
            "Freeze transfer calibration and nuisance fits before held-out band-ratio evaluation; compare predictive calibration and complexity-adjusted errors.",
            "Independent transfer-function calibrant, molecular coverage, chemical-state probe, blinded substrate/wavelength holdout.",
            "Out-of-sample superiority of a flexible EM model is not unique identification of field-selective causality.",
            "Prediction-vs-causal-attribution distinction, not a claim that all such measurements are impossible."))
        attacks.append(_attack(case,3,"mechanistic_hypothesis","ALTERNATE_KNOWN_MECHANISM",
            "Chemistry-dependent Raman polarizability could shift band ratios under the same substrate and optical transfer function.",
            "Resonance/charge-transfer-dependent changes in state-resolved Raman tensors compete with local EM mode selectivity.",
            {"alternate":"adsorption_or_charge_transfer_dependent_mode_intensities"},
            "Develop an orthogonal chemistry perturbation or marker; do not equate field-selective residuals with unique EM origin.",
            "Independent adsorption/oxidation-state spectroscopy and two reporter systems with contrasting chemical sensitivity.",
            "A positive orientation-only residual does not select between known EM and known chemical Raman contributions.",
            "Within the adsorbate/substrate/wavelength domain; chemical markers may be imperfect."))
    elif p == RESIDENCE:
        attacks.append(_attack(case,1,"quantitative_or_ordinal_prediction","COUNTERMODEL",
            "Both band intensities fluctuate, but every nonzero per-window ratio stays exactly 2.0 for arbitrary kinetics.",
            "Two adsorption states have band vectors (10,5) and (20,10). Both are proportional, so any nonnegative time-weighted mixture has ratio 2.",
            {"state_vectors":[[10.,5.],[20.,10.]],"mixture_weights":[0.2,0.8],"ratio_for_all_windows":2.0},
            "Require nonproportional state-resolved spectra before predicting kinetic changes in band-ratio dispersion; retain intensity covariance separately.",
            "Independent state-specific emission ratio and band-resolved covariance, plus held-out integration-window statistics.",
            "High field heterogeneity and changing exchange rates do not by themselves imply a band-ratio interaction.",
            "Positive-emission linear accumulation with common relative spectral response."))
        attacks.append(_attack(case,2,"restricted_null","ALTERNATE_KNOWN_MECHANISM",
            "The same observable autocovariance could come from blinking or photophysical two-state switching rather than adsorption-site exchange.",
            "A two-state blinking process with the same transition generator and state emission vectors as a two-state exchange process is observationally equivalent for the measured band-vector trajectory.",
            {"equivalence":"same_transition_generator_and_emission_map","second_order_statistics":"identical"},
            "Frame kinetics as an effective model unless an independent occupancy/exchange marker separates states.",
            "SERS-independent adsorption occupancy/kinetics channel and illumination-dependent blinking control with matched sampling.",
            "Time correlation alone cannot uniquely identify a residence-time mechanism; a broad dynamic known-physics null survives.",
            "Finite-state emission/transition equivalence, even for entire observed trajectories."))
        attacks.append(_attack(case,3,"heldout_test","IDENTIFIABILITY_ATTACK",
            "Distinct hidden-state dwell-time distributions can yield similar low-order covariance, so a good two-state Markov fit need not verify Markov residence dynamics.",
            "Semi-Markov state dwell distributions and multi-state mixtures may agree on limited second-order statistics while differing in higher-order lag dependence.",
            {"failure_mode":"finite_window_second_order_observational_equivalence"},
            "Add pre-registered higher-order statistics or residence-distribution tests as exploratory alternate branch, with independent kinetics.",
            "Held-out long/short integration windows, dwell-time survival curve, multi-lag cross-band temporal tests.",
            "A positive held-out covariance score is useful predictive performance, not microscopic state identification.",
            "Applies where noise and acquisition cadence limit temporal resolution."))
    return attacks


def prepare_challenges(developed_path: str | Path, out_dir: str | Path) -> dict:
    check_not_under(out_dir,[developed_path])
    original=load(developed_path)
    if (original.get("schema_version") != M6_SCHEMA or
        original.get("status") != "M6_DEVELOPMENT_SHADOW_COMPLETED_SCIENCE_UNREVIEWED"):
        raise ValueError("requires frozen completed M6 development output")
    assert_authority(original.get("authority", {}), "M6 envelope")
    branches=original.get("development_branches",[])
    if not branches or len(branches)!=original.get("branch_count"):
        raise ValueError("M6 branch count mismatch")
    cases=[_case_for_branch(b) for b in branches]
    ids=[c["parent_idea_id"] for c in cases]
    if len(ids)!=len(set(ids)):
        raise ValueError("M6.1 pilot requires one branch per distinct parent")
    for c in cases:
        c["attacks"]=_calibrated_attacks(c)
    result={"schema_version":SCHEMA, "status":"M61_CALIBRATED_CHALLENGES_PREPARED_SCIENCE_UNREVIEWED",
            "developed_sha256":file_sha(developed_path), "case_count":len(cases),
            "cases":cases, "authority":{"scientific_truth_authority":False,"novelty_certified":False,
                  "production_selection_authority":False,"ResearchIdea_population_modified":False}}
    save_new(Path(out_dir)/"M61_ADVERSARIAL_CASES.json",result)
    return result


def _validate_attack(a: dict, case: dict) -> None:
    required=("attack_id","target_field","target_claim_exact","attack_type","countermodel",
              "witness","predicted_observation","implication","repair_target","independent_discriminator","local_scope")
    if any(k not in a for k in required) or any(not a[k] for k in required):
        raise ValueError("missing attack fields")
    if a["attack_type"] not in ATTACK_TYPES or a["target_field"] not in case["source_proposal"]:
        raise ValueError("invalid attack type or target")
    if a["target_claim_exact"] != case["source_proposal"][a["target_field"]]:
        raise ValueError("attack must be grounded in the actual proposal text")
    if not isinstance(a["witness"],dict) or not a["witness"]:
        raise ValueError("attack must supply a structured countermodel witness")
    assert_authority(a,"attack")


def validate_cases(doc: dict) -> list[dict]:
    if doc.get("schema_version")!=SCHEMA or doc.get("status")!="M61_CALIBRATED_CHALLENGES_PREPARED_SCIENCE_UNREVIEWED":
        raise ValueError("invalid M6.1 cases envelope")
    assert_authority(doc.get("authority",{}),"cases")
    cases=doc.get("cases",[])
    if not cases or len(cases)!=doc.get("case_count"):
        raise ValueError("invalid case count")
    seen=set()
    for c in cases:
        if c["parent_idea_id"] in seen:raise ValueError("duplicate parent")
        seen.add(c["parent_idea_id"])
        if sha(c["source_proposal"])!=c["source_proposal_sha256"]:
            raise ValueError("case source proposal mutated")
        validate_proposal(c["source_proposal"],{c["parent_idea_id"]})
        for a in c["attacks"]:_validate_attack(a,c)
        ids=[a["attack_id"] for a in c["attacks"]]
        if len(ids)!=len(set(ids)):raise ValueError("duplicate attacks")
    return cases


def validate_delta(item: dict, case: dict, source_type: str) -> dict:
    allowed={"parent_idea_id","branch_role","attack_ids","changed_fields", "added_controls",
             "added_independent_variables", "added_assumptions", "fork_rationale",
             "scientific_truth_authority", "novelty_certified", "source_of_proposal"}
    if set(item)-allowed:raise ValueError("unexpected branch delta fields")
    if item.get("parent_idea_id")!=case["parent_idea_id"] or item.get("branch_role") not in ROLES:
        raise ValueError("branch role/parent mismatch")
    assert_authority(item,"branch delta")
    if item.get("source_of_proposal")!=source_type:
        raise ValueError("incorrect proposal source provenance")
    if not isinstance(item.get("fork_rationale"),str) or not item["fork_rationale"].strip():
        raise ValueError("fork rationale required")
    known={a["attack_id"] for a in case["attacks"]}
    ids=item.get("attack_ids")
    if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or not set(ids).issubset(known):
        raise ValueError("branch must cite its parent-specific grounded attack IDs")
    changes=item.get("changed_fields")
    if not isinstance(changes,dict) or set(changes)-PATCHABLE or not ROLE_MUST_CHANGE[item["branch_role"]].issubset(changes):
        raise ValueError("missing role-specific scientific changed fields")
    for name, value in changes.items():
        if not isinstance(value,str) or len(value.strip())<18:
            raise ValueError(f"too-short scientific change: {name}")
        if value==case["source_proposal"].get(name):
            raise ValueError("changed field exactly matches original")
    for name in ("added_controls","added_independent_variables","added_assumptions"):
        v=item.get(name,[])
        if not isinstance(v,list) or not all(isinstance(x,str) and x.strip() for x in v):
            raise ValueError("bad added list")
    result=copy.deepcopy(case["source_proposal"])
    result.update(changes)
    result["local_id"]={ROLES[0]:"repair_01",ROLES[1]:"experiment_01",ROLES[2]:"alternate_01"}[item["branch_role"]]
    result["source_of_proposal"]=source_type
    for field, add in (("controls","added_controls"),("independent_variables","added_independent_variables"),("assumptions","added_assumptions")):
        result[field]=list(dict.fromkeys([*result[field],*item.get(add,[])]))
    result["scientific_truth_authority"]=False;result["novelty_certified"]=False
    validate_proposal(result,{case["parent_idea_id"]})
    return result


def materialize_branches(cases_path: str | Path, deltas_path: str | Path, out_dir: str | Path) -> dict:
    check_not_under(out_dir,[cases_path,deltas_path])
    doc=load(cases_path); cases=validate_cases(doc)
    input_doc=load(deltas_path)
    if input_doc.get("schema_version")!=SCHEMA or input_doc.get("status")!="M61_DIVERGENT_DELTAS_PROPOSED_SCIENCE_UNREVIEWED":
        raise ValueError("bad deltas document")
    if input_doc.get("cases_sha256")!=file_sha(cases_path):raise ValueError("case lineage SHA mismatch")
    assert_authority(input_doc.get("authority",{}),"deltas document")
    by_parent={c["parent_idea_id"]:c for c in cases}
    deltas=input_doc.get("deltas",[])
    expected={(c["parent_idea_id"],r) for c in cases for r in ROLES}
    actual={(d.get("parent_idea_id"),d.get("branch_role")) for d in deltas}
    if len(deltas)!=len(expected) or actual!=expected:
        raise ValueError("exactly one of each branching role per source required")
    result=[]
    for delta in deltas:
        case=by_parent[delta["parent_idea_id"]]
        source=delta["source_of_proposal"]
        if source not in {"ANALYST_SEED","MODEL_GENERATED"}:raise ValueError("unsupported branch provenance")
        proposal=validate_delta(delta,case,source)
        result.append({"fork_id":"m61_fork:"+sha([case["source_branch_id"],delta["branch_role"],sha(proposal)])[:20],
                       "source_m6_branch_id":case["source_branch_id"],
                       "parent_idea_id":case["parent_idea_id"],
                       "source_proposal_sha256":case["source_proposal_sha256"],
                       "branch_role":delta["branch_role"],"attack_ids":delta["attack_ids"],
                       "fork_rationale":delta["fork_rationale"],
                       "proposal":proposal,"provenance":source,
                       "scientific_review_status":"PROPOSED_UNREVIEWED",
                       "countermodel_resolved":"NOT_ASSESSED","novelty_certified":False,
                       "scientific_truth_authority":False,"production_selection_authority":False})
    output={"schema_version":SCHEMA,"status":"M61_DIVERGENT_SCIENTIFIC_POPULATION_GENERATED_UNREVIEWED",
            "source_cases_sha256":file_sha(cases_path),"source_deltas_sha256":file_sha(deltas_path),
            "parent_count":len(cases),"fork_count":len(result),"forks":result,
            "metrics_non_scientific":{"roles_per_parent":len(ROLES),"novel_mechanisms_proved":0,
                         "causal_scientific_gain":"NOT_MEASURED","scientific_success":"NOT_CERTIFIED"},
            "authority":{"ResearchIdea_population_modified":False,"scientific_truth_authority":False,
                         "novelty_certified":False,"production_selection_authority":False}}
    save_new(Path(out_dir)/"M61_DIVERGENT_POPULATION.json",output)
    return output


def print_witness_checks() -> dict:
    regime=(4*1.5+1*1.0)/(1*1.5+2*1.0)
    residence_1=(10*.2+20*.8)/(5*.2+10*.8)
    residence_2=(10*.85+20*.15)/(5*.85+10*.15)
    if not all(abs(x-2.0)<1e-12 for x in (regime,residence_1,residence_2)):
        raise AssertionError("calibration physical counterexample violated")
    return {"status":"M61_EXACT_COUNTERMODEL_CALIBRATIONS_PASS", "regime_ratio":regime,
            "residence_ratio_1":residence_1,"residence_ratio_2":residence_2,
            "scientific_truth_authority":False}


def critic_prompt(case: dict) -> tuple[str,str]:
    system=("You are an adversarial scientific physicist, not a novelty certifier. Challenge the EXACT "
            "provided proposal rather than giving generic advice. Search aggressively; claim conservatively. "
            "Return one JSON object with key 'attacks' containing 1-3 attacks; preserve exact target_claim_exact "
            "from source_proposal[target_field], attack_type must be COUNTERMODEL, ALTERNATE_KNOWN_MECHANISM, "
            "or IDENTIFIABILITY_ATTACK. Include explicit countermodel, falsifiable implication, "
            "separating measurement and limitation. Never invent empirical results; science authority=false. "
            "Do not claim to refute broad known physics by refuting restricted M0. "
            "Do not imply that constructed witnesses refute every qualified conditional prediction.")
    schema={"required_fields":("attack_id","target_field","target_claim_exact","attack_type","countermodel",
               "witness","predicted_observation","implication","repair_target","independent_discriminator",
               "local_scope","scientific_truth_authority","novelty_certified")}
    return system,json.dumps({"case":case,"schema":schema},ensure_ascii=False)


def branch_prompt(case: dict) -> tuple[str,str]:
    system=("You develop a DIVERGENT scientific research population from specific countermodels. "
            "Produce exactly three deltas under key 'deltas' with branch_role REPAIR_PREDICTION, "
            "DESIGN_DECISIVE_EXPERIMENT, OPEN_ALTERNATE_BRANCH. For each cite actual case attack_ids. "
            "Respect all role-required changed_fields, keep parent id, never claim new physics or evidence. "
            "Do not merely add verbose controls: for alternate change the central research question and mechanism. "
            "No claims of certification. Only JSON; output `changed_fields` (text values), `added_controls`, "
            "`added_independent_variables`, `added_assumptions`, `fork_rationale`, `attack_ids`, `branch_role`, "
            "`parent_idea_id`, `scientific_truth_authority=false`, `novelty_certified=false`, "
            "`source_of_proposal=MODEL_GENERATED`. Do not repeat unchanged original proposal fields.")
    return system,json.dumps({"case":case,"role_must_change":{k:sorted(v) for k,v in ROLE_MUST_CHANGE.items()},
                             "patchable_fields":sorted(PATCHABLE)},ensure_ascii=False)

# These are fixed analyst-constructed calibration proposals (not autonomously discovered results).
# They are deliberately concise *deltas* applied to the unmodified M6 source proposal.
_SEED_DELTAS = {
    REGIME: {
        "REPAIR_PREDICTION": {
            "quantitative_or_ordinal_prediction": "The preregistered full two-site forward model predicts held-out ratios conditional on independently estimated site spectral tensors and population weights. High measured field heterogeneity alone does NOT imply a positive orientation residual: site contributions may cancel. Report held-out uncertainty rather than assert a compulsory positive S-response.",
            "restricted_null": "Restricted orientation-only predictive null on a preregistered wavelength/substrate domain, distinct from the broader EM plus chemical-response model. A cancelled ensemble ratio is permitted despite spectrally selective local sites.",
            "local_falsifier": "In the frozen measured domain, locally reject the proposed calibrated conditional residual model only if its independently parametrized predictions are reproducibly miscalibrated on held-out conditions while a prespecified alternative predicts those outcomes; a null orientation residual alone does not falsify local selectivity.",
            "heldout_test": "Pre-register sites whose band-specific transfer functions are independently calibrated, including a cancellation configuration predicted to have high heterogeneity but near-zero aggregate residual. Fit no case-specific spectral weights on held-out data; predict ratios and confidence intervals across new cap thickness and laser wavelengths.",
        },
        "DESIGN_DECISIVE_EXPERIMENT": {
            "intervention": "Hold substrate morphology and excitation transfer approximately fixed. Introduce two chemically compatible reporter populations of distinguishable band patterns with independently assayed occupancy, and vary their surface fractions in a randomized blocked preparation to test predicted weighted cancellation and inversion of site contributions.",
            "observable": "Joint molecular occupancy map, calibrated band-specific optical transfer estimates, reporter-resolved Raman band-vector and orientation probes, full uncertainty; compare predicted mixture ratios against same-region blind repeats.",
            "heldout_test": "Calibrate state-specific Raman response and coverage without the target mixtures; prospectively predict band ratios for held-out mixture weights, including a condition where two strongly mode-selective sites cancel in aggregate. Do not re-estimate weights from target band ratios.",
            "local_falsifier": "If blinded mixture ratios reproducibly depart from independently calibrated weighted-site predictions beyond preregistered uncertainty while a chemistry-dependent alternative accounts for them, the specified mixing model fails locally; failure does not disprove the existence of mode-specific EM enhancement.",
        },
        "OPEN_ALTERNATE_BRANCH": {
            "research_question": "Could substrate-dependent adsorption/charge-transfer states, rather than frequency-selective electromagnetic weighting, explain residual mode-dependent SERS ratios after controlling orientation and independently calibrated optical transfer?",
            "mechanistic_hypothesis": "Multiple adsorption or charge-transfer states may alter band-specific Raman polarizability; a field-calibrated residual that tracks independent chemical-state markers may signal chemical-enhancement heterogeneity without any novel fundamental SERS physics.",
            "restricted_null": "An EM-only plus independently measured orientation response model with fixed molecular Raman tensors over adsorption conditions; broad known-physics alternatives already permit chemistry-modified polarizability.",
            "quantitative_or_ordinal_prediction": "At independently matched optical transfer and orientation, a controlled oxidation or ligand perturbation may produce a band-selective ratio change that covaries with an independent chemical-state measurement; the sign and size must be predicted from a separately calibrated state-specific response model rather than asserted universally.",
            "heldout_test": "Randomize chemical-state perturbations at otherwise matched substrate and laser conditions; calibrate independent adsorption-state markers and predict held-out band-ratio vectors versus an EM-only baseline and a broad EM-plus-chemical model without fitting on held-out outcomes.",
        },
    },
    RESIDENCE: {
        "REPAIR_PREDICTION": {
            "quantitative_or_ordinal_prediction": "For a two-state stationary exchange with independently determined emissions, the band-vector covariance decays at the measured exchange rate, but the ratio variance response depends on state spectral nonproportionality. If all state vectors are proportional, every nonzero per-window band ratio stays constant even when total intensity fluctuates.",
            "restricted_null": "The baseline includes established dynamic adsorption, photophysical switching and chemical-enhancement fluctuations, not merely static orientation. A kinetic effect on temporal covariance is expected under existing physics and does not identify adsorption residence uniquely.",
            "local_falsifier": "Locally refute the *particular independently parametrized Markov observation model* if held-out cross-band covariance and window statistics repeatedly fall outside calibrated bounds while preregistered alternatives fit; a zero ratio effect with proportional state spectra is an expected null outcome.",
            "heldout_test": "Use independent emission ratios and kinetic rates to preselect one near-proportional state pair and one nonproportional state pair; predict both windowed intensity covariance and ratio distributions at unseen integration times without retuning hidden state emissions.",
        },
        "DESIGN_DECISIVE_EXPERIMENT": {
            "intervention": "Use a reversible concentration-jump or isotope-label exchange to perturb adsorption kinetics while separately varying illumination power to modulate blinking. Keep direct optical and orientation controls, measure both perturbations within matched field-heterogeneity strata.",
            "observable": "Parallel SERS-independent exchange-kinetic signal, intensity vector cross-correlation, power-dependent blinking markers, per-window band ratio distribution and kinetic response under perturbation reversal.",
            "heldout_test": "Preregister the kinetics intervention separately from a laser-power blinking control, fit both dynamical observation models only to training acquisitions, then predict held-out temporal spectra, dwell distributions and integration-window covariance for the crossed interventions.",
            "local_falsifier": "If independently constrained residence/exchange models fail on held-out crossed perturbations but an independently parameterized photoblinking model succeeds, the adsorption-specific interpretation lacks local support; a correlation time alone is not evidence for adsorption.",
        },
        "OPEN_ALTERNATE_BRANCH": {
            "research_question": "Do non-exponential adsorption residence-time distributions produce reproducible integration-window scaling of time-resolved SERS band-vector statistics beyond a preregistered stationary two-state Markov baseline?",
            "mechanistic_hypothesis": "A semi-Markov adsorption population with broad or multi-timescale dwell distributions may produce nonexponential correlations and finite-window scaling distinct from a single-rate Markov model, but both are known stochastic dynamics and require independent molecular-state constraints.",
            "restricted_null": "Stationary two-state Markov exchange with exponential dwell times and independently calibrated emissions; the broad known-physics family includes semi-Markov residence, blinking, and multistate photo/chemical dynamics.",
            "quantitative_or_ordinal_prediction": "Given a preregistered dwell-time survival distribution estimated independently, integration-window covariance and multi-lag cross-correlations should follow its derived nonexponential prediction on held-out acquisition windows; simple exponential Markov predictions may fail conditionally, without certifying a new mechanism.",
            "heldout_test": "Acquire an independent kinetic dwell-time survival curve, fit an exponential Markov baseline and a constrained semi-Markov model on training windows, and predict held-out long-lag correlations and multi-window band-vector dispersion, also testing a photoblinking alternative.",
        },
    },
}


def build_analyst_seed_deltas(cases_path: str | Path, out_dir: str | Path) -> dict:
    check_not_under(out_dir,[cases_path])
    cases=validate_cases(load(cases_path))
    deltas=[]
    for case in cases:
        pid=case["parent_idea_id"]
        if pid not in _SEED_DELTAS or len(case["attacks"])<3:
            raise ValueError(f"No fixed analyst physics-calibration scaffold for {pid}. Use model opt-in.")
        attack_ids=[a["attack_id"] for a in case["attacks"]]
        for index,role in enumerate(ROLES):
            delta={"parent_idea_id":pid,"branch_role":role,
                   "attack_ids":attack_ids if role==ROLES[0] else [attack_ids[min(index,2)]],
                   "changed_fields":copy.deepcopy(_SEED_DELTAS[pid][role]),
                   "added_controls":[],"added_independent_variables":[],"added_assumptions":[],
                   "fork_rationale":("Analyst-calibrated scientific direction: "+role+
                        ". These are candidate research specifications, not findings."),
                   "scientific_truth_authority":False,"novelty_certified":False,
                   "source_of_proposal":"ANALYST_SEED"}
            validate_delta(delta,case,"ANALYST_SEED")
            deltas.append(delta)
    doc={"schema_version":SCHEMA,"status":"M61_DIVERGENT_DELTAS_PROPOSED_SCIENCE_UNREVIEWED",
         "cases_sha256":file_sha(cases_path),"deltas":deltas,
         "generation_mode":"OFFLINE_ANALYST_CALIBRATION_NOT_AUTONOMOUS_DISCOVERY",
         "authority":{"scientific_truth_authority":False,"novelty_certified":False,
                      "production_selection_authority":False}}
    save_new(Path(out_dir)/"M61_ANALYST_DELTAS.json",doc)
    return doc


def merge_model_attacks(cases_path: str | Path, attacks_path: str | Path, out_dir: str | Path) -> dict:
    check_not_under(out_dir,[cases_path,attacks_path])
    base=load(cases_path);cases=validate_cases(base)
    source=load(attacks_path)
    if source.get("schema_version")!=SCHEMA or source.get("status")!="M61_MODEL_ATTACKS_PROPOSED_SCIENCE_UNREVIEWED":
        raise ValueError("invalid model attack envelope")
    assert_authority(source.get("authority",{}),"attack envelope")
    if source.get("cases_sha256")!=file_sha(cases_path):raise ValueError("model attacks refer to different cases SHA")
    sets=source.get("attack_sets",[])
    if len(sets)!=len(cases):raise ValueError("model attack parent count mismatch")
    index={c["parent_idea_id"]:c for c in cases}
    seen=set()
    for record in sets:
        pid=record["parent_idea_id"]
        if pid not in index or pid in seen:raise ValueError("model attack duplicate/unknown parent")
        seen.add(pid)
        case=index[pid]
        if record.get("source_proposal_sha256")!=case["source_proposal_sha256"]:
            raise ValueError("model attacks source proposal SHA mismatch")
        attacks=record.get("attacks",[])
        if not isinstance(attacks,list) or not 1<=len(attacks)<=3:
            raise ValueError("model attacks require 1..3 per source")
        for j, attack in enumerate(attacks):
            attack["attack_id"]="m61_model_attack:"+sha([pid,case["source_proposal_sha256"],j,attack])[0:20]
            _validate_attack(attack,case)
        case["attacks"].extend(attacks)
        case["calibration_status"]="MODEL_HYPOTHESIS_PLUS_ANALYST_CALIBRATION_UNREVIEWED"
    base["upstream_model_attacks_sha256"]=file_sha(attacks_path)
    base["cases"]=[index[c["parent_idea_id"]] for c in cases]
    validate_cases(base)
    save_new(Path(out_dir)/"M61_ENRICHED_CASES.json",base)
    return base


def export_evolution_feedback(cases_path: str | Path, population_path: str | Path,
                              out_dir: str | Path) -> dict:
    """Produce untrusted SIS-compatible *input context*, not a ResearchIdeaNode.

    A future generation adapter may choose to use these hints, but this function
    does not import or mutate SIS or register children in its population.
    """
    check_not_under(out_dir,[cases_path,population_path])
    cases=validate_cases(load(cases_path))
    cindex={c["parent_idea_id"]:c for c in cases}
    pop=load(population_path)
    if pop.get("schema_version")!=SCHEMA or pop.get("status")!="M61_DIVERGENT_SCIENTIFIC_POPULATION_GENERATED_UNREVIEWED":
        raise ValueError("requires produced speculative population")
    assert_authority(pop.get("authority",{}),"population")
    if pop.get("source_cases_sha256")!=file_sha(cases_path):
        raise ValueError("population and cases SHA mismatch")
    hints=[];seen=set()
    for f in pop["forks"]:
        pid=f["parent_idea_id"]
        if pid not in cindex or f["branch_role"] not in ROLES:
            raise ValueError("fork has unknown source or role")
        if f["scientific_review_status"]!="PROPOSED_UNREVIEWED" or f["countermodel_resolved"]!="NOT_ASSESSED":
            raise ValueError("cannot promote unsanctioned fork status")
        assert_authority(f,"fork")
        key=(pid,f["branch_role"])
        if key in seen:raise ValueError("duplicate evolutionary context role")
        seen.add(key)
        p=f["proposal"]
        if p["parent_idea_id"]!=pid or f["source_proposal_sha256"]!=cindex[pid]["source_proposal_sha256"]:
            raise ValueError("fork source mutated")
        all_attacks={x["attack_id"]:x for x in cindex[pid]["attacks"]}
        if not set(f["attack_ids"]).issubset(all_attacks):
            raise ValueError("unknown attack provenance in fork")
        hints.append({"suggested_parent_idea_id":pid,
                      "source_m6_branch_id":f["source_m6_branch_id"],
                      "speculative_fork_id":f["fork_id"],
                      "exploration_operator_hint":f["branch_role"],
                      "research_question":p["research_question"],
                      "possible_mechanism":p["mechanistic_hypothesis"],
                      "conditional_prediction":p["quantitative_or_ordinal_prediction"],
                      "independent_measurements":p["independent_variables"],
                      "decisive_holdout":p["heldout_test"],
                      "local_falsifier":p["local_falsifier"],
                      "unresolved_attacks":[{"attack_id":a,"countermodel":all_attacks[a]["countermodel"],
                           "independent_discriminator":all_attacks[a]["independent_discriminator"]} for a in f["attack_ids"]],
                      "source_of_proposal":f["provenance"],
                      "idea_identity_verified":False,
                      "scientific_truth_authority":False,"novelty_certified":False,
                      "production_selection_authority":False})
    if len(hints)!=pop["fork_count"]:raise ValueError("fork count mismatch")
    result={"schema_version":SCHEMA,"status":"M61_SPECULATIVE_EVOLUTION_CONTEXT_NOT_IMPORTED_INTO_SIS",
            "source_cases_sha256":file_sha(cases_path),
            "source_population_sha256":file_sha(population_path),
            "hint_count":len(hints),"evolution_hints":hints,
            "authority":{"ResearchIdea_population_modified":False,"SIS_generation_executed":False,
                         "scientific_truth_authority":False,"novelty_certified":False,
                         "production_selection_authority":False}}
    save_new(Path(out_dir)/"M61_EVOLUTION_FEEDBACK_HINTS.json",result)
    return result
