"""M11: source-locked adversarial elimination + new-search tasks, never novelty certification.

Pure stdlib. Counterexamples are model-internal, not measurements or evidence.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
from zipfile import ZipFile

SCHEMA = 'm11-science-falsification-search-v1'
EXPECTED = {'SYMMETRY_BREAKING', 'HETEROGENEITY_INTERACTION', 'MEASUREMENT_LIMIT'}
EVIDENCE_AUTHORITY = False


def canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json_zip(path: Path, member: str) -> dict:
    with ZipFile(path) as z:
        return json.loads(z.read(member))


def write_new(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'No overwrites: {path}')
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        path.write_text(str(content), encoding='utf-8')


def physics_counterexamples() -> dict:
    """Exact illustrative calculations; no stochastic fit or empirical identification."""
    # B1: z-dependent intercept without any z*u interaction.
    base = {'bright': .2, 'dark': .8}
    on = {z: r * 2.0 for z, r in base.items()}
    raw_hazard_difference = on['bright'] - on['dark']
    did_z_intercepts_only = (math.log(on['bright']) - math.log(base['bright'])
                             - math.log(on['dark']) + math.log(base['dark']))
    # B2: no within-hotspot z-field interaction, but photostate correlates
    # with a hidden hotspot class whose field slope varies by class.
    weights = {'bright': .85, 'dark': .20}
    def hazard(z: str, u: int) -> float:
        w = weights[z]
        return w * (.12 * math.exp(2.0 * u)) + (1.0 - w) * (.45 * math.exp(.0 * u))
    h = {z: {str(u): hazard(z, u) for u in (0, 1)} for z in weights}
    d_hidden = (math.log(h['bright']['1']/h['bright']['0'])
                - math.log(h['dark']['1']/h['dark']['0']))
    # Under explicit parametric causal model with z in {0,1},
    # beta u + gamma z u, conditional on fully observed confounders:
    modeled_interaction = .6
    # A: symmetric equilibrium correlation C(t)=exp(-|t|); unequal channel delay.
    tau, delay = .5, .2
    instrument_artifact = math.exp(-abs(tau-delay)) - math.exp(-abs(tau+delay))
    # C: U->N->Y, conditioning on N erases valid mediated total causal response.
    a, b = .7, 1.4
    return {
        'B_INTERCEPT_ONLY_NULL': {
            'description': 'z-dependent exchange base rates yield bright-dark hazard difference with gamma=0',
            'parameters': {'k_b_0':base['bright'], 'k_d_0':base['dark'], 'field_multiplier':2.0},
            'field_on_hazard_difference':raw_hazard_difference,
            'log_field_difference_in_differences':did_z_intercepts_only,
            'conclusion': 'ORIGINAL_B_HAZARD_DIFFERENCE_NOT_SPECIFIC_FOR_GAMMA',
        },
        'B_HIDDEN_HOTSPOT_MIXTURE_NULL': {
            'description': 'within each hidden hotspot class gamma=0; differing class composition by measured z induces aggregate nonzero DiD at event onset',
            'hidden_class_weights_by_z':weights,
            'observed_initial_hazard':h,
            'aggregate_log_DiD':d_hidden,
            'gamma_within_every_hotspot_class':0.0,
            'caution':'Initial hazard only: mixture survival changes effective weights over dwell age. This is a counterexample to naive aggregated DiD, not a calibrated experiment.',
            'conclusion':'NONZERO_AGGREGATE_DID_NOT_IDENTIFICATION_OF_GAMMA',
        },
        'B_IDENTIFIABLE_PARAMETRIC_REFERENCE': {
            'description':'Under conditional log-hazard with fully measured z,u and fixed age/history, gamma produces DiD=gamma*(z_b-z_d)*(u1-u0).',
            'assumed_gamma':modeled_interaction,
            'model_DiD_for_unit_contrasts':modeled_interaction,
            'conclusion':'CONDITIONAL_TEST_ONLY_NOT_EMPIRICAL_EVIDENCE',
        },
        'A_INSTRUMENT_DELAY_NULL': {
            'description':'intrinsic equilibrium exp(-abs(lag)) correlation observed through unequal channel delay',
            'lag':tau,'delay':delay,
            'intrinsic_A':0., 'observed_A':instrument_artifact,
            'conclusion':'OBSERVED_A_NEQ_0_NOT_SUFFICIENT_FOR_NON_EQUILIBRIUM',
        },
        'C_MEDIATOR_CONDITIONING_NULL': {
            'description':'U->N->Y with N=.7U+eta and Y=1.4N+epsilon; true causal total response versus conditional direct response holding N fixed',
            'total_causal_slope':a*b, 'controlled_direct_slope_holding_occupancy':0.,
            'conclusion':'CONDITIONING_ON_OCCUPANCY_ELIMINATES_MEDIATED_SIGNAL',
        },
    }


def _source_candidates(packet: dict, raw: dict) -> dict:
    candidates = {}
    for row in packet.get('candidates', []):
        if row.get('task_id') in EXPECTED:
            if row['task_id'] in candidates:
                raise ValueError('Duplicate M10 candidate task')
            candidates[row['task_id']] = row
    if raw.get('task_id') == 'MEASUREMENT_LIMIT':
        candidates['MEASUREMENT_LIMIT'] = raw['scientific_content_raw_unmodified']
    if set(candidates) != EXPECTED:
        raise ValueError(f'M10 paid source candidates missing: {EXPECTED-set(candidates)}')
    if packet.get('novelty_certified') is not False:
        raise ValueError('M10 source must not claim certified novelty')
    if raw.get('not_promoted_to_native_parsed_candidate') is not True:
        raise ValueError('M10 C raw-status must be respected')
    if packet.get('generated_candidate_count') != 2:
        raise ValueError('Original M10 generated count must remain 2, not silently changed')
    return candidates


def _claims() -> list[dict]:
    return [
        {'claim_id':'A1', 'candidate':'SYMMETRY_BREAKING', 'claim':'Stationary detailed-balance time-reversal reciprocity gives zero intrinsic lag cross-covariance antisymmetry.',
         'disposition':'KNOWN_PHYSICS_NOT_ACADEMIC_NOVELTY', 'basis':'PRL 2023 theorem; this is an established general phenomenon',
         'failure_use':'Do not claim discovery of time-antisymmetric correlations.'},
        {'claim_id':'A2', 'candidate':'SYMMETRY_BREAKING', 'claim':'A nonzero observed antisymmetry uniquely identifies a molecular adsorption cycle.',
         'disposition':'REJECT_UNIDENTIFIABLE_ATTRIBUTION', 'basis':'Detector delay and other nonequilibrium photophysical sources can generate analogous signals',
         'failure_use':'Search for independent molecular state reporter or a different observation invariant.'},
        {'claim_id':'A3', 'candidate':'SYMMETRY_BREAKING', 'claim':'Reversal-calibrated lag antisymmetry in a specific SERS molecular-state protocol is an unreported application.',
         'disposition':'PRIOR_ART_UNRESOLVED', 'basis':'2007 SERS isotopologue 2D correlation and 2023 theoretical asymmetry are related but not an exact literature identity proof',
         'failure_use':'Exact protocol-level literature retrieval. If found, reject novelty claim.'},
        {'claim_id':'B1', 'candidate':'HETEROGENEITY_INTERACTION', 'claim':'Different bright/dark occupancy hazards identify photostate-field interaction gamma != 0.',
         'disposition':'REJECT_INVALID_DISCRIMINATOR', 'basis':'Explicit z-dependent intercept counterexample gamma=0',
         'failure_use':'Discard original hazard-contrast as gamma evidence. Do not back-count the corrected contrast as original success.'},
        {'claim_id':'B2', 'candidate':'HETEROGENEITY_INTERACTION', 'claim':'Nonzero aggregate field-by-photostate difference-in-differences identifies within-hotspot interaction.',
         'disposition':'REJECT_WITHOUT_LATENT_CLASS_CONTROLS', 'basis':'Explicit gamma=0 hidden-hotspot mixture counterexample produces nonzero aggregate DiD',
         'failure_use':'Require within-hotspot independently measured state and field; otherwise register nonidentifiability.'},
        {'claim_id':'B3', 'candidate':'HETEROGENEITY_INTERACTION', 'claim':'A photostate-dependent local-field coefficient of a hotspot occupancy-transition hazard exists and is newly predictive.',
         'disposition':'MECHANISM_UNTESTED_AND_NOVELTY_UNRESOLVED', 'basis':'Plausible conditional model, but independent measurement and exhaustive adjacent-domain prior art absent',
         'failure_use':'Preserve as speculative alternate search context only, not a surviving factual hypothesis.'},
        {'claim_id':'C1', 'candidate':'MEASUREMENT_LIMIT', 'claim':'Conditioning on occupancy provides an unbiased test of all occupancy-mediated optical response.',
         'disposition':'REJECT_INVALID_CAUSAL_CONDITIONING', 'basis':'In U->N->Y model, conditional direct effect can be zero while total causal effect is nonzero',
         'failure_use':'Separate direct and mediated transfer; reject the original conditioning identity.'},
        {'claim_id':'C2', 'candidate':'MEASUREMENT_LIMIT', 'claim':'A held-out predictive-score boundary is by itself a new physical regime transition.',
         'disposition':'REJECT_NOVELTY_AS_STATED', 'basis':'A performance criterion does not constitute a new physical threshold law; modulation/system ID are established',
         'failure_use':'Do not call a statistical detection threshold a new physics regime.'},
        {'claim_id':'C3', 'candidate':'MEASUREMENT_LIMIT', 'claim':'A specific previously unreported independent-assay-corrected SERS identifiability boundary exists.',
         'disposition':'PRIOR_ART_UNRESOLVED', 'basis':'Specific conditional physical consequence and measurement protocol not established',
         'failure_use':'Claim-level exact literature comparison before any novelty promotion.'},
    ]


def _tasks() -> list[dict]:
    return [
        {'task_id':'M11_ALTERNATE_MECHANISM', 'role':'REPLACE_REJECTED_DISCRIMINATOR',
         'starting_failures':['B1','B2'],
         'research_target':'Find a scientifically distinct SERS mechanistic explanation or predicted signature; do NOT merely rename gamma or patch the old brightness contrast.',
         'strong_nulls':['hidden-hotspot mixture with heterogeneous field slopes and no within-hotspot z*u interaction','z-dependent baseline hazard without interaction','thermal, charge-transfer, morphology, selection and timing confounds'],
         'required_scientific_delta':'Offer a falsifiable signature where a fully specified plausible null and proposed mechanism predict different held-out results. If no defensible delta, abstain.'},
        {'task_id':'M11_NEW_OBSERVABLE', 'role':'NON_EQUILIBRIUM_ALTERNATE',
         'starting_failures':['A1','A2','A3'],
         'research_target':'Search beyond generic time asymmetry: a novel SERS-specific observable, intervention, or conditional response robust to detector lag and nonequilibrium photophysical alternatives.',
         'strong_nulls':['established cross-correlation asymmetry theory','different detector response functions','non-adsorption nonequilibrium photophysics'],
         'required_scientific_delta':'Do not claim time-asymmetry theorem is new. Specify exactly what new conditional prediction a strong null cannot reproduce.'},
        {'task_id':'M11_OPEN_SERS_FRONTIER', 'role':'DIVERSITY_NOT_REPAIR',
         'starting_failures':['C1','C2','C3'],
         'research_target':'Explore another SERS scientific axis distinct from dwell/covariance/field-gated desorption, e.g. mode-specific orientation, vibrational energy redistribution, or surface chemical selectivity. No preset positive mechanism.',
         'strong_nulls':['standard |E|^4 electromagnetic SERS enhancement and Raman tensor projection','known chemical enhancement, field heterogeneity, orientation/morphology averaging','generic modulation spectroscopy and system identification'],
         'required_scientific_delta':'Derive a risky conditional prediction with explicit observables, a physical null and a possible experiment. Abstain if all proposals reduce to established physics.'},
    ]


def build_frozen_experiment(packet_zip: Path, review_zip: Path) -> dict:
    packet=read_json_zip(packet_zip,'M10_NOVEL_HYPOTHESIS_CANDIDATE_PACKET.json')
    raw=read_json_zip(review_zip,'M10_MEASUREMENT_LIMIT_RAW_FOR_SCIENCE_ONLY.json')
    verdicts=read_json_zip(review_zip,'M10_ADVERSARIAL_VERDICTS.json')
    candidates=_source_candidates(packet,raw)
    if verdicts.get('novelty_certified') is not False:
        raise ValueError('review contains novelty certification')
    claims=_claims(); tasks=_tasks()
    payload={
        'schema_version':SCHEMA,
        'status':'SCIENTIFIC_CHALLENGE_COMPLETE_NEW_IDEAS_NOT_YET_GENERATED',
        'source_sha256':{'m10_packet':file_sha(packet_zip),'m10_review':file_sha(review_zip)},
        'source_g6_report_id':packet.get('source_g6_report_id'),
        'source_candidate_count_as_reported':packet['generated_candidate_count'],
        'source_candidate_ids':{t:c.get('candidate_id',f'RAW_UNPROMOTED_{t}') for t,c in candidates.items()},
        'C_is_unpromoted_raw':True,
        'calculations':physics_counterexamples(),
        'claim_dispositions':claims,
        'search_tasks':tasks,
        'search_context_source':{
            t:{k:v for k,v in c.items() if k in ('title','conceptual_delta','novelty_atom','mechanism','risk_of_known_physics_reexpression')}
            for t,c in candidates.items()},
        'science_policy':{
            'do_not_rescue_rejected_claim':True,
            'new_revision_is_new_candidate':True,
            'nonidentifiability_is_not_mechanism_falsification':True,
            'novelty_certified':False,
            'scientific_truth_authority':False,
            'positive_premise_authority':False,
            'canonical_graph_mutated':False,
            'official_generation_executed':False,
            'SERS_scope_only':True,
        },
        'literature_boundaries':[
            {'doi':'10.1103/PhysRevLett.131.077101','scope':'General equilibrium time-correlation reciprocity and nonequilibrium asymmetry already known'},
            {'doi':'10.1021/ja077243c','scope':'SERS isotopologue cross-correlation and hotspot exchange already explored'},
            {'doi':'10.1038/ncomms10545','scope':'Plasmon-mediated direct charge excitation and reaction kinetics established'},
            {'doi':'10.1038/s41467-024-51000-w','scope':'Plasmon-induced single-molecule switching rates under junction controls established'},
        ],
    }
    payload['experiment_id']='m11:'+digest(payload)[:20]
    return payload


def make_prompt(cfg: dict, task: dict) -> dict:
    if cfg.get('schema_version')!=SCHEMA or cfg['science_policy']['novelty_certified'] is not False:
        raise ValueError('Invalid science authority boundary')
    system='''You are an adversarial scientific hypothesis generator searching SERS research space.\n
GOAL: propose one DIFFERENT scientific ResearchIdea after specific prior ideas failed null tests. You are not tasked with saving any previous hypothesis. It is acceptable to abstain. Speculative mechanisms are allowed; empirical truth and academic novelty are not certified.\n
STRICT: The explicit counterexamples are mathematics, not SERS measurements. Prior art is search context only. Do not invent citations, data, or experimentally demonstrated results. Do not state lack of search results proves novelty.\n
A candidate must expose its causal assumptions, physical null, mathematically discriminating predicted observation, measurement requirements, and falsification scope. If null can fit equally well, state NON_IDENTIFIABLE, not success. Do not post-hoc add conditions and count it as original theory surviving.\n
Return ONE JSON object (not Markdown) with keys: task_id, disposition (PROPOSE or ABSTAIN), idea_title, conceptual_change, mechanistic_model, equations, strongest_competing_null, differential_prediction, independent_measurements, experiment_and_controls, falsification_condition, observational_equivalence_risks, prior_art_queries, limits, novelty_certified (false). JSON arrays or dicts acceptable for equations and predictions. In abstention, set disposition=ABSTAIN and write a concise explanation in limits; other fields may be empty.'''
    data={
        'task_id':task['task_id'], 'research_target':task['research_target'], 'role':task['role'],
        'challenge_claim_ids':task['starting_failures'],
        'disconfirmed_or_unidentified_claims':[c for c in cfg['claim_dispositions'] if c['claim_id'] in task['starting_failures']],
        'model_internal_counterexamples':cfg['calculations'],
        'known_prior_art_bounds':cfg['literature_boundaries'],
        'strong_nulls':task['strong_nulls'],
        'required_scientific_delta':task['required_scientific_delta'],
        'source_ideas_search_context_only':cfg['search_context_source'],
        'max_candidates':1, 'novelty_claim_permission':False,
    }
    return {'task_id':task['task_id'], 'system':system, 'user':json.dumps(data,indent=2,ensure_ascii=False,sort_keys=True)}


def decode_raw(raw: str) -> Any:
    txt=raw.strip()
    if txt.startswith('```'):
        txt='\n'.join(txt.splitlines()[1:-1]).strip()
    return json.loads(txt)


def audit_generated(raw: str, task_id: str) -> dict:
    try:
        d=decode_raw(raw)
    except Exception as exc:
        return {'status':'UNPARSED_RAW_PRESERVED', 'problem':f'{type(exc).__name__}:{exc}', 'task_id':task_id}
    if not isinstance(d,dict) or d.get('task_id')!=task_id:
        return {'status':'SCHEMA_OR_IDENTITY_MISMATCH', 'task_id':task_id}
    if d.get('novelty_certified') is not False:
        return {'status':'UNAUTHORIZED_NOVELTY_CLAIM_IN_GENERATION', 'task_id':task_id}
    if d.get('disposition')=='ABSTAIN':
        return {'status':'ABSTAINED_NO_HYPOTHESIS', 'task_id':task_id}
    keys=['conceptual_change','mechanistic_model','equations','strongest_competing_null','differential_prediction',
          'independent_measurements','falsification_condition','observational_equivalence_risks']
    miss=[k for k in keys if not d.get(k)]
    if miss:
        return {'status':'SCIENTIFIC_CONTENT_INCOMPLETE_NO_PROMOTION', 'task_id':task_id,'missing':miss}
    return {'status':'PROPOSED_UNREVIEWED_NOT_NOVELTY_CERTIFIED','task_id':task_id,
            'review_required':True,'candidate':d}


def render_report(cfg: dict) -> str:
    x=cfg['calculations'];b=x['B_HIDDEN_HOTSPOT_MIXTURE_NULL']
    claims=cfg['claim_dispositions']
    lines=[
        '# M11 SERS — Adversarial Claim Elimination and Search Renewal',
        '',
        'This report contains **analytical counterexamples, not empirical results or academic novelty certification**.',
        '',
        '## B — Photostate × field',
        f"- With γ=0, z-dependent intercepts produce nonzero bright–dark hazard difference {x['B_INTERCEPT_ONLY_NULL']['field_on_hazard_difference']:.6g}, while matched log-field difference-in-differences is {x['B_INTERCEPT_ONLY_NULL']['log_field_difference_in_differences']:.3g}.",
        f"- With γ=0 **within every hidden hotspot class**, different class mixtures produce an aggregated log-DiD of {b['aggregate_log_DiD']:.6f}. This is a model-internal counterexample to unconditional interaction identification.",
        '- Therefore reject the proposed raw bright/dark diagnostic. Even a corrected DiD is only interpretable conditional on hotspot, history, age, independent field and state measurement.',
        '- A state-specific γ term remains a **speculative mechanism**, not an observed finding or established novelty. Do not retroactively score it as the original child surviving.',
        '',
        '## A — Lag asymmetry',
        f"- Intrinsic stationary equilibrium A=0, but with detector channel lag {x['A_INSTRUMENT_DELAY_NULL']['delay']}, the observed illustrative A is {x['A_INSTRUMENT_DELAY_NULL']['observed_A']:.6f}.",
        '- Time-asymmetric covariance is known general physics. Neither observing A≠0 nor drive sign reversal alone identifies adsorption cycling versus photophysical nonequilibrium and instrumentation.',
        '',
        '## C — Occupancy-mediated observability',
        f"- Structural model U→N→Y gives total causal slope {x['C_MEDIATOR_CONDITIONING_NULL']['total_causal_slope']:.3f} but direct slope holding N fixed {x['C_MEDIATOR_CONDITIONING_NULL']['controlled_direct_slope_holding_occupancy']:.3f}.",
        '- Conditioning on occupancy can erase the mediation path one wishes to test. Observed predictive-score boundary is not by itself new physics.',
        '',
        '## Claim-level dispositions',
    ]
    lines += [f"- **{r['claim_id']}** {r['disposition']}: {r['claim']}" for r in claims]
    lines += ['', '## Generational search policy',
      '- Rejected discriminatory claims are not repaired in place. A newly generated prediction is a **new candidate**, linked to the failed claim as negative search context.',
      '- Alternative scientific programs can remain speculative despite prior-art uncertainty, but cannot be counted as certified original novelty.',
      '- Three independent bounded search tasks are prepared; none has been run here. Any paid call requires a separate explicit user action.',
      '- No canonical KG mutation, no native scheduler execution, no empirical validation.',
      '', '## Primary literature boundaries (not an exhaustive novelty search)',
    ]
    lines += [f"- DOI {x['doi']}: {x['scope']}" for x in cfg['literature_boundaries']]
    lines += ['',f"Source M10 packet SHA-256: `{cfg['source_sha256']['m10_packet']}`",
            f"Source M10 review SHA-256: `{cfg['source_sha256']['m10_review']}`",'']
    return '\n'.join(lines)


def claim_csv(cfg:dict) -> str:
    cols=['claim_id','candidate','claim','disposition','basis','failure_use']
    sio=io.StringIO(); w=csv.DictWriter(sio,fieldnames=cols); w.writeheader();w.writerows(cfg['claim_dispositions'])
    return sio.getvalue()


__all__=['SCHEMA','canonical','digest','file_sha','write_new','read_json_zip','physics_counterexamples',
         'build_frozen_experiment','make_prompt','audit_generated','render_report','claim_csv']
