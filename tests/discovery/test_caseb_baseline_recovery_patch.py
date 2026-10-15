"""Offline regression tests for recovery of a model-validity failure after billing."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from pipeline_core.discovery.scientific_discovery_core.contracts import Baseline, Mechanism
from pipeline_core.discovery.scientific_discovery_core.engine import (
    _prompt, _validate_preserving_science, inspect_failed_baseline,
    recover_failed_baseline, read_json, prepare, run, sha, write_new,
)

PROBLEM = Path(os.environ.get("CASEB_PATCH_TEST_PROBLEM", "scientific_cases/CASEB_PROBLEM_PACKET.json"))
MECHANISM = {
 "title":"Reversible interfacial coupled-state hypothesis",
 "distinct_causal_claim":"A shared electronic state changes Raman susceptibility through a surface-dependent interfacial coupling.",
 "latent_state_variables":["interfacial charge q", "optical-field factor G"],
 "coupling_law":"d q/dt = k_capture*(1-q)-k_release*q",
 "quantitative_derivation":"A differential potential step must isolate the product ambiguity against two competing observables.",
 "physical_feasibility_constraints":["Known carrier bounds and charge relaxation time"],
 "novelty_risk":"Prior charge-transfer observations may already account for the result.",
 "strongest_existing_null":"Orientation-dependent Raman tensor and changing occupancy with a fixed field.",
 "discriminating_prediction":"The signal should change under an independent potential perturbation at fixed and measured field.",
 "independently_manipulated_variable":"Calibrated applied interfacial bias",
 "independently_measured_outcome":"Raman intensity plus independent near-field response",
 "confounders":["Bias dependent adsorption"],
 "falsifier":"No identifiable charge response once the conventional coupled null is bounded.",
 "source_ids":["B-R01"],
 "measurement_matrix":[[1.0,1.0],[1.0,-1.0]],
 "measurement_calibration":"NOT_CALIBRATED", "source_support_scope":"HYPOTHESIS_ONLY",
}

def malformed_response():
    a=dict(MECHANISM); a["measurement_matrix"]=[[1,2,3]]
    b=dict(MECHANISM); b["title"]="An independent competing mechanism"; b["measurement_matrix"]=[{"EM":2,"CHEM":1}]
    return {"research_action":"GENERATE_ALTERNATIVES",
            "scientific_reason":"These candidate mechanisms are speculative and require independent calibration.",
            "mechanisms":[a,b]}

class Backend:
    def __init__(self):self.calls=0
    def complete(self,**_):
        self.calls+=1
        return malformed_response(),{"prompt_tokens":100,"completion_tokens":100,"total_tokens":200}

class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/"run"
        prepare(problem_file=PROBLEM,arm="LLM_BASELINE",out=self.out,rounds=2)
        os.environ["FAKG_CASEB_AUTHORIZE_PAID"]="YES"
        self.addCleanup(lambda:os.environ.pop("FAKG_CASEB_AUTHORIZE_PAID",None))
    def stage1_failed_raw(self,raw=None):
        m=read_json(self.out/"manifest.json")
        sys,user=_prompt("baseline",read_json(self.out/"problem.json"),{}, {"independent_trial":0},Baseline)
        request={"stage":"baseline_01", "system":sys, "user":user,
                 "schema":"Baseline", "model":m["model"],
                 "temperature":m["temperature"], "max_tokens":m["max_tokens"]}
        p=self.out/"calls"/"baseline_01"
        write_new(p/"request.json",request)
        write_new(p/"attempt.json",{"status":"CALL_INTENT_RECORDED", "request_sha256":sha(request)})
        if raw is not None:write_new(p/"raw_response.json",raw)
        return p
    def test_original_invalid_matrix_does_not_parse(self):
        with self.assertRaises(Exception):Baseline.model_validate(malformed_response())
    def test_recovers_two_malformed_matrices_without_model_call(self):
        raw=malformed_response();p=self.stage1_failed_raw(raw)
        before=(p/"raw_response.json").read_bytes()
        inspected=inspect_failed_baseline(self.out)
        self.assertTrue(inspected['raw_exists']);self.assertEqual(len(inspected['mechanisms']),2)
        result=recover_failed_baseline(self.out)
        self.assertEqual(result['paid_calls_performed_by_recovery'],0)
        self.assertEqual(result['quarantined_matrix_count'],2)
        self.assertEqual(before,(p/"raw_response.json").read_bytes())
        self.assertEqual([a['measurement_matrix'] for a in read_json(p/"response.json")['mechanisms']],[[],[]])
        quarantine=read_json(p/"matrix_quarantine.json")
        self.assertEqual(quarantine['diagnostics'][0]['raw_matrix'],[[1,2,3]])
        self.assertIsNone(read_json(p/"receipt.json")['usage'])
        self.assertEqual(read_json(p/"receipt.json")['recovery'],'EXPLICIT_OFFLINE_FROM_SAVED_RAW_NO_MODEL_CALL')
        self.assertRaises(FileExistsError,recover_failed_baseline,self.out)
    def test_recovery_resumes_only_five_new_calls(self):
        self.stage1_failed_raw(malformed_response())
        recover_failed_baseline(self.out)
        backend=Backend()
        res=run(self.out,backend,allow_paid=True,max_calls=6)
        self.assertEqual(backend.calls,5)
        self.assertEqual(res['calls_completed'],6)
        self.assertEqual(len(res['rounds']),6)
        self.assertEqual(len(list((self.out/'calls').glob('*/matrix_quarantine.json'))),6)
        self.assertEqual(Backend().calls,0)
    def test_no_raw_never_recover(self):
        self.stage1_failed_raw()
        with self.assertRaises(FileNotFoundError):recover_failed_baseline(self.out)
    def test_other_schema_errors_not_silently_repaired(self):
        raw=malformed_response();raw['mechanisms'][0]['title']='x'
        self.stage1_failed_raw(raw)
        with self.assertRaises(Exception):recover_failed_baseline(self.out)
        self.assertFalse((self.out/'calls'/'baseline_01'/'receipt.json').exists())
    def test_tampered_request_never_recover(self):
        p=self.stage1_failed_raw(malformed_response())
        altered=read_json(p/'request.json');altered['user']='changed'
        (p/'request.json').write_text(json.dumps(altered))
        with self.assertRaises(ValueError):recover_failed_baseline(self.out)
    def test_valid_matrix_is_unchanged(self):
        a=dict(MECHANISM)
        raw={"research_action":"GENERATE_ALTERNATIVES","scientific_reason":"Test mechanistic descriptions, uncertain and not physically confirmed.","mechanisms":[a]}
        parsed,diags=_validate_preserving_science(raw,Baseline)
        self.assertEqual(diags,[])
        self.assertEqual(parsed.mechanisms[0].measurement_matrix,[[1,1],[1,-1]])
    def test_normalization_does_not_invent_missing_fields(self):
        raw=malformed_response();del raw['mechanisms'][0]['coupling_law']
        with self.assertRaises(Exception):_validate_preserving_science(raw,Baseline)

if __name__=='__main__':unittest.main()
