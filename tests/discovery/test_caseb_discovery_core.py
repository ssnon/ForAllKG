import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from pipeline_core.discovery.scientific_discovery_core.contracts import Mechanism
from pipeline_core.discovery.scientific_discovery_core.engine import (
    extract_graph, identity_check, prepare, read_json, run, sha, verify)

PROBLEM = Path(__file__).resolve().parents[2] / 'scientific_cases' / 'CASEB_PROBLEM_PACKET.json'

MECHANISM={
 "title":"Reversible interfacial coupled-state hypothesis",
 "distinct_causal_claim":"A shared electronic state changes the response through a coupling dependent on surface orientation.",
 "latent_state_variables":["interfacial charge q","optical-field factor G"],
 "coupling_law":"d q/d t = k_capture*(1-q) - k_release*q",
 "quantitative_derivation":"An independent perturbation must isolate the product degeneracy and validate competing predictions.",
 "physical_feasibility_constraints":["Bounds on carrier population and trap relaxation rate"],
 "novelty_risk":"Known charge-transfer physics may already explain this pattern.",
 "strongest_existing_null":"Ordinary orientation-dependent Raman tensor with static plasmon and changing occupancy.",
 "discriminating_prediction":"At matched molecular loading, a reversible bias perturbation should change a selected mode susceptibility without equal plasmon change.",
 "independently_manipulated_variable":"External independently calibrated local bias",
 "independently_measured_outcome":"Raman mode ratio and independent local plasmon-scattering response",
 "confounders":["Bias-induced morphology change"],
 "falsifier":"No reproducible independently verified chemical-state shift at matched local field.",
 "source_ids":["B-R01"], "measurement_matrix":[[1,1],[1,-1]],
 "measurement_calibration":"NOT_CALIBRATED", "source_support_scope":"HYPOTHESIS_ONLY"
}

class ScriptedBackend:
 def __init__(self): self.calls=[]
 def complete(self,*,system,user,model,temperature,max_tokens):
  j=json.loads(user);stage=j['task'];self.calls.append(stage)
  if stage in ('propose','baseline'):
   body={"research_action":"GENERATE_ALTERNATIVES","scientific_reason":"These proposed explanations are unverified, intended to test discriminating observables.","mechanisms":[MECHANISM]}
  elif stage=='challenge':
   body={"rows":[{"title":MECHANISM['title'],"physical_status":"NEEDS_BOUNDS",
     "missing_or_invalid_derivation":"No independently measured carrier density or charge relaxation values.",
     "strongest_null_mimic":"Orientation and morphology can generate the same Raman response.",
     "truly_discriminating_after_null":"Requires independent surface potential calibration.",
     "source_scope_limitations":"Literature data do not establish this sample-specific carrier process.",
     "proposed_test_or_computation":"Calibrate local voltage and correlated field response.",
     "next_action_advice":"REFRAME_ROOT"}],"recurrent_failure_pattern":"Predictions require uncalibrated confounding variables."}
  else:
   body={"action":"REFRAME_ROOT","rationale":"Current candidate does not uniquely identify chemical enhancement under the strong null.",
         "target_title":None,"updated_open_question":"Can a geometry-neutral bias perturbation isolate chemical-state changes?",
         "abandoned_assumption":"A changed mode ratio uniquely diagnoses chemical enhancement.",
         "scientific_uncertainty":"Independent bias and geometry constraints are not available."}
  return body,{"prompt_tokens":100,"completion_tokens":150,"total_tokens":250,"served_model":model}

class Contract(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
  self.o=Path(self.t.name)/'run'
 def prep(self,arm='ROOT_NO_KG',**kw):
  return prepare(problem_file=PROBLEM,arm=arm,out=self.o,**kw)
 def test_packet_references(self):
  x=read_json(PROBLEM); self.assertEqual(x['schema_version'],'fakg-open-problem-v1')
  self.assertEqual(len({a['id'] for a in x['references']}),5)
 def test_prepared_no_calls(self):
  self.prep(); self.assertEqual(verify(self.o)['status'],'PREPARED_NO_CALLS');self.assertFalse((self.o/'calls').exists())
 def test_prohibit_overwrite(self):
  self.prep();self.assertRaises(FileExistsError,self.prep)
 def test_tampered_packet(self):
  self.prep();data=read_json(self.o/'problem.json');data['title']='forged';(self.o/'problem.json').write_text(json.dumps(data))
  self.assertRaises(ValueError,verify,self.o)
 def test_missing_kg_required(self):
  self.assertRaises(ValueError,self.prep,'ROOT_KG')
 def test_no_kg_leakage(self):
  g=Path(self.t.name)/'graph.json';g.write_text('{}')
  self.assertRaises(ValueError,self.prep,'ROOT_NO_KG',graph_packet=g)
 def test_lineage_requires_real_p0(self):
  self.assertRaises(ValueError,self.prep,'LINEAGE_KG')
 def test_identity_rank(self):
  self.assertEqual(identity_check(Mechanism.model_validate(MECHANISM))['rank'],2)
  q=dict(MECHANISM);q['measurement_matrix']=[[1,1],[2,2]]
  self.assertEqual(identity_check(Mechanism.model_validate(q))['status'],'UNDERIDENTIFIED')
 def test_paid_gate(self):
  self.prep();self.assertRaises(PermissionError,run,self.o,ScriptedBackend(),allow_paid=False,max_calls=6)
  self.assertFalse((self.o/'calls').exists())
 def test_live_loop_fake_backend(self):
  self.prep();os.environ['FAKG_CASEB_AUTHORIZE_PAID']='YES';self.addCleanup(lambda:os.environ.pop('FAKG_CASEB_AUTHORIZE_PAID',None))
  b=ScriptedBackend();r=run(self.o,b,allow_paid=True,max_calls=6)
  self.assertEqual(len(b.calls),6)
  self.assertEqual(r['calls_completed'],6)
  self.assertEqual(r['rounds'][-1]['decision']['action'],'REFRAME_ROOT')
  for step in r['rounds']:self.assertEqual(step['tools'][0]['identifiability']['rank'],2)
  b2=ScriptedBackend();r2=run(self.o,b2,allow_paid=True,max_calls=6)
  self.assertEqual(len(b2.calls),0)
  self.assertEqual(sha(r),sha(r2))
 def test_cap_is_enforced(self):
  self.prep();os.environ['FAKG_CASEB_AUTHORIZE_PAID']='YES';self.addCleanup(lambda:os.environ.pop('FAKG_CASEB_AUTHORIZE_PAID',None))
  b=ScriptedBackend()
  with self.assertRaises(RuntimeError):run(self.o,b,allow_paid=True,max_calls=1)
  self.assertEqual(b.calls,['propose'])
 def test_partial_call_never_replayed(self):
  self.prep();os.environ['FAKG_CASEB_AUTHORIZE_PAID']='YES';self.addCleanup(lambda:os.environ.pop('FAKG_CASEB_AUTHORIZE_PAID',None))
  class Broken:
   def complete(self,**kw):raise RuntimeError('timeout after billing unknown')
  with self.assertRaises(RuntimeError):run(self.o,Broken(),allow_paid=True,max_calls=6)
  with self.assertRaises(RuntimeError):run(self.o,ScriptedBackend(),allow_paid=True,max_calls=6)
 def test_baseline_is_independent(self):
  self.prep('LLM_BASELINE');os.environ['FAKG_CASEB_AUTHORIZE_PAID']='YES';self.addCleanup(lambda:os.environ.pop('FAKG_CASEB_AUTHORIZE_PAID',None))
  b=ScriptedBackend();r=run(self.o,b,allow_paid=True,max_calls=6)
  self.assertEqual(b.calls,['baseline']*6)
  self.assertEqual(r['calls_completed'],6)
  self.assertEqual(len(r['rounds']),6)
 def test_bad_scientific_matrix(self):
  q=dict(MECHANISM);q['measurement_matrix']=[[1,2,3]]
  self.assertRaises(ValueError,Mechanism.model_validate,q)

if __name__=='__main__':unittest.main()
