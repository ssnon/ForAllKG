from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/discovery/run_discovery_capability_audit.py'
spec=importlib.util.spec_from_file_location('cap_audit',SCRIPT)
audit_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def fixture():
 return {'schema':audit_module.SCHEMA,'source_file_index':[{'file':'source.txt','sha256':audit_module.hashlib.sha256(b'ok').hexdigest()}],
 'studies':[
 {'study':'M13','budget_parity_certified':False,'n_candidates':96,'generations':6,'replicates':4,'novelty_certified':False},
 {'study':'M14','budget_parity_certified':False,'n_candidates':96,'generations':6,'replicates':4,'novelty_certified':False},
 {'study':'M14C4','admitted_problem_count':0,'novelty_certified':False},
 {'study':'M15_R9','subjective_mean_delta':0.5,'novelty_certified':False},
 {'study':'M15_R10','subjective_mean_delta':2.5,'novelty_certified':False},
 {'study':'M16','numeric_over_known':0,'arm_means':{'BASELINE':6.5,'KNOWN_PHYSICS':9.5,'NUMERIC_PHENOMENON':9.5},'novelty_certified':False}],
 'candidate_questions':[{'problem_id':'x1','family':'SERS','question':'Q','claim_status':'UNVERIFIED_PRIOR_ART_GAP','action':'NO_CERTIFICATION','source':'x'}],
 'diagnosed_confounders':[{'id':'Z','severity':'blocking','basis':'fixture'}]}

class AuditTests(unittest.TestCase):
 def test_positive_fixture(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'snapshot.json';p.write_text(json.dumps(fixture()))
   self.assertEqual(len(audit_module.read_snapshot(p)['studies']),6)

 def test_reject_novelty_fraud(self):
  with tempfile.TemporaryDirectory() as td:
   f=fixture();f['studies'][0]['novelty_certified']=True
   p=Path(td)/'snapshot.json';p.write_text(json.dumps(f))
   with self.assertRaisesRegex(ValueError,'UNSUPPORTED_NOVELTY_CLAIM'):audit_module.read_snapshot(p)

 def test_reject_causal_budget_parity(self):
  with tempfile.TemporaryDirectory() as td:
   f=fixture();f['studies'][1]['budget_parity_certified']=True
   p=Path(td)/'snapshot.json';p.write_text(json.dumps(f))
   with self.assertRaisesRegex(ValueError,'UNSUPPORTED_BUDGET_PARITY_CLAIM'):audit_module.read_snapshot(p)

 def test_reject_unjustified_m16_superiority(self):
  with tempfile.TemporaryDirectory() as td:
   f=fixture();f['studies'][-1]['numeric_over_known']=1
   p=Path(td)/'snapshot.json';p.write_text(json.dumps(f))
   with self.assertRaisesRegex(ValueError,'M16_DELTA_NOT_ZERO'):audit_module.read_snapshot(p)

 def test_reject_claim_upgrade(self):
  with tempfile.TemporaryDirectory() as td:
   f=fixture();f['candidate_questions'][0]['claim_status']='NEW_SCIENCE'
   p=Path(td)/'snapshot.json';p.write_text(json.dumps(f))
   with self.assertRaisesRegex(ValueError,'UNSUPPORTED_PRIOR_ART_CERTIFICATION'):audit_module.read_snapshot(p)

 def test_negative_control_algebra(self):
  z=audit_module.nonidentifiability_counterexample()
  self.assertNotEqual(z['parameters_A'],z['parameters_B'])
  self.assertEqual(z['same_observations'],[30,24])
  self.assertFalse(z['scientific_novelty_certified'])

 def test_run_check_and_tampering(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);f=d/'snapshot.json';f.write_text(json.dumps(fixture()))
   o=d/'results';a=audit_module.audit(f,o)
   self.assertEqual(a['sources_verified_count'],0)
   audit_module.check(o)
   (o/'DISCOVERY_CAPABILITY_AUDIT.md').write_text('tampered')
   with self.assertRaisesRegex(ValueError,'OUTPUT_TAMPER_DETECTED'):audit_module.check(o)

 def test_source_hash_validation(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);(d/'source.txt').write_bytes(b'ok');f=d/'snapshot.json';f.write_text(json.dumps(fixture()))
   x=audit_module.audit(f,d/'results',d,require_sources=True)
   self.assertEqual(x['sources_verified_count'],1)
   (d/'source.txt').write_bytes(b'bad')
   with self.assertRaisesRegex(ValueError,'SOURCE_CONTENT_CHANGED'):
    audit_module.audit(f,d/'results2',d)

 def test_requires_source_flag(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);f=d/'snapshot.json';f.write_text(json.dumps(fixture()))
   with self.assertRaisesRegex(ValueError,'REQUIRED_SOURCE_MISSING'):
    audit_module.audit(f,d/'results',d,require_sources=True)

 def test_no_output_overwrite(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);f=d/'snapshot.json';f.write_text(json.dumps(fixture())); (d/'res').mkdir()
   with self.assertRaises(FileExistsError):audit_module.audit(f,d/'res')

if __name__=='__main__':unittest.main()
