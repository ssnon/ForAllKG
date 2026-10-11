import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.discovery import run_m15_r8b_minimal_gate as r8b
from scripts.discovery import run_m15_r9_feedback_pilot as r9

PFT='''# scuff-scatter run
# 1 omega (rad/sec)
# 2 surface label
# 3 absorbed power (watts)
# 4 scattered power (watts)
# 5 x-force (nanonewtons)
# 6 y-force (nanonewtons)
# 7 z-force (nanonewtons)
# 8 x-torque (nanonewtons * microns)
# 9 y-torque (nanonewtons * microns)
#10 z-torque (nanonewtons * microns)
9.92604 OuterShell 3.815886e-06 1.142806e-04 -6.087175e-09 -4.617193e-11 5.005866e-04 -8.068042e-11 6.828506e-10 -8.577168e-10
9.92604 InnerCore -4.047652e-07 4.047652e-07 -6.555181e-03 -6.265093e-03 -2.136495e-02 -2.744370e-11 -1.081434e-10 7.207427e-12
'''
GEOM='''*  GEOMETRY nanorod.scuffgeo
      2 surfaces
   9822 total basis functions
 Size of BEM matrix: 1.54 GB
*  REGIONS: 3
index | Label | Material properties
0 | EXTERIOR | CONST_EPS_1.7689
1 | OuterShell | FILE_Ag_eps.dat
2 | InnerCore | FILE_Au_eps.dat
*  SURFACE 0: Label = OuterShell
2280 panels
0 boundary contours
1142 total vertices (after eliminating 0 redundant vertices)
*  SURFACE 1: Label = InnerCore
994 panels
0 boundary contours
499 total vertices (after eliminating 1023607989 redundant vertices)
'''

class Fixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.r8a=self.root/'R8A';self.r8a.mkdir();self.r8b=self.root/'R8B';self.r9=self.root/'R9'
        self.geom=self.r8a/'M15R8A_GEOMETRY_AUDIT.log';self.geom.write_text(GEOM)
        (self.r8a/'nanorod.PFT').write_text(PFT)
        inp={}
        for name in ('Ag_eps.dat','Au_eps.dat','inner_core.msh','nanorod.scuffgeo','outer_shell.msh','scuff.args'):
            f=self.r8a/name;f.write_text('immutable fixture '+name)
            inp[name]=r8b.digest(f)
        plan={'schema_version':'m15r8a-auag-scuff-pilot-v1','au_ag_science_gate':'BLOCKED',
              'science_backend_qualified':False,'wavelength_nm':633,'polarization':'Ex',
              'material_assignment':{'exterior':'CONST_EPS_1.7689','inner':'Au','outer':'Ag'},
              'input_hashes':inp}
        self.p=self.r8a/'M15R8A_PLAN.json';self.p.write_text(json.dumps(plan))
        run={'schema_version':'m15r8a-auag-scuff-pilot-v1','au_ag_science_gate':'BLOCKED','science_backend_qualified':False,
             'status':'AU_AG_PFT_GENERATED_UNCALIBRATED','exit_code':0,'timed_out':False,
             'input_plan_sha256':r8b.digest(self.p),'pft_sha256':r8b.digest(self.r8a/'nanorod.PFT'),
             'command':['scuff-scatter','--geometry','nanorod.scuffgeo','--Lambda','0.633',
                        '--pwDirection','0','0','1','--pwPolarization','1','0','0','--PFTFile','nanorod.PFT']}
        (self.r8a/'M15R8A_RUN.json').write_text(json.dumps(run))

    def validated_r8b(self):
        return r8b.audit(self.r8a,self.r8b,self.geom if self.geom.exists() else None)

class R8BTests(Fixtures):
    def test_parse_and_warn(self):
        self.validated_r8b();self.assertEqual(r8b.check(self.r8b)['status'],'R8B_CHECK_OK')
        gate=json.loads((self.r8b/'M15R8B_EVIDENCE_GATE.json').read_text())
        self.assertFalse(gate['evidence']['numerical_science_feedback_permitted'])
        self.assertIn('INNER_INTERFACE_NEGATIVE_ABSORBED_POWER_NEEDS_PFT_CONVENTION_REVIEW',gate['evidence']['observed_issue_codes'])
        self.assertEqual(gate['evidence']['geometry_verified_if_supplied']['triangles']['InnerCore'],994)
    def test_geometry_counter_anomaly_no_false_science(self):
        parsed=r8b.parse_geometry(GEOM);self.assertTrue(parsed['implausible_redundant_vertex_count'])
    def test_tamper_pft_rejected(self):
        self.validated_r8b();(self.r8a/'nanorod.PFT').write_text(PFT+'# tampered\n')
        with self.assertRaisesRegex(ValueError,'SOURCE_CHANGED'):r8b.check(self.r8b)
    def test_tamper_model_file_rejected(self):
        (self.r8a/'Ag_eps.dat').write_text('changed')
        with self.assertRaisesRegex(ValueError,'INPUT_HASH_MISMATCH'):self.validated_r8b()
    def test_duplicate_audit_blocked(self):
        self.validated_r8b()
        with self.assertRaisesRegex(FileExistsError,'NO_OVERWRITE'):self.validated_r8b()
    def test_wrong_frequency_rejected(self):
        with self.assertRaisesRegex(ValueError,'WAVELENGTH_MISMATCH'):
            r8b.parse_pft((self.r8a/'nanorod.PFT'),532.0,{'OuterShell','InnerCore'})
    def test_wrong_material_rejected(self):
        run=json.loads((self.r8a/'M15R8A_RUN.json').read_text())
        plan=json.loads(self.p.read_text());plan['material_assignment']['inner']='Ag'
        with self.assertRaisesRegex(ValueError,'MODEL_NOT_EXPECTED'):
            r8b.derive_state(plan,run,r8b.parse_pft(self.r8a/'nanorod.PFT',633,{'OuterShell','InnerCore'}),None)
    def test_no_geom_is_explicit(self):
        self.geom.unlink();self.validated_r8b()
        b=json.loads((self.r8b/'M15R8B_EVIDENCE_GATE.json').read_text())
        self.assertIn('GEOMETRY_ANALYZE_LOG_NOT_SUPPLIED',b['evidence']['observed_issue_codes'])

class R9Tests(Fixtures):
    def prep(self):
        self.validated_r8b();return r9.prepare(self.r8b,self.r9,'openai/test-model',2,.3,8000)
    def test_prepare_balanced(self):
        self.prep();plan=r9.load(self.r9)
        self.assertEqual(len(plan['calls']),4)
        self.assertEqual(sum(x['arm']=='CONTROL' for x in plan['calls']),2)
        self.assertEqual(sum(x['arm']=='DIAGNOSTIC_TREATMENT' for x in plan['calls']),2)
        self.assertTrue(all('NOT independently validated' not in c['user_prompt'] for c in plan['calls'] if c['arm']=='CONTROL'))
        self.assertTrue(all('NOT independently validated' in c['user_prompt'] for c in plan['calls'] if c['arm']=='DIAGNOSTIC_TREATMENT'))
    def test_no_implicit_payment(self):
        self.prep()
        with self.assertRaisesRegex(ValueError,'REQUIRES_BOTH'):
            r9.run_paid(self.r9,2,True,False)
    def test_provider_key_required(self):
        self.prep()
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(ValueError,'OPENROUTER_API_KEY_MISSING'):
                r9.run_paid(self.r9,4,True,True)
    def test_paid_resume_and_blind_audit(self):
        self.prep()
        obj={k:'Detailed explanation about '+k for k in r9.KEYS}
        provider={'id':'mock','choices':[{'message':{'content':json.dumps(obj)}}]}
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'test-key'}),patch.object(r9,'invoke_openrouter',return_value=provider) as fn:
            a=r9.run_paid(self.r9,2,True,True)
            self.assertEqual(a['new_paid_calls'],2);self.assertEqual(a['pending_calls'],2)
            a=r9.run_paid(self.r9,4,True,True)
            self.assertEqual(a['new_paid_calls'],2);self.assertEqual(a['pending_calls'],0)
            self.assertEqual(fn.call_count,4)
        result=r9.audit(self.r9)
        self.assertEqual(result['blind_candidates'],4)
        blinded=(self.r9/'M15R9P_BLIND_REVIEW_PACKET.json').read_text()
        self.assertNotIn('DIAGNOSTIC_TREATMENT',blinded)
        self.assertNotIn('CONTROL',blinded)
        self.assertEqual(r9.check(self.r9)['saved_receipts'],4)
    def test_raw_tamper_blocked(self):
        self.test_paid_resume_and_blind_audit()
        receipt=next((self.r9/'responses').glob('*.json'))
        receipt.write_text(receipt.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'MODIFIED'):r9.check(self.r9)
    def test_blind_review_then_unblind(self):
        import csv
        self.test_paid_resume_and_blind_audit()
        path=self.r9/'M15R9P_BLIND_REVIEW_RUBRIC.csv'
        with path.open(newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            for col in ('conceptual_delta_0to2','model_discrimination_0to2','independent_controls_0to2',
                        'falsifiability_0to2','epistemic_caution_0to2'):
                row[col]='1'
            row['prior_art_risk_low_med_high']='med'
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        res=r9.unblind(self.r9,path)
        self.assertEqual(res['mean_treatment_minus_control'],0)
        self.assertFalse(res['scientific_novelty_certified'])
        with self.assertRaisesRegex(FileExistsError,'NO_OVERWRITE'):
            r9.unblind(self.r9,path)

    def test_unblind_requires_completed_scores(self):
        self.test_paid_resume_and_blind_audit()
        with self.assertRaisesRegex(ValueError,'MISSING_VALID_REVIEW_SCORE'):
            r9.unblind(self.r9,self.r9/'M15R9P_BLIND_REVIEW_RUBRIC.csv')

    def test_structurally_invalid_response_not_scientific(self):
        self.assertEqual(r9.parse_candidate('not-json')[1],'NOT_JSON')
        self.assertEqual(r9.parse_candidate('')[1],'EMPTY_CONTENT')
        self.assertEqual(r9.parse_candidate('{}')[1],'MISSING_REQUIRED_FIELDS')
    def test_output_dir_no_overwrite(self):
        self.prep()
        with self.assertRaisesRegex(FileExistsError,'NO_OVERWRITE'):
            r9.prepare(self.r8b,self.r9,'model',1,.3,8000)
    def test_model_required(self):
        self.validated_r8b()
        with self.assertRaisesRegex(ValueError,'MODEL_REQUIRED'):
            r9.prepare(self.r8b,self.r9,'',2,.3,8000)
    def test_backend_no_claim_authority(self):
        self.prep();plan=r9.load(self.r9)
        self.assertFalse(plan['quantitative_physics_authority'])
        self.assertFalse(plan['scientific_novelty_certified'])

if __name__=='__main__':unittest.main()
