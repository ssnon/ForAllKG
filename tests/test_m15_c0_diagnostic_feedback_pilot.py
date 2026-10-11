"""Synthetic fixture tests only; never claim to be real Meep execution evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/discovery/run_m15_c0_diagnostic_feedback_pilot.py'
spec = importlib.util.spec_from_file_location('m15c0', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class M15C0Tests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.base = Path(self.t.name)
        def req(folder):
            ex = folder in ('diagnostic_short_v1','diagnostic_time16_ex_v1')
            time = 8 if folder in ('diagnostic_short_v1','diagnostic_short_trans_v1') else 16
            res = 90 if folder=='diagnostic_resolution90_ey_v1' else 60
            return dict(schema_version='sers-meep-execution-request-v0',request_id=folder,
                source_solver_job_id='ex_job' if ex else 'ey_job',source_numerical_plan_id='ex_num' if ex else 'ey_num',
                geometry_family='core_shell_nanorod',core_material='Au',shell_material='Ag',
                shell_thickness_nm=20.0,nanorod_length_nm=160.,nanorod_diameter_nm=50.,
                polarization='parallel_to_nanorod_long_axis' if ex else 'perpendicular_to_nanorod_long_axis',
                field_component='Ex' if ex else 'Ey', wavelength_case_bindings=[{'source_case_id':'ex_case' if ex else 'ey_case','wavelength_nm':500.}],
                resolution_px_per_um=res,run_until_after_sources=time,background_index=1.33,material_model='meep_materials_library')
        self.sources=[]
        self.results={}
        for folder in m.SOURCE_DIRS:
            directory=self.base/folder;directory.mkdir()
            r=req(folder)
            vals = ([0.1*i+1 for i in range(8)] if r['field_component']=='Ex' else
                [1.0,1.2,1.4,1.8,2.0,1.8,1.5,1.0])
            if r['run_until_after_sources']==16: vals=[v*1.002 for v in vals]
            if r['resolution_px_per_um']==90: vals=[v*.73 for v in vals]
            obs=dict(schema_version='sers-meep-worker-result-v0',request_id=folder,
                backend='meep',worker_status='completed',field_component=r['field_component'],
                resolution_px_per_um=r['resolution_px_per_um'],run_until_after_sources=r['run_until_after_sources'],
                scientific_interpretation_permitted=False,shadow_only=True,
                evidence_eligibility='execution_sanity_only',numerical_resolution_status='underresolved',
                spectrum=[{'wavelength_nm':500. + 50*i,'scattering_cross_section_um2':v} for i,v in enumerate(vals)])
            for name,data in (('request.json',r),('result.json',obs)):
                p=directory/name;p.write_text(json.dumps(data),encoding='utf-8')
                self.sources.append({'path':str(p),'sha256':m.hashfile(p)})
            self.results[folder]=obs
        met=m.spectrum_comparison(self.results['diagnostic_time16_ey_v1'],self.results['diagnostic_resolution90_ey_v1'])
        gate=dict(schema_version='m15b-evidence-gate-v1',engineering_execution='PASS',temporal_stability='PRELIMINARY_PASS',
            spatial_stability='FAIL',quantitative_physics_feedback='BLOCKED',scientific_novelty='NOT_TESTED',
            permitted_use='numerical_diagnostic_feedback_only', ex_peak_location='UPPER_BOUNDARY_CENSORED',
            ey_spatial_relative_l1_change=met['relative_l1'],
            ey_peak_shift_nm=met['peak_b_nm']-met['peak_a_nm'],sources=self.sources)
        self.gate=self.base/'M15B_EVIDENCE_GATE.json';self.gate.write_text(json.dumps(gate))
        self.out=self.base/'C0'

    def test_prepare_preflight_and_no_paid_calls(self):
        a=m.prepare(self.gate,self.out,1501)
        self.assertEqual(a['new_paid_calls'],0)
        plan,spec=m.preflight(self.out)
        self.assertAlmostEqual(plan['metrics']['ey_spatial_relative_l1'],.369863,places=5)
        self.assertIn('worker results',spec['prompts']['B']['user'])
        self.assertNotIn('27.000%',spec['prompts']['A']['user'])
        self.assertFalse(plan['novelty_certified'])
        m.prepare(self.gate,self.out,1501) # idempotent

    def test_block_gate_bad_status(self):
        gate=m.load(self.gate);gate['quantitative_physics_feedback']='PASS'
        self.gate.write_text(json.dumps(gate))
        with self.assertRaisesRegex(ValueError,'GATE_STATUS_MISMATCH'):
            m.prepare(self.gate,self.out,1501)

    def test_block_changed_source(self):
        (self.base/'diagnostic_short_v1/request.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SOURCE_SHA_MISMATCH'):
            m.prepare(self.gate,self.out,1501)

    def test_block_changed_gate_after_freeze(self):
        m.prepare(self.gate,self.out,1501)
        gate=m.load(self.gate);gate['unrelated_marker']='CHANGED'
        self.gate.write_text(json.dumps(gate))
        with self.assertRaisesRegex(ValueError,'GATE_CHANGED_AFTER_FREEZE'):
            m.preflight(self.out)

    def test_paid_no_authorization(self):
        m.prepare(self.gate,self.out,1501)
        with self.assertRaisesRegex(ValueError,'PAID_CALL_EXPLICIT_AUTHORIZATION_REQUIRED'):
            m.generate_paid(self.out,'any','KEY','https://example.com',2,2000,False,False)

    def test_manual_review_score(self):
        m.prepare(self.gate,self.out,1501)
        folder=self.out/'manual';folder.mkdir()
        candidate={field:('Specific independent research design with matched controls and physical scope: '+field+'.') for field in m.FIELDS}
        for arm in ('A','B'):
            (folder/f'{arm}.response.json').write_text(json.dumps(candidate),encoding='utf-8')
        r=m.materialize(self.out)
        self.assertEqual(r['candidates'],2)
        private=m.load(self.out/'M15C0_PRIVATE_ARM_MAP.json')
        self.assertEqual({x['arm'] for x in private['entries']},{'A','B'})
        with (self.out/'M15C0_REVIEW_TEMPLATE.csv').open('w',newline='',encoding='utf-8') as f:
            import csv
            writer=csv.DictWriter(f,fieldnames=m.CSV_FIELDS);writer.writeheader()
            for row in private['entries']:
                for reviewer in ('R1','R2'):
                    obj={'blind_id':row['blind_id'],'reviewer_id':reviewer,
                         'unsupported_quantitative_claim':'no','rationale':'Scientific testability reviewed with independent constraints.'}
                    obj.update({key:'3' if row['arm']=='B' else '2' for key in m.SCORE_FIELDS})
                    writer.writerow(obj)
        summary=m.score(self.out)
        self.assertEqual(summary['paired_B_minus_A']['falsifiability'],1.0)
        self.assertFalse(summary['scientific_novelty_certified'])

    def test_no_accept_same_arm_missing(self):
        m.prepare(self.gate,self.out,1501)
        folder=self.out/'manual';folder.mkdir()
        candidate={field:('Specific independent research design with matched controls and physical scope: '+field+'.') for field in m.FIELDS}
        (folder/'A.response.json').write_text(json.dumps(candidate))
        with self.assertRaisesRegex(ValueError,'MISSING_RESPONSE_B'):
            m.materialize(self.out)


if __name__=='__main__': unittest.main()
