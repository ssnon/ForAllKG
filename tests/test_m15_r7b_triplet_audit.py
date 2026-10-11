import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.discovery import run_m15_r7b_triplet_audit as t


def write(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


class ThreeMeshTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.b=root/'base'; self.m=root/'eight'; self.f=root/'six'; self.o=root/'audit'
        for d in (self.b,self.m,self.f): d.mkdir()
        self.base={'r6_dir':str(root/'original'),'mesh_edge_um':0.012,'sphere_triangle_count':614,
                   'science_backend_qualified':False,
                   'comparison':{'mie_sigma_sca_um2':1.,'scuff_sigma_sca_um2':1-.036095,
                                 'relative_error_pct':3.6095,'pabs_over_psca_absolute':1e-7}}
        write(self.b/'M15R6B_SINGLE_MESH_GATE.json',self.base)
        for d,edge,tri,sigma in ((self.m,8.,1262,1-.0176725),(self.f,6.,2050,1-.008)):
            (d/'sphere.scuffgeo').write_text('same physics')
            (d/'sphere.msh').write_text('mesh'+str(edge))
            (d/'sphere.PFT').write_text(str(sigma))
            plan={'mesh_edge_nm':edge,'mesh_audit':{'faces':tri},'r6b_dir':str(self.b),
                  'r6b_sha256':t.sha(self.b/'M15R6B_SINGLE_MESH_GATE.json')}
            write(d/'M15R7A_PLAN.json',plan)
            receipt={'status':'REFINED_SCUFF_PFT_PRODUCED_UNCALIBRATED', 'exit_code':0, 'timed_out':False,
                     'command':['scuff','--Lambda','0.633','--pwDirection','0','0','1',
                                '--pwPolarization','1','0','0']}
            write(d/'M15R7A_RUN.json',receipt)
            comp={'sigma':sigma,'error':abs(sigma-1.)*100,'abs_ratio':1e-7}
            audit={'schema_version':'m15r7a-sphere-refinement-v1','science_gate':'BLOCKED',
                   'science_backend_qualified':False,'mesh_convergence_demonstrated':False,
                   'plan_sha256':t.sha(d/'M15R7A_PLAN.json'),
                   'run_sha256':t.sha(d/'M15R7A_RUN.json'),
                   'pft_sha256':t.sha(d/'sphere.PFT'),
                   'refined_scuff_um2':sigma,'refined_reference_error_pct':comp['error'],
                   'mie_reference_um2':1.,'pabs_over_psca_absolute':comp['abs_ratio']}
            write(d/'M15R7A_AUDIT.json',audit)
        def compared(pft,mie):
            sigma=pft['sigma'];return {'scuff_sigma_sca_um2':sigma,
                                     'relative_error_pct':abs(sigma-mie)/mie*100,
                                     'pabs_over_psca_absolute':1e-7}
        self.patches=[
            patch.object(t.r6b,'check',return_value={'science_gate':'BLOCKED'}),
            patch.object(t.r6,'mie_sphere_scattering',return_value=1.),
            patch.object(t.r7a,'check',return_value={'solver_run':'REFINED_SCUFF_PFT_PRODUCED_UNCALIBRATED'}),
            patch.object(t.r7a,'_load_plan',side_effect=lambda d: json.loads((d/'M15R7A_PLAN.json').read_text())),
            patch.object(t.r6b,'parse_pft',side_effect=lambda pft,**kw:{'sigma':float(pft)}),
            patch.object(t.r6b,'compare',side_effect=compared),
        ]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def test_three_mesh_supportive(self):
        r=t.audit(self.b,self.m,self.f,self.o)
        self.assertEqual(r['screen_status'],'SPHERE_3MESH_TREND_SUPPORTIVE')
        self.assertFalse(json.loads((self.o/'M15R7B_AUDIT.json').read_text())['metrics']['sphere_mesh_convergence_proved'])
        self.assertEqual(t.check(self.o)['status'],'M15R7B_CHECK_OK')

    def test_no_overwrite(self):
        t.audit(self.b,self.m,self.f,self.o)
        with self.assertRaises(FileExistsError):t.audit(self.b,self.m,self.f,self.o)

    def test_do_not_qualify_au_ag(self):
        t.audit(self.b,self.m,self.f,self.o)
        m=json.loads((self.o/'M15R7B_AUDIT.json').read_text())['metrics']
        self.assertEqual(m['au_ag_science_gate'],'BLOCKED')
        self.assertFalse(m['au_ag_materials_qualified'])
        self.assertFalse(m['au_ag_mesh_convergence_demonstrated'])

    def test_six_nm_must_be_distinct(self):
        with self.assertRaisesRegex(ValueError,'DISTINCT'):
            t.audit(self.b,self.m,self.m,self.o)

    def test_six_nm_expected(self):
        p=self.f/'M15R7A_PLAN.json';x=json.loads(p.read_text());x['mesh_edge_nm']=5.;write(p,x)
        with self.assertRaisesRegex(ValueError,'EXPECTED_MESH_SIZE'):
            t.audit(self.b,self.m,self.f,self.o)

    def test_material_geometry_unchanged(self):
        (self.f/'sphere.scuffgeo').write_text('tampered physics')
        with self.assertRaisesRegex(ValueError,'PHYSICAL_MODEL_CHANGED'):
            t.audit(self.b,self.m,self.f,self.o)

    def test_modifying_audited_pft_rejected(self):
        t.audit(self.b,self.m,self.f,self.o)
        (self.f/'sphere.PFT').write_text('0.5')
        with self.assertRaisesRegex(ValueError,'AUDIT_HASH_MISMATCH'):
            t.check(self.o)

    def test_modifying_report_rejected(self):
        t.audit(self.b,self.m,self.f,self.o)
        (self.o/'M15R7B_REPORT.md').write_text('wrong')
        with self.assertRaisesRegex(ValueError,'REPORT_CHANGED'):
            t.check(self.o)

    def test_source_gate_changed_rejected(self):
        t.audit(self.b,self.m,self.f,self.o)
        x=json.loads((self.b/'M15R6B_SINGLE_MESH_GATE.json').read_text());x['some_change']=1
        write(self.b/'M15R6B_SINGLE_MESH_GATE.json',x)
        with self.assertRaisesRegex(ValueError,'BASELINE_MISMATCH'):
            t.check(self.o)

    def test_sphere_not_supportive_if_trend_reverses(self):
        (self.f/'sphere.PFT').write_text('0.95')
        f=json.loads((self.f/'M15R7A_AUDIT.json').read_text())
        f['pft_sha256']=t.sha(self.f/'sphere.PFT')
        f['refined_scuff_um2']=0.95;f['refined_reference_error_pct']=5.0
        write(self.f/'M15R7A_AUDIT.json',f)
        r=t.audit(self.b,self.m,self.f,self.o)
        self.assertEqual(r['screen_status'],'SPHERE_3MESH_NEEDS_REVIEW')

    def test_receipt_must_succeed(self):
        p=self.f/'M15R7A_RUN.json';r=json.loads(p.read_text());r['exit_code']=2;write(p,r)
        with self.assertRaisesRegex(ValueError,'RECEIPT_INVALID'):
            t.audit(self.b,self.m,self.f,self.o)

    def test_auth_flags_not_overridden(self):
        p=self.f/'M15R7A_AUDIT.json';r=json.loads(p.read_text());r['science_backend_qualified']=True;write(p,r)
        with self.assertRaisesRegex(ValueError,'AUTHORITY_INVALID'):
            t.audit(self.b,self.m,self.f,self.o)


if __name__=='__main__': unittest.main()
