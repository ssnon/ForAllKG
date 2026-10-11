import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.discovery import run_m15_r7a_sphere_refinement as t
from scripts.discovery import run_m15_r6_scuff_control as r6

HEADER = '''# columns
# 1 omega
# 2 surface label
# 3 absorbed power (watts)
# 4 scattered power (watts)
# 5 x-force (nanoNewtons)
# 6 y-force (nanoNewtons)
# 7 z-force (nanoNewtons)
# 8 x-torque (nanoNewtons*microns)
# 9 y-torque (nanoNewtons*microns)
# 10 z-torque (nanoNewtons*microns)
'''


def pft(power=1.45e-7):
    return HEADER + f'{2*math.pi/r6.LAMBDA_UM:.14g} TestSphere 0 {power} 0 0 0 0 0 0\n'


def topology(faces=890):
    return {'status':'MESH_TOPOLOGY_PASS_PRELIMINARY','faces':faces,'nodes':440,
            'boundary_edges':0,'nonmanifold_edges':0,'orientation_inconsistent_edges':0}


class MockCompleteResult:
    def __init__(self, cmd, cwd=None, *args, **kwargs):
        self.returncode = 0
        self.stdout = 'gmsh/scuff mock stdout'
        self.stderr = ''
        if '-o' in cmd:
            Path(cmd[cmd.index('-o')+1]).write_text('mock-msh-2.2')
        if '--PFTFile' in cmd:
            (Path(cwd) / 'sphere.PFT').write_text(pft())


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.r6b=self.root/'base';self.r6b.mkdir()
        self.r6=self.root/'r6';self.r6.mkdir()
        (self.r6/'sphere.scuffgeo').write_text('OBJECT TestSphere\n MESHFILE sphere.msh\n MATERIAL CONST_EPS_2.25\nENDOBJECT\n')
        (self.r6b/'M15R6B_SINGLE_MESH_GATE.json').write_text('{}')
        self.baseline={'r6_dir':str(self.r6),'sphere_triangle_count':360, 'mesh_edge_um':0.012,
                       'comparison':{'relative_error':0.0360953343,
                          'relative_error_pct':3.60953343,'scuff_sigma_sca_um2':0.00010704648705357,
                          'mie_sigma_sca_um2':r6.mie_sphere_scattering()}}
        self.baseline_check={'r6_dir':str(self.r6)}
        self.out=self.root/'refined'
        self.patches=[
            patch.object(t,'_baseline',return_value=(self.baseline_check,self.baseline)),
            patch.object(t.r6,'executable',return_value='/bin/true'),
            patch.object(t.r5,'read_msh2',return_value=({1:(0.,0.,0.)},[(1,2,3)])),
            patch.object(t.r5,'audit',return_value=topology()),
            patch.object(t.subprocess,'run',side_effect=MockCompleteResult)
        ]
        for p in self.patches:
            p.start();self.addCleanup(p.stop)

    def test_prepare_and_verify(self):
        ans=t.prepare(self.r6b,self.out,'gmsh',8.,180)
        self.assertEqual(ans['refined_triangles'],890)
        self.assertEqual(t.check(self.out)['status'],'M15R7A_CHECK_OK')
        self.assertEqual(t.check(self.out)['solver_run'],'NOT_RUN')

    def test_no_overwrite(self):
        t.prepare(self.r6b,self.out,'gmsh',8.,180)
        with self.assertRaisesRegex(FileExistsError,'NO_OVERWRITE'):
            t.prepare(self.r6b,self.out,'gmsh',8.,180)

    def test_reject_unsupported_edge(self):
        with self.assertRaisesRegex(ValueError,'EDGE'):
            t.prepare(self.r6b,self.out,'gmsh',3.,180)
        self.assertFalse(self.out.exists())

    def test_fail_if_refinement_not_larger(self):
        with patch.object(t.r5,'audit',return_value=topology(200)):
            with self.assertRaisesRegex(ValueError,'REFINEMENT_DID_NOT_ADD_FACES'):
                t.prepare(self.r6b,self.out,'gmsh',8.,180)

    def test_change_mesh_after_preparation(self):
        t.prepare(self.r6b,self.out,'gmsh',8.,180)
        (self.out/'sphere.msh').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'R7A_INPUT_CHANGED'):
            t.check(self.out)

    def test_change_baseline_after_preparation(self):
        t.prepare(self.r6b,self.out,'gmsh',8.,180)
        (self.r6b/'M15R6B_SINGLE_MESH_GATE.json').write_text('{"tampered":true}')
        with self.assertRaisesRegex(ValueError,'BASELINE_CHANGED'):
            t.check(self.out)

    def test_run_audit_and_check(self):
        t.prepare(self.r6b,self.out,'gmsh',8.,180)
        self.assertEqual(t.run(self.out,'scuff-scatter',900)['status'],'REFINED_SCUFF_PFT_PRODUCED_UNCALIBRATED')
        ans=t.audit(self.out)
        self.assertEqual(ans['status'],'REFINED_MIE_COMPARISON_RECORDED')
        self.assertTrue(ans['refined_error_pct']<ans['coarse_error_pct'])
        self.assertFalse(ans['mesh_convergence_demonstrated'])
        self.assertEqual(t.check(self.out)['solver_run'],'REFINED_SCUFF_PFT_PRODUCED_UNCALIBRATED')
        report=(self.out/'M15R7A_AUDIT.json').read_text()
        self.assertIn('BLOCKED',report)
        self.assertIn('TOWARD_MIE',report)

    def test_run_again_blocked(self):
        t.prepare(self.r6b,self.out,'gmsh',8.,180)
        t.run(self.out,'scuff-scatter',900)
        with self.assertRaisesRegex(FileExistsError,'NO_OVERWRITE'):
            t.run(self.out,'scuff-scatter',900)

    def test_audit_again_blocked(self):
        t.prepare(self.r6b,self.out,'gmsh',8.,180)
        t.run(self.out,'scuff-scatter',900);t.audit(self.out)
        with self.assertRaisesRegex(FileExistsError,'NO_OVERWRITE'):
            t.audit(self.out)

    def test_tampered_pft_blocked(self):
        t.prepare(self.r6b,self.out,'gmsh',8.,180)
        t.run(self.out,'scuff-scatter',900)
        (self.out/'sphere.PFT').write_text(pft(1.2e-7))
        with self.assertRaisesRegex(ValueError,'PFT_CHANGED'):
            t.audit(self.out)

    def test_bad_scuff_return_blocks_audit(self):
        t.prepare(self.r6b,self.out,'gmsh',8.,180)
        t.run(self.out,'scuff-scatter',900)
        record=self.out/'M15R7A_RUN.json';obj=json.loads(record.read_text())
        obj['exit_code']=1
        record.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'SCUFF_RUN_NOT_SUCCESSFUL'):
            t.audit(self.out)

    def test_unrefined_source_bad(self):
        self.baseline['sphere_triangle_count']=1500
        with self.assertRaisesRegex(ValueError,'REFINEMENT_DID_NOT_ADD_FACES'):
            t.prepare(self.r6b,self.out,'gmsh',8.,180)

    def test_six_nm_allowed(self):
        ans=t.prepare(self.r6b,self.out,'gmsh',6.,180)
        self.assertEqual(ans['edge_nm'],6.0)


if __name__=='__main__':unittest.main()
