from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.discovery import run_m15_r8a_auag_scuff_pilot as m


def fixture():
    # Synthetic fixture never ships as scientific Au/Ag optical data.
    rows=[f'{.19+i*.02:.4f} {1+i*.005:.4f} {1.5+i*.01:.4f}' for i in range(45)]
    return ('DATA:\n  - type: tabulated nk\n    data: |\n        '+'\n        '.join(rows)+'\n').encode()

class R8ATest(unittest.TestCase):
    def test_git_blob(self):
        self.assertEqual(m.git_blob_sha1(b'hello'),hashlib.sha1(b'blob 5\0hello').hexdigest())
    def test_parser(self):
        self.assertEqual(len(m.read_nk(fixture())),45)
    def test_parser_bad_header(self):
        with self.assertRaisesRegex(ValueError,'TABULATED'):
            m.read_nk(fixture().replace(b'tabulated nk',b'tabulated n'))
    def test_parser_bad_order(self):
        raw=fixture().replace(b'0.1900',b'0.2100')
        with self.assertRaisesRegex(ValueError,'WAVELENGTH_NOT_INCREASING'):
            m.read_nk(raw)
    def test_stops_wrong_pin(self):
        with self.assertRaisesRegex(ValueError,'BLOB_MISMATCH'):
            m.source_to_eps(fixture(),'Au',633)
    def test_nk_to_eps(self):
        raw=fixture();h=m.git_blob_sha1(raw)
        with patch.dict(m.SOURCE_BLOB_SHA1,{'Au':h}):
            body,info=m.source_to_eps(raw,'Au',633)
        self.assertEqual(info['data_points'],45)
        self.assertIn('i',body)
        lines=body.strip().splitlines()
        self.assertEqual(len(lines),45)
        # increasing omega, increasing imaginary epsilon for synthetic k.
        self.assertLess(float(lines[0].split()[0]),float(lines[-1].split()[0]))
        self.assertTrue(all('e+' in r.split()[0] or 'e-' in r.split()[0] for r in lines))
    def test_invalid_material(self):
        with self.assertRaisesRegex(ValueError,'UNKNOWN_MATERIAL'):
            m.source_to_eps(fixture(),'Pt',633)
    def test_wavelength_guard(self):
        raw=fixture()
        with patch.dict(m.SOURCE_BLOB_SHA1,{'Au':m.git_blob_sha1(raw)}):
            with self.assertRaisesRegex(ValueError,'OUTSIDE_NK_RANGE'):
                m.source_to_eps(raw,'Au',1200)
    def test_nested_prepare_and_check(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);r4=base/'r4';r5=base/'r5';out=base/'out'
            r4.mkdir();r5.mkdir()
            (r4/'M15R4_BEM_CANDIDATE_PLAN.json').write_text('{}')
            (r5/'M15R5_MESH_GATE.json').write_text('{}')
            for f in ('inner_core.msh','outer_shell.msh'):
                (r5/f).write_text('mocked gmsh mesh')
            aud=base/'aud.json';aud.write_text('{}')
            au=base/'au.yml';ag=base/'ag.yml';raw=fixture();au.write_bytes(raw);ag.write_bytes(raw)
            pl={'core_material':'Au','shell_material':'Ag','medium':'air',
                'first_test_wavelength_nm':633.,'field_component':'Ex',
                'geometry_um':{'inner_length':.16,'inner_diameter':.05,
                               'outer_length':.20,'outer_diameter':.09}}
            with patch.dict(m.SOURCE_BLOB_SHA1,{'Au':m.git_blob_sha1(raw),'Ag':m.git_blob_sha1(raw)}),\
                 patch.object(m,'_load_sources',return_value=(pl,{},{})),\
                 patch.object(m,'_clearance_screen',return_value={
                     'core_vertices_inside_outer_analytic_capsule':True,
                     'min_core_vertex_to_analytic_outer_boundary_um':.019,
                     'triangle_triangle_collision_verified':False}):
                result=m.prepare(r4,r5,aud,au,ag,out)
            self.assertEqual(result['scuff_runs'],0)
            self.assertIn('FILE_Ag_eps.dat',(out/'nanorod.scuffgeo').read_text())
            self.assertIn('MESHFILE inner_core.msh',(out/'nanorod.scuffgeo').read_text())
            self.assertEqual(m.check(out)['science_gate'],'BLOCKED')
            (out/'Ag_eps.dat').write_text('tamper')
            with self.assertRaisesRegex(ValueError,'INPUT_MODIFIED'):
                m.check(out)
    def test_duplicate_output(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            with self.assertRaisesRegex(FileExistsError,'OUTPUT_DIRECTORY_EXISTS'):
                with patch.object(m,'_load_sources',return_value=(None,None,None)):
                    # Need valid sources for duplicate check; this verifies preflight fails before creation.
                    m.prepare(p,p,p,p,p,p)
    def test_explicit_run_receipt_mock(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'fake_scuff').write_text('binary')
            (out/'fake_scuff').chmod(0o755)
            (out/'M15R8A_PLAN.json').write_text('{}')
            plan={'wavelength_nm':633,'polarization':'Ex'}
            class Proc: returncode=0;stdout='ok';stderr=''
            def fake_subprocess(cmd,**kwargs):
                (out/'nanorod.PFT').write_text('test PFT')
                return Proc()
            with patch.object(m,'verify',return_value=plan),patch.object(m.subprocess,'run',side_effect=fake_subprocess):
                receipt=m.run(out,str(out/'fake_scuff'),60,1)
            self.assertEqual(receipt['status'],'AU_AG_PFT_GENERATED_UNCALIBRATED')
            self.assertEqual(json.loads((out/'M15R8A_RUN.json').read_text())['science_backend_qualified'],False)
    def test_timeout_guard(self):
        with self.assertRaisesRegex(ValueError,'TIMEOUT_OUT_OF_RANGE'):
            m.run(Path('.'),'missing',1,2)
    def test_thread_guard(self):
        with self.assertRaisesRegex(ValueError,'THREAD_COUNT_OUT_OF_RANGE'):
            m.run(Path('.'),'missing',100,25)

if __name__=='__main__': unittest.main()
