import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from scripts.discovery.run_m15_r5_mesh_verify import audit, read_msh2, load_r4

# Outward-oriented tetrahedron, volume 1/6, closed triangulated 2-manifold.
N={1:(0.,0.,0.),2:(1.,0.,0.),3:(0.,1.,0.),4:(0.,0.,1.)}
T=[(1,3,2),(1,2,4),(2,3,4),(3,1,4)]

class TestM15R5(unittest.TestCase):
    def test_tetra_closed(self):
        r=audit(N,T,1,1,.5)
        self.assertEqual(r['boundary_edges'],0)
        self.assertEqual(r['nonmanifold_edges'],0)
        self.assertEqual(r['orientation_inconsistent_edges'],0)
        self.assertAlmostEqual(r['signed_volume_um3'],1/6)

    def test_open_mesh(self):
        self.assertEqual(audit(N,T[:-1],1,1,.5)['status'],'MESH_GEOMETRY_REVIEW_REQUIRED')

    def test_reverse_mesh(self):
        self.assertEqual(audit(N,[tuple(reversed(t)) for t in T],1,1,.5)['status'],'MESH_GEOMETRY_REVIEW_REQUIRED')

    def test_orientation_break(self):
        bad=T.copy();bad[0]=tuple(reversed(bad[0]))
        self.assertGreater(audit(N,bad,1,1,.5)['orientation_inconsistent_edges'],0)

    def test_duplicate_face(self):
        with self.assertRaises(ValueError):audit(N,T+[T[0]],1,1,.5)

    def test_msh2_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.msh';p.write_text('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n4\n'+''.join(f'{k} {v[0]} {v[1]} {v[2]}\n' for k,v in N.items())+'$EndNodes\n$Elements\n4\n'+''.join(f'{i} 2 2 101 {i} {t[0]} {t[1]} {t[2]}\n' for i,t in enumerate(T,1))+'$EndElements\n')
            n,t=read_msh2(p);self.assertEqual(n,N);self.assertEqual(t,T)

    def test_disallow_msh4(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.msh';p.write_text('$MeshFormat\n4.1 0 8\n$EndMeshFormat\n')
            with self.assertRaises(ValueError):read_msh2(p)

    def test_r4_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'src').write_text('hello')
            rows={}
            for f in ['inner_core.geo','outer_shell.geo','candidate.scuffgeo.template','scuff-scatter.args.template']:
                (root/f).write_text(f);rows[f]=hashlib.sha256(f.encode()).hexdigest()
            (root/'M15R4_BEM_CANDIDATE_PLAN.json').write_text(json.dumps({'schema_version':'m15r4-bem-candidate-v0','source_request':str(root/'src'),'source_request_sha256':hashlib.sha256(b'hello').hexdigest(),'generated_files':rows}))
            self.assertIn('source_request',load_r4(root))
            (root/'src').write_text('tamper')
            with self.assertRaises(ValueError):load_r4(root)
