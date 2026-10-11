import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.discovery import run_m15_meep_bootstrap as b

class Model:
    def __init__(self,**kwargs):self.d=kwargs
    def __getattr__(self,name):
        if name in self.d:return self.d[name]
        raise AttributeError(name)
    def model_dump(self,mode=None):
        def conv(v):
            if isinstance(v,Model):return v.model_dump(mode=mode)
            if isinstance(v,list):return [conv(x) for x in v]
            if isinstance(v,dict):return {k:conv(x) for k,x in v.items()}
            return v
        return conv(self.d)
    @classmethod
    def model_validate_json(cls,text):return cls(**json.loads(text))

class FakeJobs(Model):
    def __init__(self,**kw):super().__init__(**kw)

class FakeNum(Model):pass
class FakeBuilder:
    def build_requests(self,jobs,numerical,**kwargs):
        js=jobs.jobs; ps=numerical.plans
        assert len(js)==len(ps)
        req=[]
        for i,(j,p) in enumerate(zip(js,ps)):
            d=j if isinstance(j,dict) else j.d
            q=p if isinstance(p,dict) else p.d
            if d['job_id']!=q['source_solver_job_id']:raise AssertionError('bad job lineage')
            for s in ['geometry_family','core_material','shell_material','nanorod_length_nm','nanorod_diameter_nm','shell_thickness_nm','polarization']:
                if s not in d:raise AssertionError('missing '+s)
            for s in ['minimum_geometric_feature_nm','minimum_feature_cells','resolution_status','intended_use','evidence_eligibility']:
                if s not in q:raise AssertionError('missing '+s)
            req.append(types.SimpleNamespace(source_solver_job_id=d['job_id'],request_id='req:'+d['job_id'],intended_use=q['intended_use'],evidence_eligibility=q['evidence_eligibility']))
        return types.SimpleNamespace(request_count=len(req),requests=req)

def patched():
    a=types.ModuleType('domains');a.__path__=[]
    s=types.ModuleType('domains.sers');s.__path__=[]
    sol=types.ModuleType('domains.sers.solver_execution_contracts');sol.SERSFDTDSolverJob=Model;sol.SERSFDTDSolverJobBundle=FakeJobs
    num=types.ModuleType('domains.sers.numerical_policy_contracts');num.SERSFDTDNumericalJobPlan=Model;num.SERSFDTDNumericalPlanBundle=FakeNum
    mee=types.ModuleType('domains.sers.meep_execution');mee.SERSMeepExecutionPlanner=FakeBuilder
    return patch.dict(sys.modules,{'domains':a,'domains.sers':s,
        'domains.sers.solver_execution_contracts':sol,'domains.sers.numerical_policy_contracts':num,'domains.sers.meep_execution':mee})

class TestBootstrap(unittest.TestCase):
    def test_default_guard(self):
        p=b.plan_data();self.assertGreater(p['estimated_grid_voxels'],0)
        self.assertLessEqual(p['estimated_grid_voxels'],p['max_voxels'])
        self.assertFalse(p['numerical_convergence_certified'])
    def test_small_limit_fails(self):
        with self.assertRaisesRegex(ValueError,'VOXEL_GUARD'):b.plan_data(max_voxels=10000)
    def test_high_resolution_guard(self):
        with self.assertRaisesRegex(ValueError,'VOXEL_GUARD'):b.plan_data(resolution=400)
    def test_no_negative_geometry(self):
        with self.assertRaisesRegex(ValueError,'nanorod'):b.plan_data(length=30,diameter=50)
    def test_invalid_numeric(self):
        with self.assertRaisesRegex(ValueError,'invalid'):b.plan_data(resolution=float('nan'))
    def test_sanity_pair_generation_frozen(self):
        with tempfile.TemporaryDirectory() as t,patched():
            out=Path(t)/'v1'
            ret=b.create(out,120.0,1500000)
            self.assertEqual(ret['pilot_jobs'],4)
            self.assertEqual(ret['independent_case_pairs'],2)
            self.assertEqual(b.check(out)['requests'],4)
            cb=json.loads((out/b.BOOK).read_text())
            self.assertEqual({x['variable_under_test'] for x in cb['cases']},{'polarization'})
            self.assertEqual(len({x['control_request_id'] for x in cb['cases']}),2)
            self.assertFalse(json.loads((out/b.MAN).read_text())['actual_meep_executed'])
            with self.assertRaises(FileExistsError):b.create(out,120,1500000)
    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as t,patched():
            out=Path(t)/'v1';b.create(out,120.0,1500000)
            (out/b.BOOK).write_text('{}')
            with self.assertRaisesRegex(ValueError,'BOOTSTRAP_SOURCE_MUTATED'):b.check(out)
    def test_casebook_questions_predeclared(self):
        for p in b.PAIR_CONFIG:
            self.assertGreater(len(p['research_question']),70)
            self.assertIn('Maxwell',p['strong_null'])
            self.assertIn('independ',p['research_question'].lower())
    def test_version(self):
        self.assertEqual(b.VERSION,'m15-meep-execution-sanity-bootstrap-v1')

if __name__=='__main__':unittest.main()
