from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_core.discovery.research_idea_scientific_development_m6 import (
    SCHEMA, prepare, develop, load, save_new, validate_proposal,
    validate_drafts, prompt_for_task, file_sha,
)

ROOT = Path(__file__).resolve().parents[2]  # repo_patch/
DEMO = Path(__file__).resolve().parent / "fixtures_m6"
PACKET = DEMO / "source_packet.json"
AUDIT = DEMO / "science_assessments.json"
DRAFTS = DEMO / "M6_ANALYST_SEED_DRAFTS.json"
REVISIONS = DEMO / "M6_ANALYST_REVISIONS.json"


class M6Development(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.temp = Path(self.tmp.name)
        self.packet = self.temp / "packet.json"
        self.audit = self.temp / "audit.json"
        self.packet.write_bytes(PACKET.read_bytes())
        self.audit.write_bytes(AUDIT.read_bytes())
        self.proposals = load(DRAFTS)

    def test_prepare_six_real_sources(self):
        res = prepare(self.packet, self.audit, self.temp / "out")
        self.assertEqual(len(res["tasks"]), 6)
        self.assertFalse(res["authority"]["novelty_certified"])

    def test_idea_immutable(self):
        before = (file_sha(self.packet), file_sha(self.audit))
        prepared = prepare(self.packet, self.audit, self.temp / "out")
        develop(self.temp / "out/M6_PREPARED_TASKS.json", DRAFTS, self.temp / "built", REVISIONS)
        after = (file_sha(self.packet), file_sha(self.audit))
        self.assertEqual(before, after)

    def test_develop_six_and_two_revisions(self):
        prepare(self.packet, self.audit, self.temp / "out")
        res = develop(self.temp / "out/M6_PREPARED_TASKS.json", DRAFTS, self.temp / "built", REVISIONS)
        self.assertEqual(res["branch_count"], 6)
        self.assertEqual(sum(r["revision"] is not None for r in res["development_branches"]), 2)
        self.assertEqual(len({r["branch_id"] for r in res["development_branches"]}), 6)
        self.assertTrue(all(not r["novelty_certified"] for r in res["development_branches"]))
        self.assertTrue(all(r["parent_intact"] for r in res["development_branches"]))

    def test_source_tamper_rejected(self):
        d=load(self.packet);d["records"][0]["differential_prediction"]="tampered"
        self.packet.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "provenance"):
            prepare(self.packet,self.audit,self.temp/"out")

    def test_assessment_differential_conflict(self):
        d=load(self.audit);d["records"][0]["source_differential_prediction"]="tampered"
        d["source_packet_sha256"]=file_sha(self.packet)
        self.audit.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "prediction mismatch"):
            prepare(self.packet,self.audit,self.temp/"out")

    def test_duplicate_source_id_rejected(self):
        d=load(self.packet);d["records"][1]["idea_id"]=d["records"][0]["idea_id"]
        self.packet.write_text(json.dumps(d))
        a=load(self.audit);a["source_packet_sha256"]=file_sha(self.packet)
        self.audit.write_text(json.dumps(a))
        with self.assertRaisesRegex(ValueError,"non-bijective"):
            prepare(self.packet,self.audit,self.temp/"out")

    def test_forged_assessment_authority_rejected(self):
        d=load(self.audit);d["records"][0]["scientific_novelty_certified"]=True
        self.audit.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "prohibited authority"):
            prepare(self.packet,self.audit,self.temp/"out")

    def test_invalid_unknown_parent(self):
        prepare(self.packet,self.audit,self.temp/"out")
        d=copy.deepcopy(self.proposals);d["proposals"][0]["parent_idea_id"]="unknown"
        p=self.temp/"drafts.json";p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,"unregistered source parent"):
            develop(self.temp/"out/M6_PREPARED_TASKS.json",p,self.temp/"built")

    def test_forged_novelty_proposal_rejected(self):
        prepare(self.packet,self.audit,self.temp/"out")
        d=copy.deepcopy(self.proposals);d["proposals"][0]["novelty_certified"]=True
        p=self.temp/"drafts.json";p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,"deny truth/novelty"):
            develop(self.temp/"out/M6_PREPARED_TASKS.json",p,self.temp/"built")

    def test_unknown_schema_field_rejected(self):
        p=copy.deepcopy(self.proposals["proposals"][0]);p["approve_for_production"]=True
        with self.assertRaisesRegex(ValueError,"unknown fields"):
            validate_proposal(p,{p["parent_idea_id"]})

    def test_duplicate_branch_rejected(self):
        prepare(self.packet,self.audit,self.temp/"out")
        d=copy.deepcopy(self.proposals);d["proposals"].append(copy.deepcopy(d["proposals"][0]))
        p=self.temp/"drafts.json";p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,"duplicated"):
            develop(self.temp/"out/M6_PREPARED_TASKS.json",p,self.temp/"built")

    def test_revision_add_branch_rejected(self):
        prepare(self.packet,self.audit,self.temp/"out")
        d=copy.deepcopy(load(REVISIONS));d["proposals"][0]["local_id"]="invented"
        p=self.temp/"revision.json";p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,"only update existing"):
            develop(self.temp/"out/M6_PREPARED_TASKS.json",DRAFTS,self.temp/"built",p)

    def test_no_overwrite(self):
        prepare(self.packet,self.audit,self.temp/"out")
        with self.assertRaises(FileExistsError):
            prepare(self.packet,self.audit,self.temp/"out")

    def test_output_under_source_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.packet,self.audit,self.temp)

    def test_parent_selection_and_prompts(self):
        ids=[r["idea_id"] for r in load(self.packet)["records"]][-2:]
        r=prepare(self.packet,self.audit,self.temp/"out",ids)
        self.assertEqual([t["parent_idea_id"] for t in r["tasks"]],ids)
        system,user=prompt_for_task(r["tasks"][0])
        self.assertIn("claim conservatively",system)
        self.assertIn("known_M1",user)

    def test_paid_guard_without_invoking_provider(self):
        import importlib.util
        script = ROOT / "scripts/discovery/run_research_idea_scientific_development_m6.py"
        spec = importlib.util.spec_from_file_location("m6_cli_test",script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        prepare(self.packet,self.audit,self.temp/"out",[load(self.packet)["records"][0]["idea_id"]])
        from argparse import Namespace
        x=Namespace(prepared=str(self.temp/"out/M6_PREPARED_TASKS.json"),out_dir=str(self.temp/"paid"),
            model="mock",api_key_env="OPENROUTER_API_KEY",base_url="https://invalid",temperature=0.2,
            timeout=1,max_new_calls=1,allow_paid=False,i_authorize_new_llm_calls=False)
        with self.assertRaisesRegex(ValueError,"consent flags"):
            module.run_paid(x)
        self.assertFalse((self.temp/"paid").exists())


class M6PhysicalNullCalibration(unittest.TestCase):
    @staticmethod
    def covariance_window(sigma2,t,tau):
        if t<=0 or tau<=0:raise ValueError
        return 2*sigma2*(t*tau - tau*tau*(-math.expm1(-t/tau)))/t**2

    def test_rate_scaling_does_not_change_stationary_mean(self):
        a,b = .3,.7
        for c in (.1,1,10):
            p=(c*b)/(c*a+c*b)
            self.assertAlmostEqual(p,.7)

    def test_window_variance_changes_with_rate(self):
        self.assertGreater(self.covariance_window(1,1,2), self.covariance_window(1,1,.2))

    def test_window_variance_short_limit(self):
        self.assertAlmostEqual(self.covariance_window(1,1e-4,1),1,places=4)

    def test_separable_site_band_ratio_invariant(self):
        # A_{k,s}=u_k v_s, weighting changes cancel.
        for w in ((.8,.2),(.2,.8)):
            i1=sum(2*v*x for v,x in zip((1,3),w));i2=sum(1*v*x for v,x in zip((1,3),w))
            self.assertAlmostEqual(i1/i2,2)

    def test_nonseparable_site_band_ratio_changes(self):
        def ratio(w):return (4*w+1*(1-w))/(w+3*(1-w))
        self.assertNotAlmostEqual(ratio(.8), ratio(.2))

    def test_scalar_normalization_invariant(self):
        self.assertAlmostEqual((12/20)/(6/20),12/6)

if __name__=="__main__": unittest.main()

# Additional integration tests for actual SIS-native raw contracts (synthetic frozen fixture).
class M6NativeSourceLink(unittest.TestCase):
    def test_native_mismatch_fails(self):
        from pipeline_core.discovery.research_idea_scientific_development_m6 import verify_native_sources
        with tempfile.TemporaryDirectory() as t:
            g3=Path(t)/"g3.json";g4=Path(t)/"g4.json"
            g3.write_text('{"offspring_nodes":[],"semantic_records":[]}')
            g4.write_text('{"offspring_nodes":[],"semantic_records":[]}')
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                verify_native_sources(PACKET,g3,g4)

class M6MockProvider(unittest.TestCase):
    def test_two_call_paid_loop_with_fake_provider(self):
        """No network: fake SDK verifies both opt-in generative stages function."""
        import importlib.util
        import sys
        from types import SimpleNamespace
        from argparse import Namespace
        root=Path(__file__).resolve().parents[2]
        script=root/'scripts/discovery/run_research_idea_scientific_development_m6.py'
        spec=importlib.util.spec_from_file_location('m6_cli_mock',script)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as t:
            t=Path(t)
            parent=load(PACKET)['records'][-1]['idea_id']
            prep=t/'prepared'
            prepare(PACKET,AUDIT,prep,[parent])
            sample=copy.deepcopy(load(DRAFTS)['proposals'][-1])
            sample['source_of_proposal']='MODEL_GENERATED'
            revised=copy.deepcopy(sample);revised['source_of_proposal']='MODEL_REVISED'
            sent=[]
            class Client:
                def __init__(self,**kwargs):self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
                def create(self,**kwargs):
                    sent.append(kwargs)
                    payload = revised if len(sent)==2 else sample
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'proposals':[payload]})))],
                                           usage=SimpleNamespace(model_dump=lambda:{'total_tokens':42}), id='test-mock')
            fake=SimpleNamespace(OpenAI=Client)
            args=Namespace(prepared=str(prep/'M6_PREPARED_TASKS.json'),out_dir=str(t/'stage1'),
                model='mock-model',api_key_env='M6_TEST_KEY',base_url='https://fake.invalid',temperature=.2,
                timeout=10,max_new_calls=1,allow_paid=True,i_authorize_new_llm_calls=True)
            with patch.dict(os.environ,{'M6_TEST_KEY':'test-only'}),patch.dict(sys.modules,{'openai':fake}):
                out=mod.run_paid(args)
                self.assertEqual(out['new_logical_calls'],1)
                stage1=t/'developed'
                develop(args.prepared,t/'stage1/M6_PAID_DRAFTS.json',stage1)
                args.developed=str(stage1/'M6_DEVELOPED_BRANCHES.json')
                args.out_dir=str(t/'stage2')
                out=mod.revise_paid(args)
                self.assertEqual(out['new_logical_calls'],1)
                stage3=t/'final'
                final=develop(args.prepared,t/'stage1/M6_PAID_DRAFTS.json',stage3,
                              t/'stage2/M6_PAID_REVISIONS.json')
                self.assertIsNotNone(final['development_branches'][0]['revision'])
                self.assertFalse(final['authority']['novelty_certified'])
                self.assertEqual(len(sent),2)

class M6SyntheticNativeContract(unittest.TestCase):
    def test_native_success_and_content_tamper(self):
        from pipeline_core.discovery.research_idea_scientific_development_m6 import verify_native_sources
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            packet=load(PACKET)
            files={}
            for gen in (3,4):
                items=[r for r in packet['records'] if r['generation']==gen]
                raw={'offspring_nodes':[], 'semantic_records':[]}
                for r in items:
                    raw['offspring_nodes'].append({k:r[k] for k in ('idea_id','kernel','parent_idea_ids','differential_prediction','discriminating_observation','falsification_condition')})
                    raw['semantic_records'].append({'idea_id':r['idea_id'],'conceptual_change_summary':r['conceptual_change_summary']})
                f=root/f'raw_g{gen}.json';f.write_text(json.dumps(raw))
                packet['source_shas'][f'G{gen}']=file_sha(f)
                files[gen]=f
            p=root/'source.json';p.write_text(json.dumps(packet))
            out=verify_native_sources(p,files[3],files[4])
            self.assertEqual(out['status'],'M6_NATIVE_G3_G4_SOURCE_VERIFIED')
            raw=load(files[4]);raw['offspring_nodes'][0]['differential_prediction']='forged'
            files[4].write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                verify_native_sources(p,files[3],files[4])
