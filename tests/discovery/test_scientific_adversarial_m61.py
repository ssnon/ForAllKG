import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_core.discovery.research_idea_scientific_adversarial_m61 import (
    SCHEMA, REGIME, RESIDENCE, ROLES, _calibrated_attacks, assert_authority,
    build_analyst_seed_deltas, materialize_branches, merge_model_attacks,
    prepare_challenges, print_witness_checks, validate_cases, validate_delta, export_evolution_feedback,
)
from pipeline_core.discovery.research_idea_scientific_development_m6 import load, sha, save_new
from scripts.discovery.run_research_idea_scientific_adversarial_m61 import critique_paid, branch_paid

SRC=Path(__file__).parent/"fixtures_m61/M6_REAL_FINAL_DEVELOPMENT.json"

class AdversarialTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.case_dir=self.root/"cases"
        self.cases=prepare_challenges(SRC,self.case_dir)
        self.case_path=self.case_dir/"M61_ADVERSARIAL_CASES.json"
    def test_actual_pilot_six_countermodels(self):
        self.assertEqual(2,self.cases["case_count"])
        self.assertEqual(6,sum(map(lambda c:len(c["attacks"]),self.cases["cases"])))
        self.assertEqual({REGIME,RESIDENCE},{c["parent_idea_id"] for c in self.cases["cases"]})
    def test_source_phase_revised_not_initial(self):
        self.assertTrue(all(c["source_phase"]=="REVISION" for c in self.cases["cases"]))
    def test_proposal_specific_attacks_not_old_static_counterexample(self):
        for c in self.cases["cases"]:
            atexts=[a["countermodel"] for a in c["attacks"]]
            self.assertEqual(3,len(set(atexts)))
            self.assertTrue(all(a["target_claim_exact"]==c["source_proposal"][a["target_field"]] for a in c["attacks"]))
    def test_witness_ratio_exact(self):
        d=print_witness_checks()
        self.assertEqual(2,d["regime_ratio"])
        self.assertEqual(2,d["residence_ratio_1"])
        self.assertEqual(2,d["residence_ratio_2"])
    def test_cases_authority_forgery_rejected(self):
        bad=copy.deepcopy(self.cases);bad["authority"]["production_selection_authority"]=True
        with self.assertRaises(ValueError):validate_cases(bad)
    def test_unstructured_countermodel_witness_rejected(self):
        bad=copy.deepcopy(self.cases);bad["cases"][0]["attacks"][0]["witness"]="unsupported"
        with self.assertRaisesRegex(ValueError,"structured"):validate_cases(bad)
    def test_mutated_source_proposal_rejected(self):
        bad=copy.deepcopy(self.cases)
        bad["cases"][0]["source_proposal"]["research_question"]+=" FAKED"
        with self.assertRaisesRegex(ValueError,"source proposal mutated"):validate_cases(bad)
    def test_modified_attack_target_exact_rejected(self):
        bad=copy.deepcopy(self.cases)
        bad["cases"][0]["attacks"][0]["target_claim_exact"]="UNSUPPORTED CLAIM"
        with self.assertRaisesRegex(ValueError,"grounded"):validate_cases(bad)
    def test_authority_forgery_rejected(self):
        bad=copy.deepcopy(self.cases)
        bad["cases"][0]["attacks"][0]["scientific_truth_authority"]=True
        with self.assertRaises(ValueError):validate_cases(bad)
    def test_source_shadow_authority_forgery_rejected(self):
        bad=load(SRC);bad["authority"]["novelty_certified"]=True
        f=self.root/"forged.json";save_new(f,bad)
        with self.assertRaises(ValueError):prepare_challenges(f,self.root/"forged_out")
    def test_non_m6_source_rejected(self):
        f=self.root/"wrong.json";save_new(f,{"schema_version":SCHEMA,"development_branches":[]})
        with self.assertRaises(ValueError):prepare_challenges(f,self.root/"wrong_out")
    def test_duplicate_parent_rejected(self):
        bad=copy.deepcopy(self.cases);bad["cases"][1]["parent_idea_id"]=bad["cases"][0]["parent_idea_id"]
        with self.assertRaisesRegex(ValueError,"duplicate parent"):validate_cases(bad)
    def test_branch_analyst_six_per_role(self):
        d=build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        self.assertEqual(6,len(d["deltas"]))
        self.assertEqual({r for r in ROLES},{x["branch_role"] for x in d["deltas"]})
        self.assertEqual({"ANALYST_SEED"},{x["source_of_proposal"] for x in d["deltas"]})
    def test_materialize_retains_parents_and_unreviewed(self):
        build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        p=materialize_branches(self.case_path,self.root/"seeds/M61_ANALYST_DELTAS.json",self.root/"pop")
        self.assertEqual(6,p["fork_count"])
        self.assertEqual(2,p["parent_count"])
        self.assertTrue(all(x["proposal"]["parent_idea_id"]==x["parent_idea_id"] for x in p["forks"]))
        self.assertTrue(all(x["scientific_review_status"]=="PROPOSED_UNREVIEWED" for x in p["forks"]))
        self.assertTrue(all(x["countermodel_resolved"]=="NOT_ASSESSED" for x in p["forks"]))
        self.assertTrue(all(x["novelty_certified"] is False for x in p["forks"]))
    def test_source_original_unchanged(self):
        original=SRC.read_bytes();build_analyst_seed_deltas(self.case_path,self.root/"seed")
        self.assertEqual(original,SRC.read_bytes())
    def test_duplicate_role_rejected(self):
        d=build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        d["deltas"][1]["branch_role"]=d["deltas"][0]["branch_role"]
        p=self.root/"edited.json";save_new(p,d)
        with self.assertRaises(ValueError):materialize_branches(self.case_path,p,self.root/"pop")
    def test_wrong_parent_attack_reference_rejected(self):
        d=build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        d["deltas"][0]["attack_ids"]=d["deltas"][-1]["attack_ids"]
        with self.assertRaisesRegex(ValueError,"parent-specific"):validate_delta(d["deltas"][0],self.cases["cases"][0],"ANALYST_SEED")
    def test_missing_role_required_science_rejected(self):
        d=build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        del d["deltas"][0]["changed_fields"]["restricted_null"]
        with self.assertRaisesRegex(ValueError,"role-specific"):validate_delta(d["deltas"][0],self.cases["cases"][0],"ANALYST_SEED")
    def test_unapproved_new_authority_rejected(self):
        d=build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        d["deltas"][0]["scientific_truth_authority"]=True
        with self.assertRaises(ValueError):validate_delta(d["deltas"][0],self.cases["cases"][0],"ANALYST_SEED")
    def test_sha_tamper_in_deltas_rejected(self):
        d=build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        d["cases_sha256"]="bad_hash"
        p=self.root/"tampered.json";save_new(p,d)
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):materialize_branches(self.case_path,p,self.root/"pop")
    def test_overwrite_rejected(self):
        build_analyst_seed_deltas(self.case_path,self.root/"seed")
        with self.assertRaises(FileExistsError):build_analyst_seed_deltas(self.case_path,self.root/"seed")
    def test_export_feedback_is_non_authoritative_and_linked(self):
        build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        materialize_branches(self.case_path,self.root/"seeds/M61_ANALYST_DELTAS.json",self.root/"pop")
        p=export_evolution_feedback(self.case_path,self.root/"pop/M61_DIVERGENT_POPULATION.json",self.root/"feedback")
        self.assertEqual(6,p["hint_count"])
        self.assertFalse(p["authority"]["SIS_generation_executed"])
        self.assertTrue(all(x["idea_identity_verified"] is False for x in p["evolution_hints"]))
        self.assertTrue(all(x["unresolved_attacks"] for x in p["evolution_hints"]))
    def test_export_feedback_rejects_unrelated_source(self):
        build_analyst_seed_deltas(self.case_path,self.root/"seeds")
        materialize_branches(self.case_path,self.root/"seeds/M61_ANALYST_DELTAS.json",self.root/"pop")
        p=self.root/"altered_case.json";doc=load(self.case_path);doc["test_untrusted_metadata"]="changed";save_new(p,doc)
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):
            export_evolution_feedback(p,self.root/"pop/M61_DIVERGENT_POPULATION.json",self.root/"feedback")
    def test_broad_existing_null_preserved(self):
        for c in self.cases["cases"]:
            self.assertGreater(len(c["source_proposal"]["known_physics_envelope"]),100)
            self.assertEqual("KNOWN_PHYSICS_WITNESS_NOT_SCIENTIFIC_VERDICT",c["calibration_status"])
    def test_unknown_family_not_invent_science(self):
        fake=copy.deepcopy(load(SRC))
        row=fake["development_branches"][0]
        row["parent_idea_id"]="research_idea:other"
        row["initial_proposal"]["parent_idea_id"]="research_idea:other"
        row["revision"]["parent_idea_id"]="research_idea:other"
        fake["development_branches"]=[row];fake["branch_count"]=1
        p=self.root/"unknown.json";save_new(p,fake)
        doc=prepare_challenges(p,self.root/"unknown_out")
        self.assertEqual([],doc["cases"][0]["attacks"])
        self.assertIn("NO_DOMAIN_CALIBRATION",doc["cases"][0]["calibration_status"])
    def test_critique_paid_rejects_unapproved(self):
        args=types.SimpleNamespace(cases=str(self.case_path),out_dir=str(self.root/"paid"),
             max_new_calls=2,allow_paid=False,i_authorize_new_llm_calls=False)
        with self.assertRaisesRegex(ValueError,"authorization"):critique_paid(args)
    def test_branch_paid_rejects_call_budget(self):
        args=types.SimpleNamespace(cases=str(self.case_path),out_dir=str(self.root/"paid"),
             max_new_calls=3,allow_paid=True,i_authorize_new_llm_calls=True)
        with self.assertRaisesRegex(ValueError,"1..2"):branch_paid(args)

class FakePaidStages(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.case_path=self.root/"cases/M61_ADVERSARIAL_CASES.json"
        prepare_challenges(SRC,self.root/"cases")
        self.analyst=build_analyst_seed_deltas(self.case_path,self.root/"seed")
    def _args(self,mode):
        return types.SimpleNamespace(cases=str(self.case_path),out_dir=str(self.root/mode),
            max_new_calls=2,allow_paid=True,i_authorize_new_llm_calls=True,
            api_key_env="FAKE_M61_KEY",base_url="https://invalid.test",timeout=1,
            model="fake/model",temperature=0)
    def _fake_module(self,docs):
        state={"count":0}
        class Reply:
            def __init__(self,content):
                self.choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
                self.id="fake-provider-id";self.usage=None
        class Client:
            def __init__(self,**kwargs):pass
            @property
            def chat(self):
                def create(**kw):
                    idx=state["count"];state["count"]+=1
                    return Reply(json.dumps(docs[idx]))
                return types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        return types.SimpleNamespace(OpenAI=Client),state
    def test_model_critique_to_divergent_paid_mock_e2e(self):
        cases=validate_cases(load(self.case_path))
        attackdocs=[]
        for case in cases:
            a=copy.deepcopy(case["attacks"][0]);del a["attack_id"]
            attackdocs.append({"attacks":[a]})
        mod,state=self._fake_module(attackdocs)
        with patch.dict(os.environ,{"FAKE_M61_KEY":"not-a-real-key"}),patch.dict(sys.modules,{"openai":mod}):
            r=critique_paid(self._args("paid_critic"))
        self.assertEqual(2,r["new_logical_calls"]);self.assertEqual(2,state["count"])
        model_path=self.root/"paid_critic/M61_MODEL_ATTACKS.json"
        merged=merge_model_attacks(self.case_path,model_path,self.root/"merged")
        self.assertEqual(8,sum(len(c["attacks"]) for c in merged["cases"]))
        enriched_path=self.root/"merged/M61_ENRICHED_CASES.json"
        new_cases=validate_cases(load(enriched_path))
        docs=[]
        for case in new_cases:
            mine=[copy.deepcopy(d) for d in self.analyst["deltas"] if d["parent_idea_id"]==case["parent_idea_id"]]
            for x in mine:x["source_of_proposal"]="MODEL_GENERATED"
            docs.append({"deltas":mine})
        mod,state=self._fake_module(docs)
        args=self._args("paid_branches");args.cases=str(enriched_path)
        with patch.dict(os.environ,{"FAKE_M61_KEY":"not-a-real-key"}),patch.dict(sys.modules,{"openai":mod}):
            r=branch_paid(args)
        self.assertEqual(2,r["new_logical_calls"]);self.assertEqual(2,state["count"])
        self.assertEqual(6,len(load(self.root/"paid_branches/M61_MODEL_DELTAS.json")["deltas"]))
        p=materialize_branches(enriched_path,self.root/"paid_branches/M61_MODEL_DELTAS.json",self.root/"population")
        self.assertEqual(6,p["fork_count"])
        self.assertEqual({"MODEL_GENERATED"},{x["provenance"] for x in p["forks"]})
        self.assertFalse(p["authority"]["novelty_certified"])
    def test_fake_provider_cannot_sneak_truth_authority(self):
        cases=validate_cases(load(self.case_path)); docs=[]
        for case in cases:
            a=copy.deepcopy(case["attacks"][0]);a["scientific_truth_authority"]=True
            docs.append({"attacks":[a]})
        mod,_=self._fake_module(docs)
        with patch.dict(os.environ,{"FAKE_M61_KEY":"x"}),patch.dict(sys.modules,{"openai":mod}):
            with self.assertRaises(ValueError):critique_paid(self._args("bad_paid"))
        self.assertTrue((self.root/"bad_paid/CALL_1_CRITIQUE_RAW.json").exists())
        self.assertFalse((self.root/"bad_paid/M61_MODEL_ATTACKS.json").exists())
