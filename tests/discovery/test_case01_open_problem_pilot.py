"""Eight no-network smoke/contract tests for Case01 open-problem pilot."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

THIS=Path(__file__).resolve()
ROOT=THIS.parents[2]
# After add-only repo installation, frozen sources live at CASE01_PILOT_KIT,
# not in the repository. Locally, packaged source data is adjacent.
SOURCES=Path(os.environ["CASE01_PILOT_KIT"]) if os.getenv("CASE01_PILOT_KIT") else ROOT/"frozen_sources"
spec=importlib.util.spec_from_file_location("case01_pilot",ROOT/"scripts/discovery/run_case01_open_problem_pilot.py")
assert spec and spec.loader
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=SOURCES.resolve()
        pilot.source_check(cls.source)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.out=Path(self.tmp.name)/"pilot"

    def tearDown(self):
        self.tmp.cleanup()

    def setup_out(self):
        return pilot.prepare(self.source,self.out,"openai/gpt-5.6-luna",0.7,8500)

    def test_sources_are_frozen(self):
        p=pilot.source_check(self.source)
        self.assertEqual(p["source_abstracts"],12)
        self.assertEqual(p["original_native_candidates"],29)
        self.assertEqual(p["selected_lineage_count"],8)

    def test_common_science_bitwise_identical(self):
        base=pilot.prompt(self.source,"PROBLEM_ONLY")
        fac=pilot.prompt(self.source,"FAKG_LINEAGE_CONTEXT")
        a=json.loads(base["messages"][1]["content"])
        b=json.loads(fac["messages"][1]["content"])
        self.assertEqual(a["shared_evidence"],b["shared_evidence"])
        self.assertEqual(base["original_pdf_images"],fac["original_pdf_images"])
        self.assertNotIn("supplementary_inspiration_only",a)
        self.assertEqual(len(b["supplementary_inspiration_only"]["eight_retained_shadow_candidates"]),8)
        self.assertEqual(len(b["supplementary_inspiration_only"]["two_direct_ye_candidates_excluded_from_native_selection"]),2)
        for private in ["spectral pivot","conditional covariance error","strong-null mathematical review"]:
            self.assertNotIn(private,base["messages"][1]["content"].lower())

    def test_prepare_verify_no_overwrite(self):
        self.setup_out()
        self.assertEqual(pilot.verify(self.out)["status"],"FROZEN_PROVENANCE_AND_PROMPTS_PASS")
        with self.assertRaises(FileExistsError):self.setup_out()

    def test_prompt_mutation_fails_closed(self):
        self.setup_out()
        p=self.out/"prompts"/"PROBLEM_ONLY.json"
        p.write_text(p.read_text()+" ",encoding="utf-8")
        with self.assertRaises(ValueError):pilot.verify(self.out)

    def test_strict_schema_and_abstain(self):
        with self.assertRaises(ValueError):pilot.parse_candidates('{"candidates":[],"abstention_reason":""}')
        with self.assertRaises(ValueError):pilot.parse_candidates('{"candidates":[],"abstention_reason":"valid","abstention_reason":"oops"}')
        self.assertEqual(pilot.parse_candidates('{"candidates":[],"abstention_reason":"unidentifiable"}')["candidates"],[])

    def test_cli_paid_gate_blocks_without_key_or_api(self):
        self.setup_out()
        code=ROOT/"scripts/discovery/run_case01_open_problem_pilot.py"
        env={k:v for k,v in os.environ.items() if k not in ("CASE01_ALLOW_PAID","OPENROUTER_API_KEY")}
        p=subprocess.run([sys.executable,str(code),"run-paid","--out",str(self.out),
                   "--arm","PROBLEM_ONLY"],capture_output=True,text=True,env=env)
        self.assertEqual(p.returncode,2)
        self.assertIn("PAID_BLOCKED_EXPLICIT_AUTHORIZATION_REQUIRED",p.stderr)
        self.assertFalse((self.out/"attempts").exists())

    def test_both_abstain_blind_is_valid(self):
        self.setup_out()
        for arm in pilot.ARMS:
            raw=json.dumps({"candidates":[],"abstention_reason":"No distinct test survived."})
            pilot.create_json(self.out/"responses"/f"{arm}.json",
                {"finish_reason":"stop","prompt_sha256":pilot.read(self.out/"RUN_PLAN.json")["prompt_sha256"][arm],"raw":raw})
            pilot.create_json(self.out/"parsed"/f"{arm}.json",pilot.parse_candidates(raw))
        obj=pilot.blind(self.out)
        self.assertEqual(obj["candidates"],0)
        p=pilot.read(self.out/"BLIND_REVIEW_PUBLIC.json")
        self.assertEqual(p["abstention_occurrence_without_arm"],2)

    def test_blinding_with_mock_replies_no_paid_calls(self):
        self.setup_out()
        example=pilot.SCHEMA_EXAMPLE["candidates"][0].copy()
        for k,sample in example.items(): example[k]=["mock derivation"] if isinstance(sample,list) else "mock proposition"
        for arm in pilot.ARMS:
            raw=json.dumps({"candidates":[example],"abstention_reason":""})
            pilot.create_json(self.out/"responses"/f"{arm}.json",
                {"finish_reason":"stop","prompt_sha256":pilot.read(self.out/"RUN_PLAN.json")["prompt_sha256"][arm],"raw":raw})
            pilot.create_json(self.out/"parsed"/f"{arm}.json",pilot.parse_candidates(raw))
        out=pilot.blind(self.out)
        self.assertEqual(out["candidates"],2)
        pub=pilot.read(self.out/"BLIND_REVIEW_PUBLIC.json")
        priv=pilot.read(self.out/"BLIND_ASSIGNMENT_PRIVATE.json")
        self.assertEqual(len(pub["items"]),2)
        self.assertEqual(set(priv),{"B01","B02"})
        self.assertNotIn("FAKG_LINEAGE_CONTEXT",json.dumps(pub))
        self.assertNotIn("PROBLEM_ONLY",json.dumps(pub))

if __name__=="__main__":unittest.main(verbosity=2)
