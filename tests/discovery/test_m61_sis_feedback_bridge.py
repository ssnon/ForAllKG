"""M6.2 opt-in bridge: no models, no networks, no mutation of user input."""
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2] # repo_patch
BRIDGE = ROOT / "pipeline_core/discovery/research_idea_m61_feedback_bridge.py"
KIT = HERE.parents[3]
PATCHER = ROOT / "scripts/discovery/m62_patch_sis_runner.py"
HINTS = ROOT / "tests/discovery/fixtures_m62/M61_EVOLUTION_FEEDBACK_HINTS.json"
POP = ROOT / "tests/discovery/fixtures_m62/M61_DIVERGENT_POPULATION.json"

@dataclasses.dataclass
class Prompt:
    task_id: str
    system_prompt: str
    user_prompt: str
    prompt_sha256: str

# Use authentic EpistemicG4Prompt when installed in ForAllKG; isolate fake only for
# standalone package unit tests in a container without ForAllKG.
try:
    from pipeline_core.discovery.research_idea_epistemic_generational_evolution import EpistemicG4Prompt as Prompt
except ModuleNotFoundError:
    name="pipeline_core.discovery.research_idea_epistemic_generational_evolution"
    mod=types.ModuleType(name)
    mod.EpistemicG4Prompt=Prompt
    sys.modules[name]=mod

spec=importlib.util.spec_from_file_location("m62_test_bridge", BRIDGE)
bridge=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=bridge
spec.loader.exec_module(bridge)
pspec=importlib.util.spec_from_file_location("m62_patcher", PATCHER)
patcher=importlib.util.module_from_spec(pspec)
sys.modules[pspec.name]=patcher
pspec.loader.exec_module(patcher)

class MockBackend:
    def __init__(self): self.prompts=[]
    def generate(self,p):self.prompts.append(p);return {"ok":True}
    def repair(self,p,prior,feedback):self.prompts.append(p);return {"ok":True}

class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.b=bridge.load_m61_feedback_bundle(HINTS,POP)
        self.parents=list(self.b.by_parent)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tmp=Path(self.temp.name)

    def copied(self):
        h=self.tmp/"h.json";p=self.tmp/"p.json"
        h.write_bytes(HINTS.read_bytes());p.write_bytes(POP.read_bytes())
        return h,p

    def test_six_model_generated_hints_two_parents(self):
        self.assertEqual(len(self.parents),2)
        self.assertEqual(sum(map(len,self.b.by_parent.values())),6)
        for rows in self.b.by_parent.values():
            self.assertEqual([x["exploration_operator_hint"] for x in rows],list(bridge.ROLE_ORDER))
            self.assertEqual({x["source_of_proposal"] for x in rows},{"MODEL_GENERATED"})

    def test_sha_tamper_fail(self):
        h,p=self.copied()
        d=json.loads(p.read_text());d["forks"][0]["fork_rationale"] += " changed"
        p.write_text(json.dumps(d),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):
            bridge.load_m61_feedback_bundle(h,p)

    def test_hints_tamper_wrong_parent_fail(self):
        h,p=self.copied()
        d=json.loads(h.read_text());d["evolution_hints"][0]["suggested_parent_idea_id"]="research_idea:FORGED"
        h.write_text(json.dumps(d),encoding="utf-8")
        with self.assertRaises(ValueError):bridge.load_m61_feedback_bundle(h,p)

    def test_authority_tamper_fail(self):
        h,p=self.copied()
        d=json.loads(h.read_text());d["authority"]["scientific_truth_authority"]=True
        h.write_text(json.dumps(d),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"authority"):
            bridge.load_m61_feedback_bundle(h,p)

    def test_unreviewed_production_guard(self):
        h,p=self.copied()
        d=json.loads(p.read_text());d["forks"][0]["production_selection_authority"]=True
        p.write_text(json.dumps(d),encoding="utf-8")
        d=json.loads(h.read_text());d["source_population_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
        h.write_text(json.dumps(d),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"authority"):
            bridge.load_m61_feedback_bundle(h,p)

    def test_attack_lineage_guard(self):
        h,p=self.copied()
        d=json.loads(h.read_text());d["evolution_hints"][0]["unresolved_attacks"][0]["attack_id"]="m61_attack:FORGED"
        h.write_text(json.dumps(d),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"attack IDs"):
            bridge.load_m61_feedback_bundle(h,p)

    def test_selected_only_no_parent_selection_change(self):
        selected=[self.parents[0],"research_idea:other"]
        report=self.b.selection_report(selected)
        self.assertEqual(report["selected_parent_idea_ids"],selected)
        self.assertEqual(report["hint_matched_count"],3)
        self.assertEqual(report["hint_deferred_count"],3)
        self.assertFalse(report["parent_selection_modified"])

    def test_prompt_augmented_for_selected(self):
        backend=MockBackend()
        ad=bridge.M61FeedbackPromptAdapter(backend=backend,bundle=self.b,task_to_idea={"one":self.parents[0],"two":"research_idea:unselected"})
        src=Prompt("one","base sys",json.dumps({"original":"kept"}),"original-hash")
        ad.generate(src)
        actual=backend.prompts[-1]
        parsed=json.loads(actual.user_prompt)
        self.assertEqual(parsed["original"],"kept")
        ctx=parsed["m61_adversarial_development_search_context_only"]
        self.assertEqual(len(ctx["branch_suggestions"]),3)
        self.assertIn("NON-AUTHORITATIVE",actual.system_prompt)
        self.assertNotEqual(actual.prompt_sha256,"original-hash")
        self.assertEqual(ctx["parent_id"],self.parents[0])
        ad.repair(src,{},"feedback")
        self.assertEqual(actual.prompt_sha256,backend.prompts[-1].prompt_sha256)

    def test_unselected_prompt_bitwise_unchanged(self):
        backend=MockBackend()
        ad=bridge.M61FeedbackPromptAdapter(backend=backend,bundle=self.b,task_to_idea={"none":"research_idea:else"})
        src=Prompt("none","base sys","raw text","original-hash")
        ad.generate(src)
        self.assertIs(backend.prompts[-1],src)

    def test_no_network_no_SIS_no_claim_authority(self):
        report=self.b.selection_report(self.parents)
        for key in ("ResearchIdea_population_modified","SIS_generation_executed","novelty_certified","scientific_truth_authority","production_selection_authority"):
            self.assertFalse(report[key])

    def test_inputs_byte_identical(self):
        before=(bridge.file_sha256(HINTS),bridge.file_sha256(POP))
        bridge.bounded_feedback_context(self.b,self.parents[0])
        after=(bridge.file_sha256(HINTS),bridge.file_sha256(POP))
        self.assertEqual(before,after)

class PatcherTest(unittest.TestCase):
    @staticmethod
    def synthetic():
        # Matching anchors drawn from exact newSchema_proxy runner; not a fake validation of the entire 799-line source.
        return '''from __future__ import annotations
from pipeline_core.discovery.research_idea_offspring_execution import (\n    X,\n)
def parser():
    p.add_argument("--generate-next", action="store_true")
    return p
def main():
    args = parser().parse_args()
    feedback_cache = args.feedback_cache_dir.expanduser().resolve() if args.feedback_cache_dir else None
    _write(sp / f"sis_v3_4.g{generation + 1}_generation_plan.json", next_plan)
    if next_plan.task_count:
        adapter = NoveltyAwareOffspringBackendAdapter()
        raw_next, prompts = execute_epistemic_generation(
            backend=adapter,
        )
        if args.save_prompts:
            _write_augmented_prompts(work / "augmented_generation_prompts", adapter)
'''

    def test_source_patch_optin(self):
        code=patcher.patch_source(self.synthetic())
        self.assertIn("--m61-feedback-hints",code)
        self.assertIn("M61FeedbackPromptAdapter",code)
        self.assertIn("prompt_capture_adapter",code)
        compile(code,"runner","exec")

    def test_duplicate_patch_refused(self):
        code=patcher.patch_source(self.synthetic())
        with self.assertRaisesRegex(ValueError,"ALREADY_PATCHED"):patcher.patch_source(code)

    def test_anchor_drift_refused(self):
        src=self.synthetic().replace("NoveltyAwareOffspringBackendAdapter", "renamed")
        with self.assertRaisesRegex(ValueError,"ANCHOR_MISMATCH"):patcher.patch_source(src.replace('        raw_next, prompts = execute_epistemic_generation(', '        renamed_call('))

if __name__=="__main__":unittest.main()
