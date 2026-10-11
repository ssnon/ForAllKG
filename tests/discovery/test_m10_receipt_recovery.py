"""Regression: a paid receipt with a missing equation never triggers a second paid call."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.discovery.run_sers_academic_novelty_v1 import (
    pending_generation_tasks, recover_saved,
)
from pipeline_core.discovery.sers_academic_novelty_v1 import digest, read_json, write_new

FIELDS = {
    "title": "Synthetic novel claim candidate",
    "conceptual_delta": "A conditional mechanism change",
    "novelty_atom": "A testable conditional contrast",
    "mechanism": "A conditional interaction between latent variables",
    "assumptions": ["Constant baseline"],
    "equations": ["p(y|do(x=1)) != p(y|do(x=0))"],
    "strong_nulls": ["Equal likelihood under conditional invariance"],
    "differential_predictions": ["Outcome differs under calibrated intervention"],
    "independent_observables": ["Independent state measurement"],
    "interventions": ["Vary x with control"],
    "falsification_scope": "The contrast fails in preregistered conditions",
    "prior_art_queries": ["q1", "q2", "q3", "q4"],
    "feasibility_limits": ["Assay may be imperfect"],
    "risk_of_known_physics_reexpression": "The baseline may already explain the effect",
}

class M10ReceiptRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)
        self.model = "openai/gpt-5.6-luna"
        self.tasks = [{"task_id":n,"source_g6":["g6:a","g6:b"],"system":"X","user":n}
                      for n in ["SYMMETRY_BREAKING", "HETEROGENEITY_INTERACTION", "MEASUREMENT_LIMIT"]]
        self.plan = {"status":"PREPARED_SEARCH_ONLY_NO_NOVELTY_CLAIM","tasks":self.tasks}
        self.planpath = self.root / "M10_SPRINT_PLAN.json"
        write_new(self.planpath,self.plan)
        write_new(self.root/"model_lock.json",{"model":self.model})

    def _receipt(self, task, *, with_equations=True, alias=False):
        tid=task["task_id"]
        response={"task_id":tid, **FIELDS}
        if not with_equations:
            response.pop("equations")
        if alias:
            response["model_equations"] = ["x(t)=x(0)exp(-kt)"]
        write_new(self.root/"intents"/(tid+".json"),{"task_id":tid})
        write_new(self.root/"receipts"/(tid+".json"), {
            "task_id":tid,"model":self.model,"prompt_sha256":digest(task),
            "finish_reason":"stop","raw_response":json.dumps(response),
            "scientific_truth_authority":False,
        })
        return response

    def test_saved_invalid_receipt_is_skipped_not_recalled(self):
        self._receipt(self.tasks[0]); self._receipt(self.tasks[1],with_equations=False)
        write_new(self.root/"parsed"/(self.tasks[0]["task_id"]+".json"),
                  {"task_id":self.tasks[0]["task_id"],**FIELDS})
        todo,unresolved=pending_generation_tasks(self.plan,self.root,self.model)
        self.assertEqual([t[0]["task_id"] for t in todo],["MEASUREMENT_LIMIT"])
        self.assertEqual(unresolved,["HETEROGENEITY_INTERACTION"])
        reportpath=self.root/"audit.json"
        recover_saved(SimpleNamespace(plan=self.planpath,out=reportpath))
        report=read_json(reportpath)
        self.assertEqual(report["new_provider_calls"],0)
        row=report["results"][1]
        self.assertEqual(row["status"],"SAVED_RESPONSE_SCIENTIFICALLY_INCOMPLETE")
        self.assertIn("equations", row["validation_error"])
        self.assertFalse((self.root/"parsed"/"HETEROGENEITY_INTERACTION.json").exists())

    def test_alias_preserves_model_supplied_actual_equation(self):
        self._receipt(self.tasks[1],with_equations=False,alias=True)
        reportpath=self.root/"audit.json"
        recover_saved(SimpleNamespace(plan=self.planpath,out=reportpath))
        row=read_json(reportpath)["results"][1]
        self.assertEqual(row["status"],"RECOVERED_OFFLINE_VALID")
        self.assertEqual(row["verbatim_field_alias"],"model_equations->equations")
        parsed=read_json(self.root/"parsed"/"HETEROGENEITY_INTERACTION.json")
        self.assertEqual(parsed["equations"],["x(t)=x(0)exp(-kt)"])
        self.assertFalse(parsed["novelty_certified"])
        self.assertFalse(parsed["offline_receipt_recovery"]["scientific_equation_invented"])

    def test_alias_without_equation_is_not_promoted(self):
        self._receipt(self.tasks[1],with_equations=False,alias=True)
        p=self.root/"receipts"/"HETEROGENEITY_INTERACTION.json"
        receipt=read_json(p)
        raw=json.loads(receipt["raw_response"])
        raw["model_equations"]=["A theoretical idea without specified equations"]
        receipt["raw_response"]=json.dumps(raw)
        p.write_text(json.dumps(receipt), encoding="utf-8")
        recover_saved(SimpleNamespace(plan=self.planpath,out=self.root/"audit.json"))
        row=read_json(self.root/"audit.json")["results"][1]
        self.assertEqual(row["status"],"SAVED_RESPONSE_SCIENTIFICALLY_INCOMPLETE")
        self.assertFalse((self.root/"parsed"/"HETEROGENEITY_INTERACTION.json").exists())

    def test_tampered_receipt_never_unblocks_new_calls(self):
        self._receipt(self.tasks[1],with_equations=False)
        p=self.root/"receipts"/"HETEROGENEITY_INTERACTION.json"
        corrupted=read_json(p);corrupted["prompt_sha256"]="wrong"
        p.write_text(json.dumps(corrupted),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"M10_RECEIPT_IDENTITY_OR_PROMPT_MISMATCH"):
            pending_generation_tasks(self.plan,self.root,self.model)

    def test_unknown_intent_must_not_be_recalled(self):
        write_new(self.root/"intents"/"MEASUREMENT_LIMIT.json",{"attempted":True})
        with self.assertRaisesRegex(RuntimeError,"Unresolved paid intent"):
            pending_generation_tasks(self.plan,self.root,self.model)

    def test_recover_never_overwrites_valid_parsed(self):
        self._receipt(self.tasks[0]);p=self.root/"parsed"/"SYMMETRY_BREAKING.json"
        write_new(p,{"task_id":"SYMMETRY_BREAKING",**FIELDS})
        recover_saved(SimpleNamespace(plan=self.planpath,out=self.root/"audit.json"))
        self.assertEqual(read_json(p)["equations"],FIELDS["equations"])

if __name__=="__main__":unittest.main()
