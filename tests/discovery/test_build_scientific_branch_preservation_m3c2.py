"""Hermetic M3-C2 contract tests; synthetic data only, no private artifacts."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.discovery.build_scientific_branch_preservation_m3c2 import (
    create_plan, execute, load_json, sha256, validate_inputs,
)


ARMS = ("V31_FROZEN", "V34_FROZEN", "M3A_NEW")
ROOTS = [f"research_idea:p0_{i}" for i in range(8)]


class TestM3C2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.commit = "synthetic_commit"
        self.lineage = {"status": "M3C0_STRUCTURAL_LINEAGE_AUDIT_PASS_NO_SCIENCE_CERTIFICATION", "code_commit": self.commit, "arms": []}
        self.trajectory = {"status": "SHA_VERIFIED_SCIENTIFIC_DELTAS_READY_FOR_HUMAN_REVIEW", "commit": self.commit,
                           "authoritative_science_judgment": False, "original_data_mutated": False,
                           "root_set_conserved_across_arms": True, "rows": []}
        self.reviews = []
        self.manifest = {}
        for arm in ARMS:
            items, active, historical = [], [], []
            for i, root in enumerate(ROOTS[:7]):
                term = root if arm == "V31_FROZEN" else f"research_idea:{arm}_child_{i}"
                if arm == "M3A_NEW" and i == 0:
                    historical.append(root)
                active.append(term)
                hyp = f"hypothesis:{arm}_{i}"
                blind = f"B{len(self.trajectory['rows'])+1:02d}"
                items.append({"blind_id": blind, "hypothesis_id": hyp, "terminal_research_idea_id": term,
                              "p0_root_research_idea_ids": [root], "lineage_validation": "PASS_STRUCTURAL_ONLY"})
                steps = [{"idea_id": root}, {"idea_id": term}] if root != term else [{"idea_id": root}]
                self.trajectory["rows"].append({"blind_id": blind, "arm": arm, "hypothesis_id": hyp,
                     "terminal_idea_id": term, "p0_root_idea_id": root, "terminal_birth": 3 if root != term else 1,
                     "lineage_validation": "PASS_SOURCE_HASH_AND_KERNEL_TRAJECTORY", "steps_earliest_to_latest": steps})
                self.reviews.append({"blind_id": blind, "arm": arm, "p0_root_id": root,
                    "terminal_idea_id": term, "terminal_birth": str(3 if root != term else 1),
                    "capability_preservation_decision": "PRESERVE_PARENT_PROGRAM" if i == 0 else "KEEP_BASELINE_REFERENCE",
                    "falsifier_logical_validity": "NOT_CHECKED", "decisive_experiment_and_independent_observables": "TBD"})
            digest = hashlib.sha256(arm.encode()).hexdigest()
            self.manifest[f"{arm}.final_portfolio"] = digest
            self.lineage["arms"].append({"arm": arm, "terminal_hypothesis_count": 7,
                "p0_research_idea_ids": ROOTS[:], "items": items,
                "final_sha256": digest, "final_active_research_idea_ids": active,
                "historical_only_idea_ids": historical})
        self.paths = {name: self.base / name for name in (
            "lineage.json", "manifest.json", "trajectory.json", "integrity.json", "review.csv", "review_manifest.json")}
        self._write_inputs()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_inputs(self):
        for k, v in (("lineage.json", self.lineage), ("manifest.json", self.manifest),
                     ("trajectory.json", self.trajectory)):
            self.paths[k].write_text(json.dumps(v), encoding="utf-8")
        with self.paths["review.csv"].open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.reviews[0]))
            writer.writeheader()
            writer.writerows(self.reviews)
        self.integrity = {"original_data_mutated": False,
            "input_m3c0_audit_sha256": sha256(self.paths["lineage.json"]),
            "input_m3c0_manifest_sha256": sha256(self.paths["manifest.json"])}
        self.paths["integrity.json"].write_text(json.dumps(self.integrity), encoding="utf-8")
        self.paths["review_manifest.json"].write_text(json.dumps({
           "status": "POSTLOCK_AI_DRAFT_NOT_EXPERT_CERTIFIED", "commit": self.commit, "row_count": 21,
           "output_files_sha256": {"M3C1_REVIEW_SHEET_AI_POSTLOCK_DRAFT_PRIVATE.csv": sha256(self.paths["review.csv"])},
           "input_files_sha256": {"M3C1_KERNEL_TRAJECTORIES_PRIVATE.json": sha256(self.paths["trajectory.json"]),
                                   "M3C1_INPUT_INTEGRITY_PRIVATE.json": sha256(self.paths["integrity.json"])},
        }), encoding="utf-8")

    def _validate(self):
        return validate_inputs(lineage=self.lineage, trajectory=self.trajectory, review_rows=self.reviews,
            integrity=self.integrity, source_hashes=self.manifest,
            lineage_path=self.paths["lineage.json"], source_hashes_path=self.paths["manifest.json"],
            expected_commit=self.commit)

    def _args(self, output="output"):
        from argparse import Namespace
        return Namespace(lineage=self.paths["lineage.json"], source_hashes=self.paths["manifest.json"],
            trajectories=self.paths["trajectory.json"], integrity=self.paths["integrity.json"],
            review_csv=self.paths["review.csv"], review_manifest=self.paths["review_manifest.json"], output_dir=self.base / output,
            target_arm="M3A_NEW", expected_commit=self.commit)

    def test_hermetic_end_to_end_output(self):
        before = {k: sha256(p) for k, p in self.paths.items()}
        plan = execute(self._args())
        self.assertEqual(plan["lineage_and_review_rows"], 21)
        self.assertEqual(plan["status"], "M3C2_STRUCTURAL_PLAN_READY_HUMAN_SCIENCE_REVIEW_REQUIRED")
        self.assertFalse(plan["simplification_authorized"])
        self.assertFalse(plan["strict_grounding_or_falsifier_validation_executed"])
        self.assertTrue((self.base / "output/M3C2_HUMAN_REVIEW_TEMPLATE_PRIVATE.json").is_file())
        self.assertEqual(before, {k: sha256(p) for k, p in self.paths.items()})

    def test_no_silent_loss_parent_archived(self):
        arms, traj, reviews, commit = self._validate()
        plan = create_plan(arm_rows=arms, trajectories=traj, reviews=reviews,
            target_arm="M3A_NEW", commit=commit, file_hashes={})
        root_parent = next(p for p in plan["proposals_PRIVATE"] if p["source_arm"] == "V31_FROZEN" and p["blind_id"] == "B01")
        self.assertEqual(root_parent["target_reachability"], "HISTORICAL_IDEA_NOT_TERMINAL_REALIZED")
        self.assertTrue(root_parent["scientific_review_REQUIRED"])

    def test_cross_arm_child_is_not_auto_imported(self):
        a, t, r, c = self._validate()
        plan = create_plan(arm_rows=a, trajectories=t, reviews=r, target_arm="M3A_NEW", commit=c, file_hashes={})
        frozen = next(p for p in plan["proposals_PRIVATE"] if p["blind_id"] == "B09")
        self.assertEqual(frozen["target_reachability"], "FROZEN_SOURCE_ARM_ONLY")
        self.assertEqual(frozen["proposed_preservation_action"], "RETAIN_FROZEN_BASELINE_REFERENCE")
        self.assertFalse(plan["cross_arm_hypothesis_import_executed"])

    def test_input_hash_tamper_stops(self):
        self.paths["lineage.json"].write_text(self.paths["lineage.json"].read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self._validate()

    def test_postlock_csv_tampering_rejected(self):
        self.paths["review.csv"].write_text(self.paths["review.csv"].read_text().replace("PRESERVE_PARENT_PROGRAM", "DELETE_ALL"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "review CSV SHA mismatch"):
            execute(self._args())

    def test_missing_review_rejected(self):
        self.reviews.pop()
        with self.assertRaisesRegex(ValueError, "21"):
            self._validate()

    def test_duplicate_blind_rejected(self):
        self.reviews[1]["blind_id"] = self.reviews[0]["blind_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._validate()

    def test_root_mismatch_rejected(self):
        self.reviews[0]["p0_root_id"] = ROOTS[-1]
        with self.assertRaisesRegex(ValueError, "root mismatch"):
            self._validate()

    def test_wrong_decision_rejected(self):
        self.reviews[0]["capability_preservation_decision"] = "DELETE_ALL"
        with self.assertRaisesRegex(ValueError, "decision unknown"):
            self._validate()

    def test_inconsistent_frozen_portfolio_sha_rejected(self):
        self.manifest["V31_FROZEN.final_portfolio"] = "0000"
        with self.assertRaisesRegex(ValueError, "portfolio SHA mismatch"):
            self._validate()

    def test_repository_private_output_rejected(self):
        from argparse import Namespace
        args = self._args()
        args.output_dir = Path(__file__).resolve().parents[2] / "PRIVATE_WRONG_PLACE"
        with self.assertRaisesRegex(ValueError, "OUTSIDE the repository"):
            execute(args)

    def test_overwrite_refused(self):
        execute(self._args())
        with self.assertRaises(FileExistsError):
            execute(self._args())

    def test_unexpected_commit_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexpected HEAD"):
            validate_inputs(lineage=self.lineage, trajectory=self.trajectory, review_rows=self.reviews,
                integrity=self.integrity, source_hashes=self.manifest, lineage_path=self.paths["lineage.json"],
                source_hashes_path=self.paths["manifest.json"], expected_commit="different")

    def test_review_template_never_approves(self):
        execute(self._args())
        data = load_json(self.base / "output/M3C2_HUMAN_REVIEW_TEMPLATE_PRIVATE.json")
        self.assertEqual(data["status"], "UNREVIEWED_NO_AUTHORITY")
        self.assertTrue(all(r["keep_as_separate_research_option"] is None for r in data["decisions"]))


if __name__ == "__main__":
    unittest.main()
