from types import SimpleNamespace
import hashlib
import unittest

from pipeline_core.discovery.novelty_specification_source_trace import (
    trace_specification_sources,
)


class SourceTraceTests(unittest.TestCase):
    def hypothesis(self, **changes):
        values = dict(
            hypothesis_id="hypothesis:trace-test",
            inferential_bridge="금속 종류가 관계를 바꾼다.",
            assumptions=[],
            predicted_observations=[SimpleNamespace(
                observable="관계의 차이", expected_direction="unspecified",
                rationale="흡착 강도를 통제한 뒤에도 차이가 남는다.")],
            falsification_criteria=[SimpleNamespace(
                observable="관계의 차이", falsifying_outcome="차이가 없다.")],
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def field(self, h, raw, accepted="", name="required_bridge"):
        return trace_specification_sources(h, {name: raw}, {name: accepted})["fields"][name]

    def test_exact_unicode_offsets_and_hash_are_recoverable(self):
        h = self.hypothesis()
        f = self.field(h, "관계를 바꾼다.", "관계를 바꾼다.")
        m = f["raw_exact_matches"][0]
        self.assertEqual(h.inferential_bridge[m["start"]:m["end"]], m["quote"])
        self.assertEqual(m["source_sha256"], hashlib.sha256(h.inferential_bridge.encode()).hexdigest())
        self.assertEqual(f["sanitizer_state"], "RETAINED")

    def test_empty_draft_does_not_recover_nonempty_source(self):
        f = self.field(self.hypothesis(), "")
        self.assertEqual(f["sanitizer_state"], "EMPTY")
        self.assertEqual(f["raw_exact_matches"], [])
        self.assertEqual(f["nonempty_source_paths"], ["inferential_bridge"])

    def test_rejected_exact_source_is_not_promoted(self):
        h = self.hypothesis()
        f = self.field(h, h.inferential_bridge)
        self.assertEqual(f["raw_source_match_state"], "EXACT_UNIQUE")
        self.assertEqual(f["sanitizer_state"], "REJECTED")
        self.assertEqual(f["accepted_exact_matches"], [])

    def test_duplicate_source_matches_are_not_silently_selected(self):
        h = self.hypothesis(assumptions=["금속 종류가 관계를 바꾼다."])
        f = self.field(h, h.inferential_bridge)
        self.assertEqual(f["raw_source_match_state"], "EXACT_MULTIPLE")
        self.assertEqual(len(f["raw_exact_matches"]), 2)

    def test_paraphrase_is_not_declared_scientifically_missing(self):
        h = self.hypothesis()
        report = trace_specification_sources(h, {"required_bridge": "금속에 따라 관계가 달라진다."}, {})
        self.assertEqual(report["fields"]["required_bridge"]["raw_source_match_state"], "NO_EXACT_MATCH")
        self.assertFalse(report["scientific_sufficiency_assessed"])
        self.assertFalse(report["branch_attribution_assessed"])

    def test_prediction_rationale_and_falsifier_keep_separate_paths(self):
        h = self.hypothesis()
        p = self.field(h, h.predicted_observations[0].rationale, name="predicted_observation")
        f = self.field(h, "차이가 없다.", name="falsification_condition")
        self.assertEqual(p["raw_exact_matches"][0]["source_path"], "predicted_observations[0].rationale")
        self.assertEqual(f["raw_exact_matches"][0]["source_path"], "falsification_criteria[0].falsifying_outcome")

    def test_missing_source_and_whitespace_change_remain_distinct(self):
        f = self.field(self.hypothesis(inferential_bridge=""), "")
        self.assertEqual(f["nonempty_source_paths"], [])
        f = self.field(self.hypothesis(), "금속  종류가 관계를 바꾼다.")
        self.assertEqual(f["raw_source_match_state"], "NO_EXACT_MATCH")

    def test_diagnostics_do_not_mutate_inputs(self):
        h = self.hypothesis()
        raw = {"required_bridge": h.inferential_bridge}
        accepted = {"required_bridge": ""}
        trace_specification_sources(h, raw, accepted)
        self.assertEqual(raw, {"required_bridge": h.inferential_bridge})
        self.assertEqual(accepted, {"required_bridge": ""})


if __name__ == "__main__":
    unittest.main()
