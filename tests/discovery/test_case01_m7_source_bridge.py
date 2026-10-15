"""Offline regression tests. Invoke from repo root with CASE01_TRACKB_KIT env set."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'discovery' / 'run_case01_m7_source_bridge.py'
spec = importlib.util.spec_from_file_location('case01_m7_bridge', SCRIPT)
bridge = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(bridge)


class TestCase01M7Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kit = os.getenv('CASE01_TRACKB_KIT')
        if not kit:
            raise unittest.SkipTest('CASE01_TRACKB_KIT must point to frozen original Case01 Track B kit')
        cls.kit = Path(kit)

    def test_source_provenance_and_card_identity(self):
        audit = bridge.verify_frozen(self.kit)
        self.assertEqual(audit['native_card_count'], 8)
        self.assertEqual(audit['page_count'], 8)
        self.assertEqual(audit['paper_doi'], bridge.YE_DOI)
        for arm in bridge.ARMS:
            task = bridge.build_task(self.kit, arm)
            self.assertEqual(task['parent_idea_id'], bridge.ANCHOR)
            self.assertEqual(task['role'], 'OPEN_ALTERNATE_BRANCH')
            self.assertEqual(len(task['original_kernel']['other_native_portfolio_cards_search_context_only']), 7)

    def test_experimental_delta_source_depth_only(self):
        abr = bridge.build_evidence(self.kit, 'ABSTRACT_ONLY')
        ful = bridge.build_evidence(self.kit, 'YE_FULLTEXT_FIGURES')
        self.assertNotIn('ye_main_paper_page_text', abr)
        self.assertNotIn('ye_original_pdf_figure_pages_attached_as_images', abr)
        self.assertEqual(abr['retrieved_prior_art_abstracts_same_in_both_arms'], ful['retrieved_prior_art_abstracts_same_in_both_arms'])
        self.assertEqual(abr['ye_paper_abstract'], ful['ye_paper_abstract'])
        self.assertEqual(len(ful['ye_main_paper_page_text']), 8)
        self.assertEqual(ful['ye_original_pdf_figure_pages_attached_as_images'], [3,4,5,6])

    def test_m7_native_import_and_prompts(self):
        from pipeline_core.discovery.scientific_development_m7 import create_model_prompt
        for arm in bridge.ARMS:
            messages = bridge.build_messages(self.kit, arm)
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0]['role'], 'system')
            self.assertIn('return_exact_json_keys_and_types', messages[1]['content'])
            self.assertIn('SOURCE EVIDENCE', messages[1]['content'])
            self.assertIn(bridge.ANCHOR, messages[1]['content'])
        self.assertNotIn('ye_main_paper_page_text', bridge.build_messages(self.kit, 'ABSTRACT_ONLY')[1]['content'])
        a = bridge.build_messages(self.kit, 'ABSTRACT_ONLY')
        b = bridge.build_messages(self.kit, 'YE_FULLTEXT_FIGURES')
        self.assertEqual(a[0]['content'], b[0]['content'])
        self.assertEqual(a[1]['content'].split('SOURCE EVIDENCE')[0], b[1]['content'].split('SOURCE EVIDENCE')[0])

    def test_prepare_immutable_and_paid_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'outputs'
            prepared = bridge.prepare(self.kit, out)
            self.assertEqual(prepared['status'], 'READY_NO_PROVIDER_CALLS')
            self.assertEqual(bridge.check_frozen(out)[1]['new_paid_calls'], 0)
            with self.assertRaises(FileExistsError): bridge.prepare(self.kit, out)
            # No paid call should be possible without explicit invocation flags.
            rc = bridge.main(['run-paid','--out',str(out),'--arm','ABSTRACT_ONLY'])
            self.assertEqual(rc,2)
            self.assertFalse((out/'attempts'/'ABSTRACT_ONLY.json').exists())
            # Changing a frozen prompt blocks all later calls.
            prompt_path = out/'prompts'/'ABSTRACT_ONLY.json'
            v = json.loads(prompt_path.read_text())
            v['messages'][1]['content'] += 'DRIFT'
            prompt_path.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError, 'PROMPT_CHANGED'): bridge.check_frozen(out)


if __name__ == '__main__': unittest.main()
