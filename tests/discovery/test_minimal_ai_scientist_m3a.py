"""M3-A tests: safe orchestration, no mocked scientific quality claims."""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/discovery/run_minimal_ai_scientist_m3a.py'
spec = importlib.util.spec_from_file_location('m3a', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class M3ATest(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.case = self.root / 'existing-case'
        self.case.mkdir()
        for f in ['hypothesis.context.json', 'frontier_idea_population.shadow.json', 'idea_evolution.shadow.json']:
            (self.case / f).write_text('{}')
        self.a = m.parser().parse_args(['--case-dir', str(self.case), '--output-dir', str(self.root/'result')])

    def test_default_is_offline_existing_and_seed_only(self):
        self.assertEqual(self.a.source_mode, 'existing')
        self.assertEqual(self.a.through, 'seed')
        self.assertFalse(self.a.allow_paid)

    def test_no_paid_gate_for_sis(self):
        self.a.through = 'sis'
        self.a.model = 'dummy'
        with self.assertRaises(PermissionError): m.preflight(self.a)

    def test_no_paid_gate_for_new_stage7(self):
        self.a.source_mode = 'generate'
        self.a.case_dir = None
        self.a.source = 'molecular orientation'
        self.a.target = 'Raman intensity'
        self.a.question = 'Why?'
        self.a.data_root = self.root
        self.a.model = 'dummy'
        with self.assertRaises(PermissionError): m.preflight(self.a)

    def test_does_not_run_paid_in_plan_mode(self):
        self.a.through = 'verified'
        self.a.model = 'dummy'
        self.a.plan_only = True
        with patch.object(m, 'run_stage') as stage:
            x = m.run(self.a)
        self.assertEqual(x['status'], 'PLAN_ONLY')
        self.assertFalse((self.root/'result').exists())
        stage.assert_not_called()

    def test_legacy_portfolio_not_requested_by_seed(self):
        cmd = m.seed_cmd(self.a, self.case, self.root/'result')
        self.assertIn('--evaluation', cmd)
        self.assertIn('--candidate-pool', cmd)
        self.assertNotIn('--allow-evaluation-llm', cmd)
        self.assertNotIn('materialized.shadow.portfolio.json', ' '.join(cmd))

    def test_new_source_flag_parity_no_stage7_selection_materialization(self):
        self.a.data_root = self.root
        self.a.source, self.a.target, self.a.question = 'A', 'B', 'Why?'
        self.a.model = 'dummy'
        args = m.stage7_cmd(self.a, self.root/'result'/'case')
        for flag in ('--frontier-idea-population-shadow', '--idea-evolution-shadow',
                     '--higher-order-shadow', '--direct-higher-order-shadow',
                     '--stop-after-initial-semantic'):
            self.assertIn(flag, args)
        self.assertNotIn('--scientific-portfolio-selection-shadow', args)
        self.assertNotIn('--scientific-portfolio-max-retained', args)

    def test_m2d_opts_not_implicitly_activated(self):
        self.a.model = 'dummy'
        cmd = m.sis_cmd(self.a, self.case, self.root/'result')
        self.assertNotIn('--exact-prior-art-cache', cmd)
        self.assertNotIn('--incremental-program-family', cmd)
        self.assertIn('--p0-execution', cmd)

    def test_disallow_out_inside_existing_case(self):
        self.a.output_dir = self.case/'output'
        with self.assertRaises(ValueError): m.preflight(self.a)

    def test_disallow_preexisting_output(self):
        (self.root/'result').mkdir()
        with self.assertRaises(FileExistsError): m.preflight(self.a)

    def test_missing_parity_pin_fails_before_writes(self):
        self.a.expected_p0_execution = self.root/'not_existing.json'
        with self.assertRaises(FileNotFoundError): m.run(self.a)
        self.assertFalse((self.root/'result').exists())

    def test_seed_summary_cross_checks(self):
        out = self.root/'result'
        (out/'direct_seed').mkdir(parents=True)
        m.write_json(out/'direct_seed'/'direct_seed.summary.json', dict(status='DIRECT_P0_READY', p0_execution_report_id='expected', selected_research_idea_count=8, early_materialization_llm_calls=0, early_hypothesis_materialization_executed=False))
        m.write_json(out/'direct_seed'/'p0.execution.json', dict(report_id='different', g4_population_count=8, generation_index=2, grounding_required_before_scientific_claim=True))
        with self.assertRaisesRegex(ValueError, 'report ID'): m.validate_seed(out)

    def test_sis_requires_identical_p0(self):
        out = self.root/'result'
        (out/'sis_v3_4').mkdir(parents=True)
        m.write_json(out/'sis_v3_4'/'arm.summary.json', dict(source_p0_report_id='different'))
        with self.assertRaisesRegex(ValueError, 'different P0'):
            m.validate_sis(out, {'p0_execution_report_id': 'expected'})

    def test_end_to_end_offline_seed_mode_records_state(self):
        def fake(label, cmd, logdir):
            self.assertEqual(label,'direct_seed')
            o=self.root/'result'/'direct_seed'
            o.mkdir(parents=True)
            m.write_json(o/'direct_seed.summary.json', dict(status='DIRECT_P0_READY',p0_execution_report_id='seed:123',selected_research_idea_count=8,early_materialization_llm_calls=0,early_hypothesis_materialization_executed=False,portfolio_evaluation_llm_calls=0,old_p0_exact_parity_checked=True))
            m.write_json(o/'p0.execution.json', dict(report_id='seed:123',generation_index=2,g4_population_count=8,grounding_required_before_scientific_claim=True))
        with patch.object(m,'run_stage',side_effect=fake) as stage:
            result=m.run(self.a)
        self.assertEqual(result['status'],'SEED_READY')
        self.assertTrue(result['p0_parity_checked'])
        self.assertEqual(result['early_materialization_llm_calls'],0)
        self.assertEqual(stage.call_count,1)
        self.assertTrue((self.root/'result'/'M3A_REPORT.md').is_file())

    def test_verified_mode_calls_existing_final_verifier_and_archive(self):
        self.a.model='fake'
        self.a.through='verified'
        self.a.allow_paid=True
        out=self.root/'result'
        stages=[]
        def fake(label, cmd, logs):
            stages.append(label)
            if label=='direct_seed':
                d=out/'direct_seed';d.mkdir(parents=True)
                m.write_json(d/'direct_seed.summary.json',dict(status='DIRECT_P0_READY',p0_execution_report_id='seed:123',selected_research_idea_count=8,early_materialization_llm_calls=0,early_hypothesis_materialization_executed=False,portfolio_evaluation_llm_calls=0,old_p0_exact_parity_checked=False))
                m.write_json(d/'p0.execution.json',dict(report_id='seed:123',generation_index=2,g4_population_count=8,grounding_required_before_scientific_claim=True))
            elif label=='sis_v3_4':
                d=out/'sis_v3_4';d.mkdir()
                m.write_json(d/'final.materialized.portfolio.json',{'hypotheses':[{'hypothesis_id':'h1'}]})
                m.write_json(d/'arm.summary.json',dict(source_p0_report_id='seed:123',child_generation_steps=2,grounding_required_before_scientific_claim=True,canonical_graph_mutated=False,external_prior_art_promoted_to_positive_premise=False,final_portfolio=str(d/'final.materialized.portfolio.json'),final_hypothesis_count=1,continuation_generation_llm_calls=2,continuation_realization_llm_calls=3))
            elif label=='program_archive':
                d=out/'program_archive';d.mkdir()
                m.write_json(d/'M2F_SUMMARY.json',dict(status='M2F_ARCHIVE_AND_TERMINAL_VIEW_READY', archived_research_ideas=3,terminal_active_research_ideas=2,replaced_parent_ideas_preserved_in_archive=1,terminal_hypotheses=1))
            elif label=='common_verifier':
                d=out/'verification';d.mkdir()
                m.write_json(d/'verification.summary.json',dict(status='DIAGNOSTIC_ONLY'))
        with patch.object(m,'run_stage',side_effect=fake):
            result=m.run(self.a)
        self.assertEqual(stages,['direct_seed','sis_v3_4','program_archive','common_verifier'])
        self.assertEqual(result['status'],'COMMON_VERIFIER_COMPLETED_NONAUTHORITATIVE')
        self.assertFalse(result['novelty_or_scientific_quality_certified'])
        self.assertEqual(result['sis_generation_calls_partial'],2)

    def test_malformed_seed_failure_is_not_misreported_as_success(self):
        def fake(label, cmd, logdir):
            d=self.root/'result'/'direct_seed';d.mkdir(parents=True)
            m.write_json(d/'direct_seed.summary.json',dict(status='DIRECT_P0_READY',p0_execution_report_id='p',selected_research_idea_count=8,early_materialization_llm_calls=0,early_hypothesis_materialization_executed=False))
            m.write_json(d/'p0.execution.json',dict(report_id='wrong',generation_index=2,g4_population_count=8,grounding_required_before_scientific_claim=True))
        with patch.object(m,'run_stage',side_effect=fake):
            with self.assertRaises(ValueError): m.run(self.a)
        state=m.read_json(self.root/'result'/'m3a.run.json')
        self.assertEqual(state['status'],'FAILED_VALIDATION_AT_SEED')

    def test_failed_command_is_recorded(self):
        with patch.object(m,'run_stage',side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):m.run(self.a)
        state=m.read_json(self.root/'result'/'m3a.run.json')
        self.assertEqual(state['status'],'FAILED_AT_DIRECT_SEED')


if __name__=='__main__':
    unittest.main()
