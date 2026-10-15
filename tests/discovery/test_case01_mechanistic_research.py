from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline_core.discovery.scientific_research_control import runner as r

SCIENCE = Path(os.environ.get('CASE01_TEST_SCIENCE', '/mnt/data/CASE01_OPEN_PROBLEM_DISCOVERY_PILOT/frozen_sources'))
CONTEXT = Path(os.environ.get('CASE01_TEST_CONTEXT', '/mnt/data/case01_reentry_staging_native/hypothesis.context.json'))


def action(kind, **kwargs):
    value = {'action': kind,
             'scientific_reason': 'This physical limitation is central to understanding the reported blinking.',
             'research_question_now': 'How can the microscopic state generate the measured blinking and controls?'}
    value.update(kwargs)
    return value


def make_idea(tag='1', family='ELECTRONIC STATES'):
    return {
        'local_id': tag,
        'title': 'Microscopic switching explanation for multiple Raman modes',
        'causal_family': family,
        'proposed_state_variables': ['q: hypothetical interfacial state'],
        'causal_mechanism': 'An interfacial state makes reversible transitions that affect the mode-dependent Raman susceptibility.',
        'why_standard_explanations_are_incomplete': 'Independent local changes do not account for a coupled cross-mode response without further assumptions.',
        'strong_conventional_null': 'Independent local switching, laser drift and variable adsorption with measured field changes.',
        'conditional_prediction': 'Specific conditional correlations between vibrational bands and temperature-independent state changes would occur.',
        'how_the_prediction_follows': 'Both Raman channels couple to the same interfacial variable, which changes their optical response.',
        'null_mimic_risk': 'A common thermal input could also couple nominally independent local reporters.',
        'independent_test_or_observation': 'Simultaneously measure distinct Raman bands and temperature while varying adsorption conditions.',
        'falsification_outcome': 'The coupling mechanism would be contradicted if independently monitored channels showed no dependence.',
        'evidence_ids': ['YE:P1'],
        'novelty_uncertainty': 'This is speculative; neither a scientific discovery nor prior-art novelty is certified.'}


def observation(tag='O1', page='YE:P1'):
    return {
        'observation_id': tag, 'kind': 'REPORTED_OBSERVATION',
        'statement': 'The source page describes intermittent Raman signals in a measured sample.',
        'experimental_condition': 'Reported sample and excitation regime',
        'evidence_ids': [page],
        'why_it_constrains_mechanisms': 'A proposed mechanism must generate changing Raman intensity rather than only a static shift.'}


def mechanism(iid, handled=None, missing=None):
    return {
        'target_idea_id': iid,
        'causal_sequence': [
            'Excitation changes a microscopic interfacial state.',
            'The interfacial state changes the molecular Raman tensor.',
            'The measured Raman time series responds with intermittent bursts.'
        ],
        'observable_generation': 'A stochastic state transition leads to mode-dependent intermittency and requires measured lifetimes.',
        'explained_observation_ids': handled or [],
        'unaddressed_observation_ids': missing or [],
        'physical_assumptions_not_yet_supported': ['State transition timescale has no independent measurement'],
        'what_must_be_derived_or_checked_next': 'Show how a stochastic transition rate produces the observed statistics.'}


def ensure_context_when_repo_not_present():
    # In the user's WSL checkout use the genuine native HypothesisContext.
    # The model's patch-only test environment lacks that module.
    try:
        from pipeline_core.discovery.hypothesis_contracts import HypothesisContext
    except ImportError:
        module = types.ModuleType('pipeline_core.discovery.hypothesis_contracts')
        class StubContext:
            @classmethod
            def model_validate_json(cls, raw):
                x = json.loads(raw)
                return types.SimpleNamespace(context_id=x['context_id'],
                    context_sha256=x['context_sha256'], source_packet_id=x['source_packet_id'])
        module.HypothesisContext = StubContext
        sys.modules['pipeline_core.discovery.hypothesis_contracts'] = module



class MechanisticResearch(unittest.TestCase):
    def setUp(self):
        ensure_context_when_repo_not_present()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.old = self.path/'r01'
        self.new = self.path/'r02'
        r.prepare(out=self.old, pages_file=SCIENCE/'Ye_main_text_by_page.json',
                  works_file=SCIENCE/'hypothesis_axis_open_world.selected_works.json',
                  native_context_file=CONTEXT, max_steps=12)
        r.record_step(self.old, action('EXPLORE', ideas=[make_idea()]))
        self.iid = next(iter(r.verify(self.old)['state']['ideas']))
        r.record_step(self.old, action('CHALLENGE', target_idea_id=self.iid,
                        challenge_result='NULL_MIMICS',
                        challenge_null_mimic='Common laser and thermal artifacts make the same correlation.',
                        challenged_prediction='Both Raman modes should change in a synchronized manner.'))
        r.record_step(self.old, action('END', target_idea_id=self.iid,
                        final_disposition='PROVISIONAL_RESEARCH_CANDIDATE',
                        final_justification='This is a provisional research direction despite the unresolved null mimic.'))
        self.old_checksum = r.digest(r.verify(self.old)['state'])

    def create_new(self):
        result = r.prepare(out=self.new, pages_file=SCIENCE/'Ye_main_text_by_page.json',
                           works_file=SCIENCE/'hypothesis_axis_open_world.selected_works.json',
                           native_context_file=CONTEXT, research_profile='mechanistic',
                           prior_episode=self.old, max_steps=20)
        self.assertEqual(result['schema_version'], r.V2_MANIFEST)
        return r.verify(self.new)

    def test_prior_episode_frozen_and_prior_ideas_unforced(self):
        initial = self.create_new()['state']
        self.assertEqual(initial['step'], 0)
        self.assertFalse(initial['ended'])
        self.assertEqual(initial['prior_research_sha256'], self.old_checksum)
        self.assertEqual(initial['ideas'][self.iid]['status'], 'CONTESTED')
        self.assertEqual(r.verify(self.old)['state']['step'], 3)
        self.assertEqual(r.digest(r.verify(self.old)['state']), self.old_checksum)
        payload = r.prompt_payload(initial, r.verify(self.new)['source_index'], r.verify(self.new)['manifest'])
        self.assertIn('research_decision_context', payload)
        self.assertIn('prior_research_trajectory_not_scientific_evidence', payload)
        self.assertEqual(payload['action_schema'], r.ActionV2.model_json_schema())
        self.assertNotIn('read_evidence', payload['prior_research_trajectory_not_scientific_evidence'])

    def test_old_episode_remains_legacy_payload(self):
        checked = r.verify(self.old)
        payload = r.prompt_payload(checked['state'], checked['source_index'], checked['manifest'])
        self.assertEqual(payload['action_schema'], r.Action.model_json_schema())
        self.assertNotIn('research_decision_context', payload)
        self.assertEqual(checked['manifest']['schema_version'], 'fakg-case01-native-autonomous-research-v1')

    def test_observations_source_grounding_and_gap_account(self):
        self.create_new()
        r.record_step(self.new, action('DEEPEN', target_idea_id=self.iid,
                         new_prediction_or_derivation='State transitions can drive intermittent measured signal.',
                         observation_updates=[observation()],
                         mechanistic_account=mechanism(self.iid, handled=['O1'])))
        verified = r.verify(self.new)
        self.assertIn('O1', verified['state']['observation_register'])
        self.assertFalse(verified['state']['observation_register']['O1']['science_validated_externally'])
        self.assertEqual(len(verified['state']['mechanistic_accounts']), 1)
        self.assertEqual(verified['state']['mechanistic_accounts'][0]['explained_observation_ids'], ['O1'])
        self.assertEqual(verified['state']['step'], 1)

    def test_newly_requested_source_cannot_be_observed_without_reading_it(self):
        # Previous paper ends at P1 in the synthetic R01, so P2 is genuinely unread.
        self.create_new()
        self.assertNotIn('YE:P2', r.verify(self.new)['state']['read_ids'])
        with self.assertRaisesRegex(ValueError, 'unread'):
            r.record_step(self.new, action('READ_EVIDENCE', evidence_to_read=['YE:P2'],
                             observation_updates=[observation(page='YE:P2')]))
        self.assertEqual(r.verify(self.new)['state']['step'], 0)
        r.record_step(self.new, action('READ_EVIDENCE', evidence_to_read=['YE:P2']))
        r.record_step(self.new, action('DEEPEN', target_idea_id=self.iid,
                         new_prediction_or_derivation='Additional sample conditions constrain switching.',
                         observation_updates=[observation(page='YE:P2')]))
        self.assertEqual(r.verify(self.new)['state']['step'], 2)

    def test_duplicate_observation_not_rewritten(self):
        self.create_new()
        r.record_step(self.new, action('DEEPEN', target_idea_id=self.iid,
                         new_prediction_or_derivation='A transition timescale may be measured.',
                         observation_updates=[observation()]))
        with self.assertRaisesRegex(ValueError, 'duplicate observation id'):
            r.record_step(self.new, action('DEEPEN', target_idea_id=self.iid,
                              new_prediction_or_derivation='Another physical timescale is plausible.',
                              observation_updates=[observation()]))
        self.assertEqual(r.verify(self.new)['state']['step'], 1)

    def test_invalid_account_references_not_accepted(self):
        self.create_new()
        with self.assertRaisesRegex(ValueError, 'unregistered observation'):
            r.record_step(self.new, action('DEEPEN', target_idea_id=self.iid,
                             new_prediction_or_derivation='Transition produces intermittent signal.',
                             mechanistic_account=mechanism(self.iid, handled=['O_DOES_NOT_EXIST'])))

    def test_end_preserves_null_mimic_warning_not_level_ii(self):
        self.create_new()
        r.record_step(self.new, action('DEEPEN', target_idea_id=self.iid,
                        new_prediction_or_derivation='State lifetime controls observed burst duration.',
                        observation_updates=[observation()],
                        mechanistic_account=mechanism(self.iid, missing=['O1'])))
        r.record_step(self.new, action('END', target_idea_id=self.iid,
                        final_disposition='PROVISIONAL_RESEARCH_CANDIDATE',
                        final_justification='No independently established discovery; this is provisional.',
                        ideas=[make_idea(tag='FINAL')]))
        state = r.verify(self.new)['state']
        self.assertTrue(state['ended'])
        self.assertEqual(len(state['ideas']), 1)
        self.assertFalse(state['final_record']['level_ii_certified'])
        self.assertTrue(state['final_record']['unresolved_null_mimic_warning'])
        self.assertEqual(state['final_record']['unexplained_observation_ids'], ['O1'])
        self.assertEqual(state['final_record']['observations_extracted'], 1)

    def test_mutated_prior_history_fails_provenance(self):
        self.create_new()
        self.assertEqual(r.verify(self.new)['completed_steps'], 0)
        p = self.new/'sources.json'
        obj = r.load(p)
        obj['prior_research']['prior_challenges'][0]['status'] = 'CONDITIONAL_SURVIVAL'
        p.write_text(json.dumps(obj, ensure_ascii=False))
        with self.assertRaisesRegex(ValueError, 'tampered'):
            r.verify(self.new)

    def test_failed_model_call_recovers_without_backend(self):
        self.create_new()
        checked = r.verify(self.new)
        payload = r.prompt_payload(checked['state'], checked['source_index'], checked['manifest'])
        req = {'model': checked['manifest']['model'], 'temperature': checked['manifest']['temperature'],
               'max_tokens': checked['manifest']['max_tokens'],
               'messages': [{'role': 'system', 'content': r.SYSTEM_V2},
                            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
               'research_step': 1, 'input_state_sha256': r.digest(checked['state'])}
        call_dir = self.new/'calls/step_01'
        r.write_once(call_dir/'request.json', req)
        r.write_once(call_dir/'attempt.json', {'request_sha256': r.digest(req)})
        response = action('DEEPEN', target_idea_id=self.iid,
                          new_prediction_or_derivation='The mechanism would cause stochastic Raman intermittency.',
                          observation_updates=[observation()],
                          mechanistic_account=mechanism(self.iid, handled=['O1']))
        r.write_once(call_dir/'raw_response.json', {'raw_text':json.dumps(response), 'usage':{},'provider_response_id':'test-free'})
        outcome = r.recover(self.new)
        self.assertEqual(outcome['status'], 'RECOVERED_NO_CALL')
        self.assertEqual(r.verify(self.new)['state']['step'], 1)

    def test_next_action_bias_is_visible_not_forced(self):
        self.create_new()
        checked = r.verify(self.new)
        decision = r.prompt_payload(checked['state'], checked['source_index'], checked['manifest'])['research_decision_context']
        self.assertTrue(decision['no_observations_extracted_yet'])
        self.assertIn(self.iid, decision['candidates_without_generating_mechanistic_account'])
        r.record_step(self.new, action('DEEPEN', target_idea_id=self.iid,
                        new_prediction_or_derivation='Test source observations.', observation_updates=[observation()]))
        checked = r.verify(self.new)
        decision = r.prompt_payload(checked['state'], checked['source_index'], checked['manifest'])['research_decision_context']
        self.assertIn('O1', decision['unaccounted_observation_ids'])
        self.assertTrue(decision['decision_not_forced'])

    def test_prior_pivot_does_not_fail_initial_explore(self):
        # Cover inherited pivots combined with the new episode's empty trajectory.
        self.create_new()
        src = r.load(self.new/'sources.json')
        src['prior_research']['prior_pivots'] = [{'idea_id': self.iid, 'step':3, 'abandoned_assumption':'hypothesis invalid', 'new_direction':'another mechanism'}]
        r.write_once(self.path/'other.json',src)
        # Separate initial_state test: no writes/mutations to the verified episode.
        state = r.initial_state(src)
        self.assertTrue(state['pivots'])
        self.assertEqual(state['trajectory'], [])
        out = r.apply_action(state, r.ActionV2.model_validate(action('EXPLORE',ideas=[make_idea(tag='2',family='NEW PHOTOPHYSICS')])),src)
        self.assertEqual(out['step'], 1)


if __name__ == '__main__':
    unittest.main()
