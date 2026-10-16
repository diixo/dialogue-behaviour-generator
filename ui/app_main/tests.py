"""UI integration tests with temporary artifacts, without torch or model downloads."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase, override_settings
from django.urls import reverse

from generator import generate_episode
from . import artifacts


class DashboardTests(SimpleTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.runs = self.root / 'data', self.root / 'runs'
        self.data.mkdir()
        self.runs.mkdir()
        self.settings_override = override_settings(DIALOGUE_DATA_DIR=self.data, DIALOGUE_RUN_DIRS=[self.runs])
        self.settings_override.enable()
        self.addCleanup(self.settings_override.disable)
        self.addCleanup(artifacts._index.cache_clear)
        self.addCleanup(artifacts._json.cache_clear)
        self.episodes = {}
        for split, count in (('train', 28), ('validation', 2), ('test', 2)):
            self.episodes[split] = [generate_episode(random.Random(i), f'{split}_{i:06d}', split)
                                    for i in range(count)]
            self.write_rows(self.data / f'{split}.episodes.jsonl', self.episodes[split])

    @staticmethod
    def write_rows(path, rows):
        path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')

    @staticmethod
    def write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

    def make_experiment(self):
        path = self.runs / 'ubar'
        config = {'task': 'ubar', 'base_model': 'gpt2', 'epochs': 2, 'style': 'plain',
                  'loss_scope': 'targets', 'batch_size': 4, 'learning_rate': 8e-5,
                  'gradient_accumulation_steps': 1,
                  'train_hash': artifacts.dataset_index('train').digest,
                  'validation_hash': artifacts.dataset_index('validation').digest,
                  'train': {'examples': 28, 'target_tokens': 900},
                  'validation': {'examples': 2, 'target_tokens': 60}}
        self.write_json(path / 'experiment.json', config)
        self.write_json(path / 'metrics.json', {'eval_loss': .7, 'best_checkpoint': str(path / 'checkpoint-8')})
        self.write_json(path / 'checkpoint-8' / 'trainer_state.json', {
            'epoch': 2, 'global_step': 8,
            'log_history': [{'epoch': 1, 'step': 4, 'loss': 1.2},
                            {'epoch': 1, 'step': 4, 'eval_loss': .9},
                            {'epoch': 2, 'step': 8, 'eval_loss': .7}]})
        return artifacts.artifact_id(path / 'experiment.json')

    def make_report(self, name='eval-ubar', **overrides):
        path = self.runs / name
        metrics = {'state.exact': {'value': .5, 'numerator': 1, 'denominator': 2},
                   'behaviour.repeat_known': {'value': 0, 'numerator': 0, 'denominator': 2}}
        config = {'schema_version': 1, 'mode': 'full', 'seed': 42, 'max_turns': 17,
                  'splits': ['test'], 'corpus_hash': artifacts.dataset_index('test').digest,
                  'episodes': 2, 'evaluated_turns': 2, 'context': 'reference_history_per_transition',
                  'initial_state': 'provided', 'metrics': metrics,
                  'models': {'dst': {'path': str(self.runs / 'ubar'), 'experiment': {'task': 'ubar', 'style': 'plain'},
                                     'generation': {'decoding': 'greedy', 'max_new_tokens': 384}}},
                  'errors': {'state.mismatch': 1},
                  'groups': {'reaction=DIRECT_ANSWER': {'examples': 2, 'metrics': metrics}},
                  'data_review_candidates': [{'error': 'state.mismatch', 'reaction': 'DIRECT_ANSWER',
                                              'target': 'mood', 'count': 1}], **overrides}
        self.write_json(path / 'report.json', config)
        records = []
        for i in range(2):
            records.append({'episode_id': f'test_{i:06d}', 'turn': 1,
                            'tags': {'reaction': 'DIRECT_ANSWER', 'action': 'ASK', 'target': 'mood', 'depth': '0-2'},
                            'errors': ['state.mismatch'] if i == 0 else [],
                            'metrics': {'state.exact': i, 'behaviour.repeat_known': 0},
                            'reference': {'state': {'known': {'mood': 'happy'}}},
                            'stages': {'dst': {'text': '<script>alert(1)</script>', 'tokens': 4, 'stopped': True,
                                              'parsed': {'known': {'mood': 'sad'}},
                                              'input': {'history': [], 'state': {}, 'action': None}}}})
        self.write_rows(path / 'predictions.jsonl', records)
        return artifacts.artifact_id(path / 'report.json')

    def test_overview_and_empty_artifact_lists(self):
        response = self.client.get(reverse('app_main:main'))
        self.assertEqual(response.context['episode_count'], 32)
        self.assertEqual(response.context['report_count'], 0)
        for name in ('experiments', 'reports'):
            self.assertEqual(self.client.get(reverse(f'app_main:{name}')).status_code, 200)
        self.assertRedirects(self.client.get('/report'), reverse('app_main:reports'))

    def test_dataset_pagination_filters_and_fingerprint(self):
        response = self.client.get(reverse('app_main:datasets'))
        self.assertEqual(len(response.context['items']), 25)
        response = self.client.get(reverse('app_main:datasets'), {'page': 2})
        self.assertEqual(len(response.context['items']), 3)
        response = self.client.get(reverse('app_main:datasets'), {'q': 'train_000027', 'action': 'CLOSE'})
        self.assertEqual(response.context['items'].paginator.count, 1)
        expected = hashlib.sha256(json.dumps(self.episodes['train'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(artifacts.dataset_index('train').digest, expected)

    def test_linked_examples_use_same_turn_and_keep_simulator_facts_out_of_input(self):
        response = self.client.get(reverse('app_main:episode', args=['test', 'test_000000']))
        self.assertEqual(response.status_code, 200)
        examples = response.context['examples']
        self.assertEqual([(e['view'], e['turn']) for e in examples],
                         [('extraction', 0), ('policy', 0), ('response', 0), ('history', 0)])
        state = json.loads(examples[0]['completion'])['state']
        self.assertEqual(state, self.episodes['test'][0]['turns'][0]['state_before'])
        self.assertEqual(json.loads(response.context['after_json']), state)
        self.assertNotIn('hidden_user', json.dumps([e['messages'] for e in examples]))
        self.assertNotIn('simulator', json.dumps([e['messages'] for e in examples]))

    def test_terminal_turn_extracts_last_user_message_before_closing(self):
        last = len(self.episodes['test'][0]['turns']) - 1
        response = self.client.get(reverse('app_main:episode', args=['test', 'test_000000']), {'turn': last})
        self.assertEqual(response.context['next_turn'], last)
        self.assertEqual([e['view'] for e in response.context['examples']], ['extraction', 'policy', 'response', 'history'])
        self.assertTrue(all(e['turn'] == last for e in response.context['examples']))

    def test_invalid_routes_and_path_inputs_do_not_expose_files(self):
        for url in ('/data/?split=../runs', '/data/test/missing/', '/data/test/test_000000/?turn=-1',
                    '/data/test/test_000000/?turn=abc', '/experiments/unknown/', '/benchmarks/unknown/'):
            self.assertEqual(self.client.get(url).status_code, 404, url)

    def test_json_fallback_and_cache_refresh(self):
        path = self.data / 'validation.episodes.jsonl'
        original = artifacts.dataset_index('validation')
        changed = deepcopy(self.episodes['validation'])
        changed[0]['turns'][0]['assistant'] = 'Changed opening sentence.'
        self.write_rows(path, changed)
        refreshed = artifacts.dataset_index('validation')
        self.assertNotEqual(original.digest, refreshed.digest)
        self.assertEqual(refreshed.record(0)['turns'][0]['assistant'], 'Changed opening sentence.')
        path.unlink()
        self.write_json(path.with_suffix('.json'), changed)
        self.assertEqual(artifacts.dataset_index('validation').record(0)['episode_id'], 'validation_000000')

    def test_broken_dataset_is_reported_and_duplicate_ids_are_rejected(self):
        path = self.data / 'train.episodes.jsonl'
        self.write_rows(path, [self.episodes['train'][0]] * 2)
        self.assertEqual(self.client.get('/data/').status_code, 422)
        self.assertContains(self.client.get('/'), 'Duplicate episode_id')
        path.write_text('{broken', encoding='utf-8')
        self.assertEqual(self.client.get('/data/').status_code, 422)

    def test_experiment_checkpoints_chart_and_related_report(self):
        identifier = self.make_experiment()
        report_id = self.make_report()
        response = self.client.get(reverse('app_main:experiment', args=[identifier]))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['checkpoints'][0]['best'])
        self.assertEqual(len(response.context['chart']['series']), 2)
        self.assertTrue(all(m['matches'] for m in response.context['matches']))
        self.assertEqual(response.context['reports'][0]['id'], report_id)

    def test_benchmark_filters_and_reference_prediction_detail_escape_text(self):
        identifier = self.make_report()
        url = reverse('app_main:report_detail', args=[identifier])
        response = self.client.get(url, {'error': 'state.mismatch', 'tag': 'reaction=DIRECT_ANSWER'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['items'].paginator.count, 1)
        self.assertEqual(response.context['selected_group']['examples'], 2)
        response = self.client.get(url, {'metric': 'state.exact', 'value': '0'})
        self.assertEqual(response.context['items'].paginator.count, 1)
        response = self.client.get(url, {'error': 'state.mismatch', 'target': 'activity'})
        self.assertEqual(response.context['items'].paginator.count, 0)
        detail = self.client.get(reverse('app_main:prediction', args=[identifier, 0]))
        self.assertEqual(detail.status_code, 200)
        self.assertContains(detail, '&lt;script&gt;')
        self.assertNotContains(detail, '<script>alert(1)</script>')
        self.assertTrue(detail.context['source_url'])
        self.assertEqual(detail.context['stages'][0]['changes'][0]['field'], 'known.mood')

    def test_rollout_and_unverified_corpus_never_link_to_wrong_source_turn(self):
        for options in ({'mode': 'rollout'}, {'corpus_hash': 'different'}):
            identifier = self.make_report(**options)
            response = self.client.get(reverse('app_main:prediction', args=[identifier, 0]))
            self.assertIsNone(response.context['source_url'])

    def test_session_report_shows_generated_context_and_retains_recorded_source_link(self):
        identifier = self.make_report(mode='session', context='predicted_history_with_recorded_users',
            initial_state='empty_before_first_user', session_summary={
                'generated_turns': 1, 'termination_counts': {'premature_close': 1}})
        response = self.client.get(reverse('app_main:report_detail', args=[identifier]))
        self.assertContains(response, 'Session: сообщения пользователя взяты из датасета')
        self.assertContains(response, 'Завершённых ходов: 1 / 2')
        self.assertNotContains(response, 'Replay: каждый переход')
        detail = self.client.get(reverse('app_main:prediction', args=[identifier, 0]))
        self.assertTrue(detail.context['source_url'])

    def test_report_without_predictions_and_malformed_manifest(self):
        identifier = self.make_report()
        (self.runs / 'eval-ubar' / 'predictions.jsonl').unlink()
        response = self.client.get(reverse('app_main:report_detail', args=[identifier]))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['has_predictions'])
        self.assertTrue(response.context['metric_rows'])
        self.write_json(self.runs / 'broken' / 'report.json', [])
        self.assertEqual(len(artifacts.discover()[2]), 1)
        self.assertEqual(self.client.get(reverse('app_main:reports')).status_code, 200)

    def test_comparison_checks_conditions_and_keeps_support_counts(self):
        first = self.make_report('first')
        second = self.make_report('second')
        url = reverse('app_main:compare')
        response = self.client.get(url, {'run': [first, second]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['comparison_issues'], [])
        self.assertEqual(response.context['rows'][0]['cells'][0]['denominator'], 2)
        self.make_report('second', mode='rollout', corpus_hash='other', seed=43)
        response = self.client.get(url, {'run': [first, second]})
        self.assertTrue(any('corpus_hash' in issue for issue in response.context['comparison_issues']))
        self.assertTrue(any('mode' in issue for issue in response.context['comparison_issues']))
        self.assertTrue(any('seed' in issue for issue in response.context['comparison_issues']))
        self.assertTrue(self.client.get(url, {'run': [first]}).context['selection_error'])

    def test_comparison_does_not_confirm_missing_generation_settings(self):
        first, second = self.make_report('first'), self.make_report('second', models={})
        response = self.client.get(reverse('app_main:compare'), {'run': [first, second]})
        self.assertTrue(response.context['comparison_issues'])

    def test_comparison_requires_recorded_context_style(self):
        first = self.make_report('first')
        second = self.make_report('second', models={
            'dst': {'path': 'model', 'experiment': {}, 'generation': {'decoding': 'greedy'}}})
        response = self.client.get(reverse('app_main:compare'), {'run': [first, second]})
        self.assertTrue(any('формат контекста не записан' in issue
                            for issue in response.context['comparison_issues']))
