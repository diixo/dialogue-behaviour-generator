"""Offline objective, causality, error propagation and training integration tests."""

from copy import deepcopy
import contextlib
import gc
import io
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from benchmark import response_variants, run_benchmark
from compile_sft import serialize_action
from data import (SEGMENTS, SPECIAL_TOKENS, SessionDataset, TaskDataset, encode, segment,
                  state_text, task_records, ubar_header, ubar_user)
from dialogue_format import get_format
from generator import Action, State, commit_action, compile_views, generate_episode, select_action
from inference import HFPredictor, Prediction, Request, parse_extraction, parse_state
from trainer import train_experiment


def tokenizer_for(episodes):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    texts = []
    for task in ('dst', 'extraction', 'policy', 'response', 'history'):
        for record in task_records(episodes, task):
            rendered = get_format('plain').render(record['messages'], record['completion'])
            texts.extend(rendered.values())
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=700, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['<eos>', '<unk>']))
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token='<eos>',
                                   unk_token='<unk>', pad_token='<eos>')


def history_key(history):
    return tuple((message['role'], message['content']) for message in history)


class ReplayOracle:
    """Fixture references stay inside this test; Request has no target fields."""
    def __init__(self, episode):
        views = compile_views(episode)
        self.states = {history_key(row['history_before'] + [{'role': 'user', 'content': row['utterance']}]):
                       row for row in views['extraction']}
        self.responses = {history_key(row['history']): row['target_response'] for row in views['history']}
        self.requests = []

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        key = history_key(request.history)
        if request.task in ('dst', 'extraction'):
            row = self.states[key]
            text = state_text(row['target_state']) if request.task == 'dst' else json.dumps({
                'state': row['target_state'], 'observation': row['target_observation']})
        elif request.task == 'policy':
            text = serialize_action(select_action(State.from_snapshot(request.state)).json())
        else:
            text = self.responses[key]
        return Prediction(text)


class ObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = generate_episode(random.Random(42), 'training', 'train')
        cls.tokenizer = tokenizer_for([cls.episode])
        cls.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

    def test_dst_labels_are_state_only_with_supervised_native_eos(self):
        dataset = TaskDataset([self.episode], self.tokenizer, 'dst', max_length=2048)
        rows = compile_views(self.episode)['extraction']
        self.assertEqual(len(dataset), len(rows))
        for item, reference in zip(dataset, rows):
            labels = item['labels'][item['labels'] != -100].tolist()
            self.assertEqual(labels[-1], self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(labels[:-1])
            self.assertEqual(parse_state(text), reference['target_state'])
            self.assertNotIn('hidden_user', self.tokenizer.decode(item['input_ids']))

    def test_session_windows_preserve_every_target_without_fabricated_eos(self):
        tokenizer = self.tokenizer
        episode = self.episode
        blocks = []
        for i, turn in enumerate(episode['turns']):
            user = encode(tokenizer, ubar_user(turn['user']))
            output = encode(tokenizer, segment('dst', state_text(turn['state_before'])) +
                            segment('policy', serialize_action(turn['action'])) +
                            segment('response', turn['assistant']))
            blocks.append(user + output)
        size = len(encode(tokenizer, ubar_header())) + max(
            len(a) + len(b) for a, b in zip(blocks, blocks[1:])) + 1
        full = SessionDataset([episode], tokenizer, max_length=10000)
        self.assertEqual(tokenizer.decode(full[0]['input_ids']),
                         ubar_header() + ''.join(tokenizer.decode(block) for block in blocks) + tokenizer.eos_token)
        windows = SessionDataset([episode], tokenizer, max_length=size)
        self.assertGreater(len(windows), 1)
        targets = lambda ds: [token for item in ds for token in item['labels'].tolist() if token != -100]
        self.assertEqual(targets(full), targets(windows))
        self.assertEqual(targets(windows).count(tokenizer.eos_token_id), 1)
        self.assertEqual(windows[-1]['labels'][-1].item(), tokenizer.eos_token_id)
        self.assertTrue(all(len(item['input_ids']) <= size for item in windows))
        self.assertGreater(sum(int((item['labels'] != -100).sum()) for item in
                               SessionDataset([episode], tokenizer, max_length=size, loss_scope='all')),
                           len(targets(windows)))

    def test_overlong_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'max_length'):
            TaskDataset([self.episode], self.tokenizer, 'dst', max_length=64)

    def test_parsers_reject_incomplete_or_invalid_state(self):
        state = State().snapshot()
        del state['pending']
        with self.assertRaises(ValueError):
            parse_state(json.dumps(state))
        state = State().snapshot()
        state['known']['mood'] = 'invented'
        with self.assertRaises(ValueError):
            parse_state(json.dumps(state))
        with self.assertRaises(ValueError):
            parse_extraction(json.dumps({'state': State().snapshot()}))

    def test_inference_stops_at_native_eos_or_segment_end_and_rejects_bad_extraction(self):
        predictor = object.__new__(HFPredictor)
        predictor.tokenizer = self.tokenizer
        predictor.style, predictor.device, predictor.torch = 'plain', 'cpu', torch
        predictor.max_new_tokens, predictor.max_length = 1000, 10000
        source = compile_views(self.episode)['extraction'][0]
        request = Request('dst', source['history_before'] + [{'role': 'user', 'content': source['utterance']}],
                          source['state_before'], session=[ubar_user(source['utterance'])])
        for task in ('dst', 'extraction', 'ubar'):
            predictor.task = task
            end_id = encode(self.tokenizer, SEGMENTS['dst'][1])[0] if task == 'ubar' else self.tokenizer.eos_token_id
            for valid in (True, False):
                text = state_text(source['target_state'])
                if task == 'extraction' and valid:
                    text = json.dumps({'state': source['target_state'], 'observation': source['target_observation']})
                elif not valid and task != 'extraction':
                    text = '{}'
                tokens = encode(self.tokenizer, text) + [end_id]
                def generate(**kwargs):
                    self.assertEqual(kwargs['eos_token_id'], end_id)
                    return torch.cat([kwargs['input_ids'], torch.tensor([tokens])], dim=1)
                predictor.model = SimpleNamespace(generate=generate)
                prediction = predictor(request)
                self.assertTrue(prediction.stopped)
                if valid:
                    self.assertIsNone(prediction.error)
                    self.assertEqual(parse_state(prediction.text), source['target_state'])
                elif task == 'extraction':
                    # State-only output must not pass as a valid extraction completion.
                    self.assertIsNotNone(prediction.error)
                else:
                    with self.assertRaises(ValueError):
                        parse_state(prediction.text)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.episode = generate_episode(random.Random(42), 'held-out', 'test')
        # Starting with the user is a data contract, not an episode option.
        self.episode.pop('dialogue_start')

    def test_oracle_all_chains_and_recorded_contexts(self):
        options = [dict(chain=chain, context='reference', user_source='dataset') for chain in
                   ('dst', 'extraction', 'policy', 'response', 'history', 'policy_response', 'full')]
        options.append(dict(chain='full', context='predicted', user_source='dataset'))
        for evaluation in options:
            with self.subTest(**evaluation):
                oracle = ReplayOracle(self.episode)
                report, records = run_benchmark([self.episode], oracle, **evaluation)
                self.assertEqual(report['errors'], {})
                self.assertEqual(report['corpus_hash'], run_benchmark(
                    [self.episode], ReplayOracle(self.episode), **evaluation)[0]['corpus_hash'])
                self.assertEqual({key: report[key] for key in evaluation}, evaluation)
                self.assertNotIn('mode', report)
                self.assertTrue(records)
                self.assertEqual(len(records), len(self.episode['turns']))
                self.assertEqual(report['initial_state'], 'empty_before_first_user')
                for key, metric in report['metrics'].items():
                    self.assertEqual(metric['value'], 0 if key.startswith('behaviour.') else 1, key)
                if evaluation['chain'] == 'full':
                    dst = [request for request in oracle.requests if request.task == 'dst']
                    source = compile_views(self.episode)['extraction'][0]
                    self.assertEqual(dst[0].state, source['state_before'])
                    self.assertEqual(dst[0].state, State().snapshot())
                    self.assertEqual(len(dst), len(self.episode['turns']))
                    self.assertNotIn('<|state|>', dst[0].session[-1])
                    self.assertEqual(dst[0].history[-1]['content'], self.episode['turns'][0]['user'])
                if evaluation['context'] == 'predicted':
                    self.assertEqual(report['context'], 'predicted')
                    self.assertEqual(report['session_summary']['generated_turns'], len(records))
                    self.assertEqual(report['session_outcomes'][0]['terminal'], 'appropriate_close')

    def test_invalid_options_fail_before_prediction_or_model_loading(self):
        from benchmark import main as benchmark_main
        for options in ({'chain': 'session'}, {'context': 'unknown'}, {'user_source': 'unknown'},
                        {'chain': 'dst', 'context': 'predicted'}, {'user_source': 'simulator'}):
            with self.subTest(**options):
                oracle = ReplayOracle(self.episode)
                with self.assertRaises(ValueError):
                    run_benchmark([self.episode], oracle, **options)
                self.assertEqual(oracle.requests, [])
                args = ['benchmark.py', '--output', 'unused']
                for key, value in options.items():
                    args.extend(['--' + key.replace('_', '-'), value])
                with patch('sys.argv', args), patch('benchmark.PredictorRouter') as router, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    benchmark_main()
                self.assertEqual(error.exception.code, 2)
                router.assert_not_called()

    def test_session_carries_generated_memory_and_history_without_reference_fallback(self):
        requests = []
        def model(request):
            requests.append(deepcopy(request))
            if request.task == 'dst':
                state = deepcopy(request.state)
                state['topic'] = 'python'
                return Prediction(state_text(state))
            if request.task == 'policy':
                return Prediction('ASK(mood)' if request.state['last_action'] is None else 'ACK')
            return Prediction('A model response outside the reference templates.')
        report, records = run_benchmark([self.episode], model, chain='full', context='predicted', max_turns=1)
        dst = [r for r in requests if r.task == 'dst']
        self.assertEqual(len(dst), len(self.episode['turns']))
        self.assertEqual(dst[0].state, State().snapshot())
        self.assertEqual(dst[1].state['pending'], 'mood')
        self.assertEqual(dst[1].state['attempts']['mood'], 1)
        self.assertEqual(dst[1].state['topic'], 'python')
        self.assertEqual(dst[1].state['known'], State().snapshot()['known'])
        self.assertEqual(dst[1].history, [self.episode['messages'][0],
            {'role': 'assistant', 'content': 'A model response outside the reference templates.'},
            {'role': 'user', 'content': self.episode['turns'][1]['user']}])
        self.assertIn(segment('policy', 'ASK(mood)'), dst[1].session[0])
        self.assertNotIn(self.episode['turns'][0]['assistant'], dst[1].session[0])
        self.assertEqual(records[0]['stages']['dst']['input']['history'], [self.episode['messages'][0]])
        self.assertEqual(report['session_outcomes'][0]['terminal'], 'reference_exhausted')

    def test_session_failure_or_early_close_keeps_all_turns_in_denominator(self):
        for failure in ('dst', 'policy', 'response', 'close'):
            calls = []
            def model(request):
                calls.append(request)
                if request.task == failure:
                    return Prediction('invalid', stopped=False)
                if request.task == 'dst':
                    return Prediction(state_text(request.state))
                if request.task == 'policy':
                    return Prediction('CLOSE' if failure == 'close' else 'ASK(mood)')
                return Prediction('Bye!')
            with self.subTest(failure=failure):
                report, records = run_benchmark([self.episode], model, chain='full', context='predicted')
                self.assertEqual(len(records), len(self.episode['turns']))
                self.assertLessEqual(len(calls), 3)
                self.assertEqual(report['metrics']['state.exact']['denominator'], len(records))
                self.assertEqual(report['metrics']['action.exact']['denominator'], len(records))
                self.assertEqual(report['session_outcomes'][0]['terminal'],
                                 'premature_close' if failure == 'close' else 'invalid_generation')
                for record in records[1:]:
                    self.assertEqual(record['metrics']['dst.available'], 0)
                    self.assertEqual(record['metrics']['state.exact'], 0)
                    self.assertEqual(record['metrics']['action.exact'], 0)
                    self.assertEqual(record['metrics']['response.reference_exact'], 0)

    def test_invalid_or_unterminated_state_blocks_downstream_without_oracle_fallback(self):
        for prediction in (Prediction('{}'), Prediction(state_text(State().snapshot()), stopped=False)):
            oracle = ReplayOracle(self.episode)
            def broken(request):
                return prediction if request.task == 'dst' else oracle(request)
            report, records = run_benchmark([self.episode], broken, chain='full')
            self.assertEqual(len([r for r in oracle.requests if r.task == 'policy']), 0)
            for record in records:
                self.assertEqual(record['metrics']['state.exact'], 0)
                self.assertEqual(record['metrics']['policy.available'], 0)
                self.assertEqual(record['metrics']['action.exact'], 0)
                self.assertEqual(record['metrics']['response.expected_action_template'], 0)
            self.assertEqual(report['metrics']['action.exact']['denominator'], len(records))

    def test_wrong_but_valid_state_and_action_are_consumed_by_next_stages(self):
        oracle = ReplayOracle(self.episode)
        calls = []
        def mistaken(request):
            calls.append(deepcopy(request))
            if request.task == 'dst':
                state = json.loads(oracle(request).text)
                state['wants_exit'] = True
                return Prediction(state_text(state))
            if request.task == 'response':
                return Prediction(response_variants(request.action)[0])
            return oracle(request)
        report, records = run_benchmark([self.episode], mistaken, chain='full')
        policies = [r for r in calls if r.task == 'policy']
        responses = [r for r in calls if r.task == 'response']
        self.assertTrue(all(r.state['wants_exit'] for r in policies))
        self.assertTrue(all(r.action['type'] == 'CLOSE' for r in responses))
        self.assertGreater(report['errors']['state.wants_exit'], 0)
        self.assertTrue(any(r['metrics'].get('behaviour.premature_close') for r in records))

    def test_unknown_paraphrase_is_marked_for_review(self):
        report, records = run_benchmark([self.episode], lambda r: Prediction('A new paraphrase.'), chain='response')
        self.assertEqual(report['metrics']['response.valid']['value'], 1)
        self.assertEqual(report['metrics']['response.known_template']['value'], 0)
        self.assertIn('response.unrecognized_template_review', report['errors'])
        self.assertNotIn('response.wrong_action_template', report['errors'])

    def test_rollout_uses_own_action_and_carries_memory_before_next_observation(self):
        requests = []
        def model(request):
            requests.append(deepcopy(request))
            if request.task == 'policy':
                return Prediction('ASK(mood)')
            if request.task == 'dst':
                return Prediction(state_text(request.state))
            return Prediction(response_variants(request.action)[0])
        report, records = run_benchmark([self.episode], model, chain='full', context='predicted', user_source='simulator', seed=7, max_turns=3)
        dst = [r for r in requests if r.task == 'dst']
        self.assertEqual(dst[0].state, State().snapshot())
        self.assertEqual(dst[0].history, [self.episode['messages'][0]])
        self.assertEqual(dst[0].session, [ubar_user(self.episode['turns'][0]['user'])])
        self.assertEqual(dst[1].state['pending'], 'mood')
        self.assertEqual(dst[2].state['attempts']['mood'], 2)
        self.assertEqual(dst[1].history[1]['content'], response_variants({'type': 'ASK', 'target': 'mood'})[0])
        self.assertEqual(records[0]['environment']['user'], dst[1].history[-1]['content'])
        self.assertEqual(report['rollout_outcomes'][0]['terminal'], 'horizon_limit')
        self.assertEqual(report, run_benchmark([self.episode], model, chain='full', context='predicted', user_source='simulator', seed=7, max_turns=3)[0])

    def test_rollout_invalid_generation_and_premature_close_are_not_success(self):
        def close(request):
            if request.task == 'dst':
                return Prediction(state_text(request.state))
            return Prediction('CLOSE') if request.task == 'policy' else Prediction(response_variants(
                {'type': 'CLOSE', 'target': None})[0])
        report, _ = run_benchmark([self.episode], close, chain='full', context='predicted', user_source='simulator')
        self.assertEqual(report['rollout_outcomes'][0]['terminal'], 'premature_close')
        report, _ = run_benchmark([self.episode], lambda r: Prediction('not an action'), chain='full', context='predicted', user_source='simulator')
        self.assertEqual(report['rollout_outcomes'][0]['terminal'], 'invalid_generation')

    def test_first_state_failure_blocks_rollout_before_any_action(self):
        calls = []
        def invalid_state(request):
            calls.append(request)
            return Prediction('{}')
        report, records = run_benchmark([self.episode], invalid_state, chain='full', context='predicted', user_source='simulator')
        self.assertEqual([r.task for r in calls], ['dst'])
        self.assertEqual(calls[0].history, [self.episode['messages'][0]])
        self.assertEqual(calls[0].state, State().snapshot())
        self.assertEqual(report['rollout_outcomes'][0]['terminal'], 'invalid_generation')
        self.assertEqual(records[0]['metrics']['policy.available'], 0)
        self.assertEqual(records[0]['metrics']['response.available'], 0)

    def test_invalid_episode_start_is_rejected_before_prediction(self):
        missing_memory = deepcopy(self.episode)
        del missing_memory['turns'][0]['state_before_user']
        missing_user = deepcopy(self.episode)
        missing_user['turns'][0]['user'] = None
        supplied_facts = deepcopy(self.episode)
        supplied_facts['turns'][0]['state_before_user']['known']['mood'] = 'happy'
        for episode in (missing_memory, missing_user, supplied_facts):
            for context, user_source in (('reference', 'dataset'), ('predicted', 'dataset'), ('predicted', 'simulator')):
                with self.subTest(context=context, user_source=user_source, episode=episode['turns'][0]):
                    oracle = ReplayOracle(self.episode)
                    with self.assertRaisesRegex(ValueError, 'user-first episodes|must be empty'):
                        run_benchmark([episode], oracle, context=context, user_source=user_source)
                    self.assertEqual(oracle.requests, [])


class TrainerIntegrationTests(unittest.TestCase):
    def test_tiny_gpt2_optimizer_save_reload_and_generation_for_dst_and_ubar(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, original_threads)
        train = [generate_episode(random.Random(42), 'training', 'train')]
        validation = [generate_episode(random.Random(19), 'validation', 'validation')]
        with tempfile.TemporaryDirectory() as temporary:
            for task in ('dst', 'ubar'):
                with self.subTest(task=task):
                    tokenizer = tokenizer_for(train + validation)
                    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=2048,
                        n_ctx=2048, n_embd=16, n_layer=1, n_head=2, eos_token_id=tokenizer.eos_token_id))
                    before = model.transformer.h[0].attn.c_attn.weight.detach().clone()
                    output = Path(temporary) / task
                    metrics = train_experiment(model, tokenizer, train, validation, task=task, output=output,
                        model_name='tiny-random-gpt2-test', max_length=2048, epochs=2,
                        batch_size=1, warmup_ratio=0, use_cpu=True)
                    manifest = json.loads((output / 'experiment.json').read_text())
                    self.assertEqual(manifest['epochs'], 2)
                    self.assertEqual(metrics['epoch'], 2)
                    self.assertEqual(metrics['global_step'], 2 * manifest['train']['examples'])
                    self.assertTrue(math.isfinite(metrics['eval_loss']))
                    self.assertFalse(torch.equal(before, model.transformer.h[0].attn.c_attn.weight))
                    self.assertEqual(manifest['task'], task)
                    reloaded = HFPredictor(output, local_files_only=True, max_new_tokens=2)
                    row = compile_views(validation[0])['extraction'][0]
                    history = row['history_before'] + [{'role': 'user', 'content': row['utterance']}]
                    result = reloaded(Request('dst', history, row['state_before'], session=[ubar_user(row['utterance'])]))
                    self.assertIsInstance(result, Prediction)
                    self.assertGreater(result.tokens, 0)
                    self.assertLessEqual(result.tokens, 2)
                    from benchmark import main as benchmark_main
                    data_dir = Path(temporary) / f'data-{task}'
                    data_dir.mkdir()
                    held_out = generate_episode(random.Random(20), 'test', 'test')
                    (data_dir / 'test.episodes.json').write_text(json.dumps([held_out]), encoding='utf-8')
                    report_dir = Path(temporary) / f'benchmark-{task}'
                    args = ['benchmark.py', '--model', str(output), '--chain',
                            'dst' if task == 'dst' else 'full', '--context',
                            'reference' if task == 'dst' else 'predicted', '--data', str(data_dir),
                            '--output', str(report_dir), '--max-new-tokens', '2', '--local-files-only']
                    with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                        benchmark_main()
                    report = json.loads((report_dir / 'report.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['episodes'], 1)
                    self.assertEqual(report['chain'], 'dst' if task == 'dst' else 'full')
                    self.assertEqual(report['context'], 'reference' if task == 'dst' else 'predicted')
                    self.assertEqual(report['user_source'], 'dataset')
                    self.assertIn('dst' if task == 'dst' else 'policy', report['models'])
                    with self.assertRaisesRegex(ValueError, 'not empty'):
                        train_experiment(model, tokenizer, train, validation, task=task, output=output)
                    # Release safetensors memory maps before TemporaryDirectory cleanup on Windows.
                    del reloaded, model
                    gc.collect()


if __name__ == '__main__':
    unittest.main()
