"""Real CPU stop/resume and generation tests using a tiny local GPT-2 fixture."""

import gc
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from data import corpus_hash
from experiment_config import TRAIN_DEFAULTS
from generator import generate_episode
from job_runtime import JobCancelled, JobControl
from run_job import execute
from test_experiments import tokenizer_for, ReplayOracle


class JobIntegrationTests(unittest.TestCase):
    def test_stop_resume_optimizer_and_benchmark_with_real_model(self):
        for task in ('dst', 'ubar'):
            with self.subTest(task=task):
                self.check_stop_resume(task)

    def check_stop_resume(self, task):
        from transformers import GPT2Config, GPT2LMHeadModel
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            episodes = {split: [generate_episode(random.Random(seed), split, split)] for split, seed in
                        (('train', 42), ('validation', 19))}
            for split, rows in episodes.items():
                (data / f'{split}.episodes.json').write_text(json.dumps(rows), encoding='utf-8')
            tokenizer = tokenizer_for(episodes['train'] + episodes['validation'])
            model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=2048,
                n_ctx=2048, n_embd=16, n_layer=1, n_head=2, eos_token_id=tokenizer.eos_token_id))
            model.save_pretrained(root / 'base')
            tokenizer.save_pretrained(root / 'base')
            del model, tokenizer
            config = dict(TRAIN_DEFAULTS, task=task, model=str(root / 'base'), cpu=True,
                          local_files_only=True, limit_episodes=None, batch_size=1, max_length=2048,
                          data=str(data), data_hashes={k: corpus_hash(v) for k, v in episodes.items()})

            class StopAfterFirstStep(JobControl):
                def emit(self, **values):
                    super().emit(**values)
                    if values.get('step') == 1:
                        (self.directory / 'stop.request').touch()

            control = StopAfterFirstStep(root / 'control1')
            first = execute({'kind': 'train', 'config': config, 'output': str(root / 'first')}, control)
            self.assertEqual(first['status'], 'stopped')
            checkpoint = Path(first['checkpoint'])
            self.assertTrue((checkpoint / 'optimizer.pt').exists())
            self.assertFalse((root / 'first' / 'metrics.json').exists())
            self.assertEqual(first['metrics']['global_step'], 1)
            manifest_before = (root / 'first' / 'experiment.json').read_bytes()
            continued = execute({'kind': 'train', 'config': config | {'resume': str(checkpoint)},
                                 'output': str(root / 'continued')}, JobControl(root / 'control2'))
            self.assertEqual(continued['status'], 'completed')
            self.assertEqual(continued['metrics']['epoch'], 2)
            manifest = json.loads((root / 'continued' / 'experiment.json').read_text())
            self.assertEqual(continued['metrics']['global_step'], 2 * manifest['train']['examples'])
            self.assertEqual((root / 'first' / 'experiment.json').read_bytes(), manifest_before)
            evaluation = dict(model=str(root / 'continued'), split='validation', chain='dst' if task == 'dst' else 'full',
                              context='reference', user_source='dataset',
                              data=str(data), data_hashes={'validation': corpus_hash(episodes['validation'])},
                              limit_episodes=None, local_files_only=True, device='cpu', max_new_tokens=2,
                              seed=42, max_turns=17)
            report = execute({'kind': 'benchmark', 'config': evaluation, 'output': str(root / 'report')},
                             JobControl(root / 'control3'))
            self.assertEqual(report['status'], 'completed')
            self.assertTrue((root / 'report' / 'report.json').is_file())
            self.assertTrue((root / 'report' / 'predictions.jsonl').is_file())
            # Child processes naturally release memory maps; do so explicitly in this in-process test.
            gc.collect()

    def test_changed_dataset_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'train.episodes.json').write_text(json.dumps([
                generate_episode(random.Random(42), 'train', 'train')]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'train:'):
                execute({'kind': 'train', 'config': {'data': str(root), 'data_hashes': {'train': 'changed'}}},
                        JobControl(root / 'control'))

    def test_benchmark_progress_can_cancel_before_generation(self):
        from benchmark import run_benchmark
        episode = generate_episode(random.Random(42), 'test', 'test')
        with tempfile.TemporaryDirectory() as temporary:
            control = JobControl(temporary)
            (Path(temporary) / 'stop.request').touch()
            oracle = ReplayOracle(episode)
            for context, user_source in (('reference', 'dataset'), ('predicted', 'dataset'), ('predicted', 'simulator')):
                with self.assertRaises(JobCancelled):
                    run_benchmark([episode], oracle, context=context, user_source=user_source,
                                  progress=control.benchmark_progress)
            self.assertEqual(oracle.requests, [])
