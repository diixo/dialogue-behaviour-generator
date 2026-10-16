import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from unittest.mock import patch
import uuid

from django.test import Client, TestCase, override_settings
from django.urls import reverse

from experiment_config import TRAIN_DEFAULTS
from generator import generate_episode
from job_runtime import FileLock, LockBusy, atomic_json
from . import artifacts, jobs
from .forms import BenchmarkForm, ResumeForm
from .models import Job, Worker


class QueueTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.runs = self.root / 'data', self.root / 'runs'
        self.data.mkdir()
        self.runs.mkdir()
        override = override_settings(DIALOGUE_DATA_DIR=self.data, DIALOGUE_RUN_DIRS=[self.runs],
                                     DIALOGUE_OUTPUT_DIR=self.runs, DIALOGUE_JOBS_DIR=self.root / 'control')
        override.enable()
        self.addCleanup(override.disable)
        self.addCleanup(artifacts._index.cache_clear)
        self.addCleanup(artifacts._json.cache_clear)
        for split in ('train', 'validation', 'test'):
            episode = generate_episode(random.Random(42), split, split)
            (self.data / f'{split}.episodes.jsonl').write_text(json.dumps(episode) + '\n', encoding='utf-8')

    def config(self):
        return dict(TRAIN_DEFAULTS, cpu=True, local_files_only=True, limit_episodes=None,
                    **jobs.data_config(('train', 'validation')))

    def checkpoint(self, task='ubar', epoch=1):
        output = self.runs / task
        config = dict(TRAIN_DEFAULTS, base_model='gpt2', task=task, gradient_accumulation_steps=1,
                      train_hash=artifacts.dataset_index('train').digest,
                      validation_hash=artifacts.dataset_index('validation').digest)
        atomic_json(output / 'experiment.json', config)
        path = output / 'checkpoint-1'
        atomic_json(path / 'config.json', {})
        atomic_json(path / 'trainer_state.json', {'epoch': epoch, 'global_step': 1})
        for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt'):
            (path / name).write_bytes(b'fixture')
        return artifacts.artifact_id(path), path

    def test_forms_render_and_post_only_enqueues_idempotently(self):
        for route in ('jobs', 'new_training', 'new_benchmark', 'resume_training'):
            self.assertEqual(self.client.get(reverse('app_main:' + route)).status_code, 200)
        data = dict(TRAIN_DEFAULTS, submission_id=str(uuid.uuid4()), cpu='on')
        with patch('app_main.jobs.subprocess.Popen') as process:
            first = self.client.post(reverse('app_main:new_training'), data)
            second = self.client.post(reverse('app_main:new_training'), data)
        self.assertEqual(first.status_code, 302)
        self.assertEqual(second.url, first.url)
        process.assert_not_called()
        job = Job.objects.get()
        self.assertEqual(job.status, 'queued')
        self.assertEqual(job.config['epochs'], 2)
        self.assertEqual(job.config['model'], 'gpt2')
        self.assertNotIn('max_steps', job.config)
        self.assertEqual(job.config['data_hashes']['train'], artifacts.dataset_index('train').digest)
        self.assertFalse(Path(job.output).exists())

    def test_stop_is_csrf_protected_and_queued_jobs_never_launch(self):
        job = jobs.enqueue('train', 'queued', self.config())
        url = reverse('app_main:stop_job', args=[job.pk])
        self.assertEqual(self.client.get(url).status_code, 405)
        self.assertEqual(Client(enforce_csrf_checks=True).post(url).status_code, 403)
        self.assertEqual(self.client.post(url).status_code, 302)
        job.refresh_from_db()
        self.assertEqual(job.status, 'cancelled')
        self.assertIsNone(jobs.claim_next())

    def test_running_stop_uses_cooperative_flag_and_progress_endpoint(self):
        job = jobs.enqueue('train', 'running', self.config())
        jobs.claim_next()
        jobs.request_stop(job)
        self.assertTrue((jobs.directory(job) / 'stop.request').exists())
        atomic_json(jobs.directory(job) / 'progress.json', {'epoch': .5, 'percent': 25, 'loss': .8})
        (jobs.directory(job) / 'log.txt').write_text('<script>bad()</script>', encoding='utf-8')
        jobs.update_progress(job)
        response = self.client.get(reverse('app_main:job_status', args=[job.pk]))
        self.assertEqual(response.json()['status'], 'stopping')
        self.assertEqual(response.json()['percent'], 25)
        self.assertEqual(response['Cache-Control'], 'no-store')
        detail = self.client.get(reverse('app_main:job', args=[job.pk]))
        self.assertContains(detail, '&lt;script&gt;')
        self.assertNotContains(detail, '<script>bad()</script>')

    def test_nonfinite_progress_does_not_break_database_or_json(self):
        job = jobs.enqueue('train', 'numeric edge case', self.config())
        atomic_json(jobs.directory(job) / 'progress.json', {'loss': float('nan'), 'grad_norm': float('inf')})
        jobs.update_progress(job)
        job.refresh_from_db()
        self.assertEqual(job.progress, {'loss': None, 'grad_norm': None})

    def test_single_worker_lock_and_recovery_without_relaunch(self):
        with FileLock(self.root / 'control' / 'worker.lock'):
            with self.assertRaises(LockBusy):
                jobs.run_worker(once=True)
        job = jobs.enqueue('train', 'orphan', self.config())
        jobs.claim_next()
        with patch('app_main.jobs.subprocess.Popen') as process:
            jobs.run_worker(once=True)
        process.assert_not_called()
        job.refresh_from_db()
        self.assertEqual(job.status, 'failed')
        self.assertTrue((jobs.directory(job) / 'stop.request').exists())
        self.assertFalse(Worker.objects.exists())

    def test_recovery_preserves_completed_child_result(self):
        job = jobs.enqueue('benchmark', 'finished child', {})
        jobs.claim_next()
        atomic_json(jobs.directory(job) / 'result.json', {'status': 'completed'})
        jobs.run_worker(once=True)
        job.refresh_from_db()
        self.assertEqual(job.status, 'completed')

    def test_worker_launch_failure_and_fifo(self):
        first = jobs.enqueue('train', 'first', self.config())
        second = jobs.enqueue('train', 'second', self.config())
        claimed = jobs.claim_next()
        self.assertEqual(claimed.pk, first.pk)
        with patch('app_main.jobs.subprocess.Popen', side_effect=OSError('Cannot start Python')):
            jobs.run_one(claimed)
        first.refresh_from_db()
        self.assertEqual(first.status, 'failed')
        self.assertIn('Cannot start Python', first.error)
        self.assertEqual(jobs.claim_next().pk, second.pk)

    def test_worker_executes_child_and_collects_progress_log_and_result(self):
        # A tiny process exercises IPC and queue lifecycle; real ML is covered by test_job_runtime.
        project = self.root / 'project'
        project.mkdir()
        (project / 'run_job.py').write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "p = Path(sys.argv[1]).parent\n"
            "(p / 'progress.json').write_text(json.dumps({'percent': 100, 'phase': 'benchmark'}))\n"
            "(p / 'result.json').write_text(json.dumps({'status': 'completed'}))\n"
            "print('child completed', flush=True)\n", encoding='utf-8')
        job = jobs.enqueue('benchmark', 'child test', {})
        with override_settings(PROJECT_ROOT=project):
            jobs.run_worker(once=True)
        job.refresh_from_db()
        self.assertEqual(job.status, 'completed')
        self.assertEqual(job.progress['percent'], 100)
        self.assertIn('child completed', jobs.log_tail(job))
        self.assertFalse(jobs.worker_online())

    def test_benchmark_rejects_incompatible_stage_and_accepts_ubar(self):
        policy, _ = self.checkpoint('policy')
        data = dict(submission_id=uuid.uuid4(), chain='full', context='reference', user_source='dataset',
                    split='validation', model=policy,
                    device='cpu', max_new_tokens=384, max_turns=17, seed=42)
        self.assertFalse(BenchmarkForm(data).is_valid())
        ubar, path = self.checkpoint()
        form = BenchmarkForm(data | {'model': ubar})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.job_config()['model'], str(path.resolve()))
        self.assertFalse(BenchmarkForm(data | {'model': '../outside'}).is_valid())
        self.assertFalse(BenchmarkForm(data | {'context': 'predicted'}).is_valid())
        for source in ('dataset', 'simulator'):
            form = BenchmarkForm(data | {'model': ubar, 'context': 'predicted', 'user_source': source})
            self.assertTrue(form.is_valid(), form.errors)
            self.assertEqual(form.job_config()['context'], 'predicted')
            self.assertEqual(form.job_config()['user_source'], source)
            self.assertNotIn('mode', form.job_config())

    def test_benchmark_options_validate_and_reach_queued_job(self):
        ubar, _ = self.checkpoint()
        data = dict(submission_id=uuid.uuid4(), chain='full', context='predicted', user_source='dataset',
                    split='validation', model=ubar, device='cpu', max_new_tokens=192, max_turns=17, seed=42)
        for options in ({'chain': 'policy'}, {'context': 'reference', 'user_source': 'simulator'},
                        {'chain': 'session'}, {'user_source': 'unknown'}):
            with self.subTest(**options):
                response = self.client.post(reverse('app_main:new_benchmark'), data | options)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.context['form'].errors)
                self.assertFalse(Job.objects.exists())
        response = self.client.post(reverse('app_main:new_benchmark'), data)
        self.assertEqual(response.status_code, 302)
        config = Job.objects.get().config
        self.assertEqual({key: config[key] for key in ('chain', 'context', 'user_source')},
                         dict(chain='full', context='predicted', user_source='dataset'))
        self.assertNotIn('mode', config)

    def test_resume_keeps_budget_and_forks_artifacts(self):
        identifier, path = self.checkpoint()
        values = dict(submission_id=uuid.uuid4(), checkpoint=identifier, cpu=True, local_files_only=True)
        form = ResumeForm(values)
        self.assertTrue(form.is_valid(), form.errors)
        config = form.job_config()
        self.assertEqual(config['epochs'], 2)
        response = self.client.post(reverse('app_main:resume_training'), values)
        self.assertEqual(response.status_code, 302)
        job = Job.objects.get()
        self.assertNotEqual(Path(job.output), path.parent)
        self.assertEqual(job.config['resume'], str(path.resolve()))
        self.checkpoint(epoch=2)
        self.assertFalse(ResumeForm(values).is_valid())

    def test_resume_rejects_changed_data(self):
        identifier, _ = self.checkpoint()
        (self.data / 'train.episodes.jsonl').write_text(json.dumps(
            generate_episode(random.Random(19), 'other', 'train')) + '\n', encoding='utf-8')
        form = ResumeForm({'submission_id': uuid.uuid4(), 'checkpoint': identifier})
        self.assertFalse(form.is_valid())
        self.assertEqual(self.client.get('/jobs/resume/?job=bad').status_code, 404)
