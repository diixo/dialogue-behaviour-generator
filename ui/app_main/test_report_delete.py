"""Benchmark deletion only touches selected artifacts inside temporary run roots."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.test import Client, TestCase, override_settings
from django.urls import reverse

from . import artifacts, jobs
from .models import Job
from .report_views import report_files


class ReportDeletionTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / 'runs'
        self.runs.mkdir()
        override = override_settings(DIALOGUE_RUN_DIRS=[self.runs],
                                     DIALOGUE_JOBS_DIR=self.root / 'control')
        override.enable()
        self.addCleanup(override.disable)
        self.addCleanup(artifacts._json.cache_clear)
        self.addCleanup(artifacts._index.cache_clear)

    def make_report(self, folder=None, *, predictions=True):
        folder = folder or self.runs / 'benchmark'
        folder.mkdir(parents=True, exist_ok=True)
        config = {'chain': 'full', 'context': 'reference', 'user_source': 'dataset',
                  'metrics': {}, 'splits': ['test'], 'episodes': 1, 'evaluated_turns': 1}
        (folder / 'report.json').write_text(json.dumps(config), encoding='utf-8')
        if predictions:
            (folder / 'predictions.jsonl').write_text('', encoding='utf-8')
            (folder / 'predictions.json').write_text('[]', encoding='utf-8')
        return folder, artifacts.artifact_id(folder / 'report.json')

    def url(self, identifier):
        return reverse('app_main:delete_report', args=[identifier])

    def test_links_and_confirmation_are_read_only_and_post_requires_csrf(self):
        folder, identifier = self.make_report()
        url = self.url(identifier)
        for route in (reverse('app_main:reports'), reverse('app_main:report_detail', args=[identifier])):
            self.assertContains(self.client.get(route), url)
        client = Client(enforce_csrf_checks=True)
        confirmation = client.get(url)
        self.assertEqual(confirmation.status_code, 200)
        self.assertContains(confirmation, 'Удалить бенчмарк')
        self.assertContains(confirmation, 'predictions.jsonl')
        self.assertContains(confirmation, 'predictions.json')
        self.assertTrue((folder / 'report.json').exists())
        self.assertEqual(client.post(url).status_code, 403)
        self.assertTrue((folder / 'report.json').exists())
        self.assertEqual(self.client.delete(url).status_code, 405)
        result = client.post(url, {'csrfmiddlewaretoken': client.cookies['csrftoken'].value}, follow=True)
        self.assertRedirects(result, reverse('app_main:reports'))
        self.assertContains(result, 'удалён.')
        self.assertFalse(folder.exists())
        self.assertEqual(self.client.get(reverse('app_main:report_detail', args=[identifier])).status_code, 404)
        self.assertEqual(self.client.get(reverse('app_main:prediction', args=[identifier, 0])).status_code, 404)
        self.assertEqual(self.client.post(url).status_code, 404)

    def test_deletion_preserves_models_other_reports_user_files_and_job_history(self):
        folder, identifier = self.make_report()
        for name, value in (('model.safetensors', b'weights'), ('experiment.json', b'{"task": "ubar"}'),
                            ('notes.txt', b'keep this')):
            (folder / name).write_bytes(value)
        child, _ = self.make_report(folder / 'child-report')
        other, _ = self.make_report(self.runs / 'another-benchmark')
        job = Job.objects.create(kind='benchmark', name='Finished evaluation', config={},
                                 output=str(folder), status=Job.Status.COMPLETED)
        control = jobs.directory(job)
        control.mkdir(parents=True)
        (control / 'log.txt').write_text('saved log', encoding='utf-8')
        self.assertRedirects(self.client.post(self.url(identifier)), reverse('app_main:reports'))
        for name in ('report.json', 'predictions.jsonl', 'predictions.json'):
            self.assertFalse((folder / name).exists())
        self.assertEqual((folder / 'model.safetensors').read_bytes(), b'weights')
        self.assertTrue((folder / 'experiment.json').is_file())
        self.assertEqual((folder / 'notes.txt').read_text(), 'keep this')
        self.assertTrue((child / 'report.json').is_file())
        self.assertTrue((other / 'report.json').is_file())
        job.refresh_from_db()
        self.assertEqual(job.status, Job.Status.COMPLETED)
        snapshot = self.client.get(reverse('app_main:job_status', args=[job.pk])).json()
        self.assertEqual(snapshot['result_url'], '')
        self.assertEqual(snapshot['log'], 'saved log')

    def test_active_jobs_block_deletion_including_related_output_directories(self):
        folder, identifier = self.make_report()
        for status in jobs.ACTIVE:
            for output in (folder, self.runs, folder / 'nested'):
                with self.subTest(status=status, output=output):
                    job = Job.objects.create(kind='benchmark', name='Active evaluation', config={},
                                             output=str(output), status=status)
                    self.assertContains(self.client.get(self.url(identifier)), 'Дождитесь завершения')
                    result = self.client.post(self.url(identifier))
                    self.assertContains(result, 'Дождитесь завершения', status_code=409)
                    self.assertEqual(len(list(folder.iterdir())), 3)
                    job.delete()
        Job.objects.create(kind='benchmark', name='Unrelated evaluation', config={},
                           output=str(self.runs / 'unrelated'), status=Job.Status.RUNNING)
        self.assertRedirects(self.client.post(self.url(identifier)), reverse('app_main:reports'))

    def test_deletion_keeps_configured_root_and_accepts_report_without_predictions(self):
        folder, identifier = self.make_report(self.runs, predictions=False)
        self.assertRedirects(self.client.post(self.url(identifier)), reverse('app_main:reports'))
        self.assertTrue(folder.is_dir())
        self.assertEqual(list(folder.iterdir()), [])

    def test_old_report_and_json_only_predictions_can_be_deleted(self):
        folder, identifier = self.make_report()
        (folder / 'report.json').write_text(json.dumps({
            'mode': 'full', 'context': 'reference_history_per_transition', 'metrics': {}}), encoding='utf-8')
        (folder / 'predictions.jsonl').unlink()
        self.assertRedirects(self.client.post(self.url(identifier)), reverse('app_main:reports'))
        self.assertFalse(folder.exists())

    def test_every_target_is_checked_before_deleting_and_paths_cannot_be_submitted(self):
        folder, identifier = self.make_report()
        (folder / 'predictions.json').unlink()
        (folder / 'predictions.json').mkdir()
        result = self.client.post(self.url(identifier))
        self.assertContains(result, 'ожидался файл', status_code=409)
        self.assertTrue((folder / 'predictions.jsonl').exists())
        self.assertTrue((folder / 'report.json').exists())
        outside, outside_id = self.make_report(self.root / 'outside')
        self.assertEqual(self.client.post(self.url(outside_id)).status_code, 404)
        with self.assertRaises(artifacts.ArtifactError):
            report_files({'path': outside})
        (folder / 'predictions.json').rmdir()
        result = self.client.post(self.url(identifier), {'path': str(outside)})
        self.assertRedirects(result, reverse('app_main:reports'))
        self.assertTrue((outside / 'report.json').exists())

    def test_linked_prediction_is_rejected_before_any_file_is_removed(self):
        folder, identifier = self.make_report()
        original = Path.is_symlink

        def is_symlink(path):
            return path == folder / 'predictions.json' or original(path)

        with patch.object(Path, 'is_symlink', is_symlink):
            result = self.client.post(self.url(identifier))
        self.assertContains(result, 'удаление по ссылке', status_code=409)
        self.assertEqual(len(list(folder.iterdir())), 3)

    def test_unlink_failure_is_reported_and_manifest_is_kept_for_retry(self):
        folder, identifier = self.make_report()
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == folder / 'predictions.jsonl':
                raise PermissionError('File is in use')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'unlink', unlink):
            response = self.client.post(self.url(identifier))
        self.assertContains(response, 'File is in use', status_code=409)
        self.assertEqual(len(list(folder.iterdir())), 3)
