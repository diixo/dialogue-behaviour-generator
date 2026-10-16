"""Persistent queue and a single local worker; no ML imports in the web process."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from job_runtime import FileLock, atomic_json, lock_available
from . import artifacts
from .models import Job, Worker

ACTIVE = (Job.Status.QUEUED, Job.Status.RUNNING, Job.Status.STOPPING)


def directory(job):
    return Path(settings.DIALOGUE_JOBS_DIR) / str(job.pk)


def has_weights(path):
    return (path / 'config.json').is_file() and any((path / name).is_file() for name in (
        'model.safetensors', 'pytorch_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json'))


def checkpoints(resumable=False):
    choices = {}
    for run in artifacts.discover()[0]:
        paths = sorted(run['path'].glob('checkpoint-*'))
        if not resumable:
            paths.insert(0, run['path'])
        for path in paths:
            if path.is_symlink() or not path.is_dir() or not has_weights(path):
                continue
            if resumable and not all((path / name).is_file() for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt')):
                continue
            identifier = artifacts.artifact_id(path)
            label = run['name'] + (f' / {path.name}' if path != run['path'] else ' / итоговая модель')
            choices[identifier] = {'path': path.resolve(), 'run': run, 'task': run['config']['task'],
                                   'label': f'{label} · {run["config"]["task"]}'}
    return choices


def data_config(splits):
    hashes = {}
    for split in splits:
        index = artifacts.dataset_index(split)
        if not index.entries:
            raise artifacts.ArtifactError(f'{split}: датасет пуст.')
        hashes[split] = index.digest
    return {'data': str(Path(settings.DIALOGUE_DATA_DIR).resolve()), 'data_hashes': hashes}


def enqueue(kind, name, config, output=None, identifier=None, chat_session=None):
    job = Job(id=identifier or uuid.uuid4(), kind=kind, name=name, config=config)
    job.output = str(Path(output).resolve() if output else
                     Path(settings.DIALOGUE_OUTPUT_DIR).resolve() / f'{kind}-{job.pk}')
    with transaction.atomic():
        job, created = Job.objects.get_or_create(pk=job.pk, defaults={
            'kind': kind, 'name': name, 'config': config, 'output': job.output, 'chat_session': chat_session})
        if not created:
            if job.config != config or job.kind != kind or job.chat_session_id != getattr(chat_session, 'pk', None):
                raise ValueError('Эта форма уже отправлена с другими параметрами. Откройте новую форму.')
            return job
        if Job.objects.filter(output=job.output, status__in=ACTIVE).exclude(pk=job.pk).exists():
            raise ValueError('Для этого эксперимента уже есть задача в очереди или в работе.')
    return job


def request_stop(job):
    with transaction.atomic():
        if Job.objects.filter(pk=job.pk, status=Job.Status.QUEUED).update(
                status=Job.Status.CANCELLED, finished_at=timezone.now()):
            return
        if Job.objects.filter(pk=job.pk, status__in=(Job.Status.RUNNING, Job.Status.STOPPING)).update(
                status=Job.Status.STOPPING):
            folder = directory(job)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'stop.request').touch()


def worker_online():
    return Worker.objects.filter(pk=1, heartbeat__gte=timezone.now() - timedelta(seconds=15)).exists()


def read_control(job, filename):
    try:
        return json.loads((directory(job) / filename).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def log_tail(job):
    try:
        with (directory(job) / 'log.txt').open('rb') as source:
            source.seek(0, 2)
            source.seek(max(0, source.tell() - 24000))
            return source.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def heartbeat():
    Worker.objects.update_or_create(pk=1, defaults={'heartbeat': timezone.now()})


def update_progress(job):
    value = read_control(job, 'progress.json')
    if value:
        Job.objects.filter(pk=job.pk).update(progress=value)


def finish(job, returncode=None):
    result = read_control(job, 'result.json')
    status = result.get('status', Job.Status.FAILED)
    if status not in (Job.Status.COMPLETED, Job.Status.STOPPED, Job.Status.CANCELLED, Job.Status.FAILED):
        status = Job.Status.FAILED
    error = result.get('error', '')
    if not result:
        error = f'Процесс завершился без результата (код: {returncode}). Проверьте журнал; сохранённый checkpoint можно продолжить.'
    update_progress(job)
    Job.objects.filter(pk=job.pk).update(status=status, result=result, error=error, finished_at=timezone.now())


def claim_next():
    job = Job.objects.filter(status=Job.Status.QUEUED).order_by('created_at').first()
    if job and Job.objects.filter(pk=job.pk, status=Job.Status.QUEUED).update(
            status=Job.Status.RUNNING, started_at=timezone.now()):
        job.refresh_from_db()
        return job
    return None


def monitor(job, process=None):
    """Also adopt a child left running after the previous worker exited."""
    interrupted = False
    while True:
        heartbeat()
        update_progress(job)
        done = process.poll() is not None if process is not None else lock_available(
            Path(settings.DIALOGUE_JOBS_DIR) / 'execution.lock')
        if done:
            finish(job, process.returncode if process else None)
            if interrupted:
                raise KeyboardInterrupt
            return
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            if interrupted:
                raise
            interrupted = True
            request_stop(job)


def run_one(job):
    folder = directory(job)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / 'spec.json', {'kind': job.kind, 'config': job.config, 'output': job.output})
    environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', TOKENIZERS_PARALLELISM='false')
    try:
        with (folder / 'log.txt').open('ab') as log:
            process = subprocess.Popen([sys.executable, '-u', str(settings.PROJECT_ROOT / 'run_job.py'),
                                        str(folder / 'spec.json')], cwd=settings.PROJECT_ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            monitor(job, process)
    except OSError as error:
        Job.objects.filter(pk=job.pk).update(status=Job.Status.FAILED, error=str(error), finished_at=timezone.now())


def run_worker(once=False):
    with FileLock(Path(settings.DIALOGUE_JOBS_DIR) / 'worker.lock'):
        try:
            heartbeat()
            for job in Job.objects.filter(status__in=(Job.Status.RUNNING, Job.Status.STOPPING)):
                if lock_available(Path(settings.DIALOGUE_JOBS_DIR) / 'execution.lock'):
                    folder = directory(job)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / 'stop.request').touch()
                    finish(job)
                else:
                    monitor(job)
            while True:
                heartbeat()
                job = claim_next()
                if job:
                    run_one(job)
                if once:
                    return
                time.sleep(1)
        finally:
            Worker.objects.filter(pk=1).delete()
