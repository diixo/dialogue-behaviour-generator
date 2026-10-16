"""HTTP handlers only enqueue and inspect jobs. A separate process executes them."""

from pathlib import Path
import uuid

from django.http import JsonResponse, Http404
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.views.decorators.http import require_POST

from . import artifacts, jobs
from .forms import BenchmarkForm, ResumeForm, TrainForm
from .models import Job
from .views import page, paginate

PHASES = {'data': 'Проверка данных', 'loading': 'Загрузка модели', 'prepared': 'Данные подготовлены',
          'training': 'Обучение', 'saving': 'Сохранение checkpoint', 'benchmark': 'Оценка',
          'chat_dst': 'Модель обновляет состояние', 'chat_policy': 'Модель выбирает действие',
          'chat_response': 'Модель формулирует ответ', 'chat_history': 'Модель отвечает по переписке'}


def job_list(request):
    return page(request, 'jobs', 'Очередь задач', 'jobs', worker_online=jobs.worker_online(),
                **paginate(request, Job.objects.all()))


def submit(request, kind):
    form_class = {'train': TrainForm, 'benchmark': BenchmarkForm, 'resume': ResumeForm}[kind]
    initial = {}
    if kind == 'benchmark' and request.GET.get('model'):
        initial['model'] = request.GET['model']
    if kind == 'resume' and request.GET.get('job'):
        try:
            source_id = uuid.UUID(request.GET['job'])
        except ValueError:
            raise Http404
        source = get_object_or_404(Job, pk=source_id, kind='train')
        checkpoint = source.result.get('checkpoint') or source.progress.get('checkpoint')
        if checkpoint:
            initial['checkpoint'] = artifacts.artifact_id(Path(checkpoint))
        initial.update({key: source.config.get(key) for key in ('cpu', 'limit_episodes', 'local_files_only')})
    form = form_class(request.POST if request.method == 'POST' else None, initial=initial)
    if request.method == 'POST' and form.is_valid():
        try:
            config = form.job_config()
            name = form.cleaned_data.get('name') or (f'Обучение {config["task"]}' if kind != 'benchmark'
                                                    else f'Оценка {config["chain"]} · {config["context"]} · {config["user_source"]}')
            job = jobs.enqueue('train' if kind == 'resume' else kind, name, config,
                               identifier=form.cleaned_data['submission_id'])
            return redirect('app_main:job', identifier=job.pk)
        except (artifacts.ArtifactError, ValueError) as error:
            form.add_error(None, str(error))
    splits, issues = artifacts.dataset_summary()
    title = {'train': 'Новое обучение', 'benchmark': 'Новый бенчмарк', 'resume': 'Продолжить обучение'}[kind]
    return page(request, 'job_form', title, 'jobs', form=form, kind=kind, splits=splits, issues=issues,
                worker_online=jobs.worker_online())


def new_training(request):
    return submit(request, 'train')


def new_benchmark(request):
    return submit(request, 'benchmark')


def resume_training(request):
    return submit(request, 'resume')


def snapshot(job):
    progress = job.progress
    output = Path(job.output)
    filename, route = ('experiment.json', 'experiment') if job.kind == 'train' else ('report.json', 'report_detail')
    result_url = reverse(f'app_main:{route}', args=[artifacts.artifact_id(output / filename)]) if (output / filename).is_file() else ''
    if job.chat_session_id:
        result_url = reverse('app_main:chat', args=[job.chat_session_id])
    checkpoint = job.result.get('checkpoint') or progress.get('checkpoint')
    resumable = bool(job.kind == 'train' and job.status in ('stopped', 'failed', 'cancelled')
                     and checkpoint and Path(checkpoint).is_dir())
    return {'status': job.status, 'label': job.get_status_display(), 'active': job.active,
            'phase': PHASES.get(progress.get('phase'), 'Ожидание worker' if job.status == 'queued' else ''),
            'percent': min(100, max(0, progress.get('percent', 0))) if job.status != 'completed' else 100,
            'progress': progress, 'error': job.error, 'log': jobs.log_tail(job),
            'worker_online': jobs.worker_online(), 'result_url': result_url, 'resumable': resumable,
            'can_stop': job.status in (Job.Status.QUEUED, Job.Status.RUNNING)}


def job_detail(request, identifier):
    job = get_object_or_404(Job, pk=identifier)
    if job.chat_session_id:
        return redirect('app_main:chat', identifier=job.chat_session_id)
    return page(request, 'job', job.name, 'jobs', job=job, snapshot=snapshot(job))


def job_status(request, identifier):
    response = JsonResponse(snapshot(get_object_or_404(Job, pk=identifier)))
    response['Cache-Control'] = 'no-store'
    return response


@require_POST
def stop_job(request, identifier):
    job = get_object_or_404(Job, pk=identifier)
    jobs.request_stop(job)
    if job.chat_session_id:
        return redirect('app_main:chat', identifier=job.chat_session_id)
    return redirect('app_main:job', identifier=job.pk)
