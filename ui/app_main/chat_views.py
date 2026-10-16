"""Interactive conversations share the worker queue with training and evaluation."""

from pathlib import Path

from django.conf import settings
from django.db import IntegrityError, transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.views.decorators.http import require_POST

from . import artifacts, jobs
from .chat_forms import ChatForm, MessageForm
from .job_views import PHASES
from .models import ChatSession, Job
from .views import page, paginate, pretty


def completed_jobs(chat):
    return list(chat.jobs.filter(status=Job.Status.COMPLETED).order_by('created_at', 'id'))


def enqueue_message(chat, message, identifier, previous_job=None):
    with transaction.atomic():
        existing = Job.objects.filter(pk=identifier).first()
        if existing:
            if (existing.chat_session_id != chat.pk or existing.config.get('message') != message or
                    existing.config.get('previous_job') != (str(previous_job) if previous_job else None)):
                raise ValueError('Эта форма уже отправлена. Обновите страницу.')
            return existing
        completed = completed_jobs(chat)
        last = completed[-1] if completed else None
        if previous_job != (last.pk if last else None):
            raise ValueError('Диалог изменился в другой вкладке. Обновите страницу перед отправкой.')
        if chat.jobs.filter(status__in=jobs.ACTIVE).exists():
            raise ValueError('Дождитесь ответа на предыдущее сообщение.')
        if last and last.result.get('closed'):
            raise ValueError('Модель завершила разговор. Начните новый диалог.')
        if not message or not message.strip():
            raise ValueError('Введите сообщение.')
        config = dict(chat.config, turns=[job.result for job in completed], message=message,
                      previous_job=str(previous_job) if previous_job else None)
        return jobs.enqueue('chat', f'Чат · ход {len(completed) + 1}', config, identifier=identifier,
                            output=Path(settings.DIALOGUE_JOBS_DIR) / str(identifier) / 'chat', chat_session=chat)


def chats(request):
    form = ChatForm(request.POST if request.method == 'POST' else None,
                    initial={'model': request.GET.get('model')} if request.GET.get('model') else None)
    if request.method == 'POST' and form.is_valid():
        try:
            with transaction.atomic():
                config = form.chat_config()
                chat, _ = ChatSession.objects.get_or_create(pk=form.cleaned_data['submission_id'], defaults={'config': config})
                if chat.config != config:
                    raise ValueError('Эта форма уже отправлена с другими параметрами. Обновите страницу.')
            return redirect('app_main:chat', identifier=chat.pk)
        except (ValueError, IntegrityError) as error:
            form.add_error(None, str(error) if isinstance(error, ValueError) else 'Задача уже поставлена в очередь.')
    return page(request, 'chats', 'Чат с моделью', 'chats', form=form,
                worker_online=jobs.worker_online(), **paginate(request, ChatSession.objects.all()))


def render_chat(request, chat, form=None):
    turns = list(chat.jobs.order_by('created_at', 'id'))
    completed = [job for job in turns if job.status == Job.Status.COMPLETED]
    last = completed[-1] if completed else None
    active = next((job for job in turns if job.active), None)
    for job in turns:
        job.state_json = pretty(job.result.get('state'))
        job.memory_json = pretty(job.result.get('memory'))
        job.action_json = pretty(job.result.get('action'))
    form = form or MessageForm(initial={'previous_job': last.pk if last else None})
    return page(request, 'chat', 'Диалог с моделью', 'chats', chat=chat, turns=turns, form=form,
                active_job=active, phase=PHASES.get(active.progress.get('phase'), active.get_status_display()) if active else '',
                started=bool(completed), closed=bool(last and last.result.get('closed')),
                worker_online=jobs.worker_online(), model_id=artifacts.artifact_id(Path(chat.config['model'])))


def chat_detail(request, identifier):
    return render_chat(request, get_object_or_404(ChatSession, pk=identifier))


@require_POST
def send_message(request, identifier):
    chat = get_object_or_404(ChatSession, pk=identifier)
    form = MessageForm(request.POST)
    if form.is_valid():
        try:
            enqueue_message(chat, form.cleaned_data['message'] or None,
                            form.cleaned_data['submission_id'], form.cleaned_data['previous_job'])
            return redirect('app_main:chat', identifier=chat.pk)
        except (ValueError, IntegrityError) as error:
            form.add_error(None, str(error) if isinstance(error, ValueError) else 'Для этого диалога уже выполняется запрос.')
    return render_chat(request, chat, form)


def chat_status(request, identifier):
    chat = get_object_or_404(ChatSession, pk=identifier)
    active = chat.jobs.filter(status__in=jobs.ACTIVE).first()
    response = JsonResponse({'active': bool(active), 'worker_online': jobs.worker_online(),
                             'phase': PHASES.get(active.progress.get('phase'), active.get_status_display()) if active else ''})
    response['Cache-Control'] = 'no-store'
    return response


@require_POST
def stop_chat(request, identifier):
    chat = get_object_or_404(ChatSession, pk=identifier)
    active = chat.jobs.filter(status__in=jobs.ACTIVE).first()
    if active:
        jobs.request_stop(active)
    return redirect('app_main:chat', identifier=chat.pk)
