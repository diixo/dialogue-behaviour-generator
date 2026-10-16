"""Delete selected benchmark artifacts while retaining the job history."""

from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.shortcuts import redirect
from django.views.decorators.http import require_http_methods

from . import artifacts, jobs
from .models import Job
from .views import artifact_errors, find_artifact, page


REPORT_FILES = ('predictions.jsonl', 'predictions.json', 'report.json')


def report_files(run):
    """Resolve and check every target before deleting any file; never recurse."""
    folder = run['path'].resolve()
    roots = [Path(root).resolve() for root in settings.DIALOGUE_RUN_DIRS]
    if run['path'].is_symlink() or not any(folder.is_relative_to(root) for root in roots):
        raise artifacts.ArtifactError('Отчёт находится вне каталогов бенчмарков.')
    files = []
    for name in REPORT_FILES:
        path = folder / name
        if path.is_symlink() or path.resolve().parent != folder:
            raise artifacts.ArtifactError(f'{name}: удаление по ссылке не поддерживается.')
        if path.exists():
            if not path.is_file():
                raise artifacts.ArtifactError(f'{name}: ожидался файл.')
            files.append(path)
    if folder / 'report.json' not in files:
        raise artifacts.ArtifactError('Отчёт уже удалён. Обновите список бенчмарков.')
    return files


def check_active_jobs(folder):
    for output in Job.objects.filter(status__in=jobs.ACTIVE).values_list('output', flat=True):
        path = Path(output)
        if not path.is_absolute():
            path = settings.PROJECT_ROOT / path
        path = path.resolve()
        if folder.is_relative_to(path) or path.is_relative_to(folder):
            raise artifacts.ArtifactError(
                'В этом каталоге есть задача в очереди или в работе. '
                'Дождитесь завершения или остановите её перед удалением отчёта.')


@require_http_methods(['GET', 'POST'])
@artifact_errors
def delete_report(request, identifier):
    run = find_artifact(identifier, 'report')
    error = ''
    files = []
    try:
        files = report_files(run)
        folder = files[-1].parent
        if request.method == 'POST':
            with transaction.atomic():
                check_active_jobs(folder)
                # Remove the manifest last so an interrupted deletion can be retried.
                for path in files:
                    path.unlink()
                roots = [Path(root).resolve() for root in settings.DIALOGUE_RUN_DIRS]
                if folder not in roots:
                    try:
                        folder.rmdir()
                    except OSError:
                        # Keep directories containing models, other reports or user files.
                        pass
            messages.success(request, f'Бенчмарк «{run["name"]}» удалён.')
            return redirect('app_main:reports')
        check_active_jobs(folder)
    except (artifacts.ArtifactError, OSError) as exception:
        error = str(exception)
    response = page(request, 'report_delete', 'Удаление бенчмарка', 'reports',
                    run=run, files=files, deletion_error=error)
    if error and request.method == 'POST':
        response.status_code = 409
    return response
