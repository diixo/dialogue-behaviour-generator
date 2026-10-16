"""Read-only dataset, training-artifact and benchmark views."""

from functools import wraps
import json
import math
from pathlib import Path

from django.conf import settings
from django.core.paginator import Paginator
from django.http import Http404
from django.shortcuts import redirect, render
from django.urls import reverse

from compile_sft import compile_row, serialize_action
from generator import compile_views
from . import artifacts


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def page(request, template, title, section, **context):
    navigation = [('main', 'Обзор', 'dashboard'), ('datasets', 'Данные', 'storage'),
                  ('experiments', 'Обучение', 'model_training'), ('reports', 'Бенчмарки', 'add_chart'),
                  ('jobs', 'Очередь', 'playlist_play'), ('chats', 'Чат', 'reviews')]
    return render(request, f'app_main/{template}.html', {
        'title': title, 'section': section, 'navigation': navigation, **context,
    })


def artifact_errors(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except artifacts.ArtifactError as error:
            response = page(request, 'error', 'Не удалось прочитать данные', '', error=str(error))
            response.status_code = 422
            return response
    return wrapped


def paginate(request, rows, size=25):
    query = request.GET.copy()
    query.pop('page', None)
    return {'items': Paginator(rows, size).get_page(request.GET.get('page')),
            'page_query': query.urlencode()}


def find_artifact(identifier, kind):
    experiments, reports, issues = artifacts.discover()
    for item in experiments if kind == 'experiment' else reports:
        if item['id'] == identifier:
            return item
    raise Http404('Артефакт не найден в настроенных каталогах.')


def main(request):
    splits, issues = artifacts.dataset_summary()
    experiments, reports, discovery_issues = artifacts.discover()
    return page(request, 'index', 'Обзор', 'main', splits=splits,
                episode_count=sum(s['episodes'] for s in splits),
                turn_count=sum(s['turns'] for s in splits),
                experiment_count=len(experiments), report_count=len(reports),
                experiments=experiments[:5], reports=reports[:5], issues=issues + discovery_issues)


@artifact_errors
def datasets(request):
    split = request.GET.get('split', 'train')
    if split not in artifacts.SPLITS:
        raise Http404('Неизвестное разбиение.')
    summaries, issues = artifacts.dataset_summary()
    index = artifacts.dataset_index(split)
    rows = index.entries
    query = request.GET.get('q', '').strip()
    reaction, action = request.GET.get('reaction', ''), request.GET.get('action', '')
    target, terminal = request.GET.get('target', ''), request.GET.get('terminal', '')
    if query:
        rows = [r for r in rows if query.casefold() in r['episode_id'].casefold()]
    for value, key in ((reaction, 'reactions'), (action, 'actions'), (target, 'targets')):
        if value:
            rows = [r for r in rows if value in r[key]]
    if terminal:
        rows = [r for r in rows if r['terminal'] == terminal]
    return page(request, 'datasets', 'Данные', 'datasets', splits=summaries, split=split,
                q=query, reaction=reaction, action=action, target=target, terminal=terminal,
                distribution=artifacts.distributions(index),
                terminals=sorted({r['terminal'] for r in index.entries}),
                issues=issues, split_choices=artifacts.SPLITS,
                dataset_path=settings.DIALOGUE_DATA_DIR, **paginate(request, rows))


def differences(before, after, prefix=''):
    result = []
    for key in sorted(before.keys() | after.keys()):
        name = f'{prefix}.{key}' if prefix else key
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            result.extend(differences(old, new, name))
        elif old != new:
            result.append({'field': name, 'before': pretty(old), 'after': pretty(new)})
    return result


@artifact_errors
def episode(request, split, episode_id):
    if split not in artifacts.SPLITS:
        raise Http404
    index = artifacts.dataset_index(split)
    entry = next((r for r in index.entries if r['episode_id'] == episode_id), None)
    if entry is None:
        raise Http404('Эпизод не найден.')
    data = index.record(entry['position'])
    try:
        selected = int(request.GET.get('turn', 0))
    except ValueError:
        raise Http404('Неизвестный ход.')
    if not 0 <= selected < len(data['turns']):
        raise Http404('Неизвестный ход.')
    turn = data['turns'][selected]
    views = compile_views(data)
    extraction = next((r for r in views['extraction'] if r['turn'] == selected), None)
    next_turn = extraction['next_turn'] if extraction else None
    examples = []
    if extraction:
        examples.append(compile_row(extraction, 'extraction'))
    for task in ('policy', 'response', 'history'):
        task_turn = next_turn if extraction else selected
        row = next((r for r in views[task] if r['turn'] == task_turn), None)
        if row:
            examples.append(compile_row(row, task))
    for example in examples:
        example['messages_json'] = pretty(example['messages'])
        try:
            example['completion_display'] = pretty(json.loads(example['completion']))
        except (ValueError, TypeError):
            example['completion_display'] = example['completion']
    before = extraction['state_before'] if extraction else turn['state_before']
    after = extraction['target_state'] if extraction else turn['state_after']
    return page(request, 'episode', episode_id, 'datasets', episode=data, split=split,
                turn=turn, selected=selected, action=serialize_action(turn['action']),
                next_turn=next_turn, examples=examples, extraction=extraction,
                changes=differences(before, after), before_json=pretty(before),
                after_json=pretty(after), observation_json=pretty(turn['observation']),
                simulator_json=pretty(data.get('hidden_user')), corpus_hash=index.digest)


def experiments(request):
    runs, _, issues = artifacts.discover()
    task = request.GET.get('task', '')
    tasks = sorted({r['config']['task'] for r in runs})
    if task:
        runs = [r for r in runs if r['config']['task'] == task]
    return page(request, 'experiments', 'Обучение', 'experiments', tasks=tasks, task=task,
                issues=issues, roots=settings.DIALOGUE_RUN_DIRS, **paginate(request, runs))


def loss_chart(history):
    points = [(r.get('epoch'), r.get(key)) for key in ('loss', 'eval_loss') for r in history]
    points = [(x, y) for x, y in points if isinstance(x, (int, float))
              and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y)]
    if not points:
        return None
    xmax, ymax = max(x for x, _ in points) or 1, max(y for _, y in points) or 1
    series = []
    for key, label, color in (('loss', 'Train loss', '#1971c2'), ('eval_loss', 'Validation loss', '#2b8a3e')):
        dots = [{'x': 48 + 660 * r['epoch'] / xmax, 'y': 205 - 180 * r[key] / ymax,
                 'epoch': r['epoch'], 'value': r[key]}
                for r in history if (r.get('epoch'), r.get(key)) in points]
        if dots:
            series.append({'label': label, 'color': color, 'dots': dots,
                           'points': ' '.join(f'{d["x"]:.2f},{d["y"]:.2f}' for d in dots)})
    return {'series': series, 'xmax': xmax, 'ymax': ymax}


@artifact_errors
def experiment(request, identifier):
    run = find_artifact(identifier, 'experiment')
    metrics_path = run['path'] / 'metrics.json'
    metrics = artifacts.read_object(metrics_path) if metrics_path.is_file() else {}
    checkpoints, history, issues = [], [], []
    history_step = -1
    for directory in sorted(run['path'].glob('checkpoint-*')):
        if not directory.is_dir() or directory.is_symlink():
            continue
        state = {}
        state_path = directory / 'trainer_state.json'
        if state_path.is_file():
            try:
                state = artifacts.read_object(state_path)
            except artifacts.ArtifactError as error:
                issues.append(str(error))
        best = str(metrics.get('best_checkpoint') or '<none>').replace('\\', '/')
        checkpoints.append({'name': directory.name, 'path': directory, 'epoch': state.get('epoch'),
                            'step': state.get('global_step'),
                            'best': str(directory).replace('\\', '/').endswith(best)})
        if state.get('global_step', -1) > history_step:
            history, history_step = state.get('log_history', []), state['global_step']
    checkpoints.sort(key=lambda c: c['step'] if isinstance(c['step'], int) else -1)
    splits, data_issues = artifacts.dataset_summary()
    matches = [{'split': s['split'], 'matches': run['config'].get(s['split'] + '_hash') == s['hash']}
               for s in splits if s['split'] in ('train', 'validation')]
    _, reports, discovery_issues = artifacts.discover()
    related = []
    for report in reports:
        for model in report['config'].get('models', {}).values():
            path = Path(model.get('path', ''))
            path = path if path.is_absolute() else settings.PROJECT_ROOT / path
            if path.resolve() == run['path'].resolve() or path.resolve().parent == run['path'].resolve():
                related.append(report)
                break
    return page(request, 'experiment', run['name'], 'experiments', run=run, metrics=metrics,
                model_id=artifacts.artifact_id(run['path']),
                config_json=pretty(run['config']), metrics_json=pretty(metrics), checkpoints=checkpoints,
                history=history, chart=loss_chart(history), matches=matches, reports=related,
                issues=issues + data_issues + discovery_issues)


def reports(request):
    _, runs, issues = artifacts.discover()
    filters = []
    for key, (label, choices) in artifacts.EVALUATION_FIELDS.items():
        selected = request.GET.get(key, '')
        filters.append({'name': key, 'label': label, 'choices': choices.items(), 'selected': selected})
        if selected:
            runs = [r for r in runs if r['evaluation'][key] == selected]
    return page(request, 'reports', 'Бенчмарки', 'reports', filters=filters,
                roots=settings.DIALOGUE_RUN_DIRS, issues=issues, **paginate(request, runs))


def report(request):
    return redirect('app_main:reports')


@artifact_errors
def report_detail(request, identifier):
    run = find_artifact(identifier, 'report')
    config = run['config']
    prediction_path = artifacts.records_path(run['path'], 'predictions')
    rows = artifacts.record_index(prediction_path, 'predictions').entries if prediction_path else []
    error, tag, query = (request.GET.get(key, '').strip() for key in ('error', 'tag', 'q'))
    metric, value = request.GET.get('metric', ''), request.GET.get('value', '')
    target = request.GET.get('target', '')
    if error:
        rows = [r for r in rows if error in r['errors']]
    if target:
        rows = [r for r in rows if r['tags'].get('target') == target]
    if tag:
        field, _, value = tag.partition('=')
        rows = [r for r in rows if r['tags'].get(field) == value]
    if query:
        rows = [r for r in rows if query.casefold() in r['episode_id'].casefold()]
    if metric:
        rows = [r for r in rows if metric in r['metrics'] and
                (value not in ('0', '1') or r['metrics'][metric] == int(value))]
    selected_group = config.get('groups', {}).get(tag)
    metrics = selected_group['metrics'] if selected_group else config['metrics']
    return page(request, 'report_detail', run['name'], 'reports', run=run,
                metric_rows=artifacts.metric_rows(metrics), selected_group=selected_group,
                error=error, tag=tag, q=query, metric=metric, value=value, target=target,
                has_predictions=prediction_path is not None,
                config_json=pretty({k: v for k, v in config.items() if k not in
                                   ('metrics', 'groups', 'data_review_candidates', 'rollout_outcomes', 'session_outcomes')}),
                **paginate(request, rows))


@artifact_errors
def prediction(request, identifier, position):
    run = find_artifact(identifier, 'report')
    path = artifacts.records_path(run['path'], 'predictions')
    if path is None:
        raise Http404('Предсказания не найдены.')
    index = artifacts.record_index(path, 'predictions')
    if not 0 <= position < len(index.entries):
        raise Http404('Предсказание не найдено.')
    record = index.record(position)
    reference = record.get('reference', {})
    stages = []
    for name, result in record.get('stages', {}).items():
        expected = reference.get({'dst': 'state', 'policy': 'action', 'history': 'response'}.get(name, name))
        if name == 'extraction':
            expected = {key: reference.get(key) for key in ('state', 'observation')}
        stages.append({'name': name, 'result': result, 'input_json': pretty(result.get('input')),
                       'parsed_json': pretty(result.get('parsed')), 'expected_json': pretty(expected),
                       'changes': differences(expected, result['parsed']) if
                       isinstance(expected, dict) and isinstance(result.get('parsed'), dict) else []})
    source_url = None
    if run['evaluation']['user_source'] == 'dataset':
        for split in run['config'].get('splits', []):
            try:
                corpus = artifacts.dataset_index(split)
                if corpus.digest == run['config'].get('corpus_hash') and any(
                        r['episode_id'] == record['episode_id'] for r in corpus.entries):
                    source_url = reverse('app_main:episode', args=[split, record['episode_id']])
                    source_url += f'?turn={record["turn"]}'
            except artifacts.ArtifactError:
                pass
    return page(request, 'prediction', f'{record["episode_id"]} · ход {record["turn"]}', 'reports',
                run=run, record=record, stages=stages, source_url=source_url, position=position,
                previous=position - 1 if position > 0 else None,
                following=position + 1 if position + 1 < len(index.entries) else None,
                raw_json=pretty(record), metric_rows=artifacts.metric_rows({
                    name: {'value': value, 'numerator': value, 'denominator': 1}
                    for name, value in record.get('metrics', {}).items()}))


@artifact_errors
def compare(request):
    identifiers = list(dict.fromkeys(request.GET.getlist('run')))
    if not 2 <= len(identifiers) <= 4:
        return page(request, 'compare', 'Сравнение бенчмарков', 'reports',
                    selection_error='Выберите от двух до четырёх отчётов на странице бенчмарков.')
    runs = [find_artifact(identifier, 'report') for identifier in identifiers]
    names = sorted(set().union(*(run['config']['metrics'] for run in runs)))
    rows = []
    for name in names:
        cells = []
        for run in runs:
            value = run['config']['metrics'].get(name)
            cells.append(artifacts.metric_rows({name: value})[0] if value else None)
        rows.append({'name': name, 'cells': cells, 'violation': name.startswith('behaviour.')})
    return page(request, 'compare', 'Сравнение бенчмарков', 'reports', runs=runs, rows=rows,
                comparison_issues=artifacts.comparison_issues(runs))

