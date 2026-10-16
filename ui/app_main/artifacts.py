"""Read existing experiment artifacts; never load a model or rewrite a dataset."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from django.conf import settings

from experiment_config import evaluation_options

SPLITS = ('train', 'validation', 'test')
EVALUATION_FIELDS = {
    'chain': ('Цепочка', {'dst': 'DST — состояние', 'extraction': 'Extraction — наблюдение и состояние',
                         'policy': 'Policy — действие', 'response': 'Response — ответ',
                         'history': 'History — прямой ответ', 'policy_response': 'Policy → response',
                         'full': 'Полная: DST → policy → response'}),
    'context': ('Контекст', {'reference': 'Эталонная предыстория', 'predicted': 'Собственная предыстория модели'}),
    'user_source': ('Сообщения пользователя', {'dataset': 'Из датасета', 'simulator': 'От симулятора'}),
}


def evaluation_details(config):
    options = evaluation_options(config)
    labels = {key + '_label': choices[options[key]] for key, (_, choices) in EVALUATION_FIELDS.items()}
    return options | labels | {'label': ' · '.join(labels.values())}


class ArtifactError(ValueError):
    pass


def signature(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=48)
def _json(path, stamp):
    with Path(path).open(encoding='utf-8') as source:
        return json.load(source)


def read_json(path):
    try:
        return _json(str(path), signature(path))
    except (OSError, ValueError) as error:
        raise ArtifactError(f'{path.name}: {error}') from error


def read_object(path):
    value = read_json(path)
    if not isinstance(value, dict):
        raise ArtifactError(f'{path.name}: expected a JSON object')
    return value


@dataclass
class RecordIndex:
    path: Path
    stamp: tuple
    entries: list
    digest: str

    def record(self, position):
        try:
            if signature(self.path) != self.stamp:
                raise ArtifactError('Файл изменился. Обновите страницу.')
            if self.path.suffix == '.json':
                return read_json(self.path)[position]
            with self.path.open('rb') as source:
                source.seek(self.entries[position]['offset'])
                return json.loads(source.readline())
        except (OSError, ValueError) as error:
            raise ArtifactError(f'{self.path.name}: {error}') from error


def _records(path):
    if path.suffix == '.json':
        values = read_json(path)
        if not isinstance(values, list):
            raise ArtifactError(f'{path.name}: expected a JSON array')
        yield from enumerate(values)
    else:
        with path.open('rb') as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if line.strip():
                    yield offset, json.loads(line)


@lru_cache(maxsize=12)
def _index(path_string, stamp, kind):
    path = Path(path_string)
    entries, seen = [], set()
    digest = hashlib.sha256()
    digest.update(b'[')
    try:
        for offset, row in _records(path):
            if not isinstance(row, dict):
                raise ValueError('Expected an object for each record')
            entry = {'position': len(entries), 'offset': offset, 'episode_id': row['episode_id']}
            if kind == 'episodes':
                if row['episode_id'] in seen:
                    raise ValueError(f'Duplicate episode_id: {row["episode_id"]}')
                seen.add(row['episode_id'])
                turns = row['turns']
                if not turns or [turn['index'] for turn in turns] != list(range(len(turns))):
                    raise ValueError(f'Non-contiguous turns: {row["episode_id"]}')
                if row['split'] != path.name.split('.')[0]:
                    raise ValueError(f'Wrong split: {row["episode_id"]}')
                entry.update(
                    turns=len(turns), extraction=sum(t['user'] is not None for t in turns),
                    persona=row.get('persona', ''), scenario=row.get('scenario', ''),
                    terminal=row.get('terminal_reason', ''), preview=row['messages'][0]['content'],
                    reactions=[t['observation']['reaction'] for t in turns if t.get('observation')],
                    actions=[t['action']['type'] for t in turns],
                    targets=[t['action'].get('target') or 'none' for t in turns],
                )
                # Same canonical JSON representation as data.corpus_hash, streamed per episode.
                if entries:
                    digest.update(b', ')
                digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            else:
                entry.update(turn=row['turn'], tags=row.get('tags', {}), errors=row.get('errors', []),
                             metrics=row.get('metrics', {}))
            entries.append(entry)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ArtifactError(f'{path.name}, record {len(entries) + 1}: {error}') from error
    digest.update(b']')
    if signature(path) != stamp:
        raise ArtifactError(f'{path.name}: файл изменился во время чтения; обновите страницу.')
    return RecordIndex(path, stamp, entries, digest.hexdigest() if kind == 'episodes' else '')


def record_index(path, kind):
    try:
        return _index(str(path), signature(path), kind)
    except OSError as error:
        raise ArtifactError(f'{path.name}: {error}') from error


def records_path(directory, stem):
    for extension in ('.jsonl', '.json'):
        path = Path(directory) / (stem + extension)
        if path.is_file():
            return path
    return None


def dataset_index(split):
    if split not in SPLITS:
        raise ArtifactError('Неизвестное разбиение датасета.')
    path = records_path(settings.DIALOGUE_DATA_DIR, f'{split}.episodes')
    if path is None:
        raise ArtifactError(f'Не найден {split}.episodes.jsonl или .json в {settings.DIALOGUE_DATA_DIR}')
    return record_index(path, 'episodes')


def dataset_summary():
    summaries, issues = [], []
    for split in SPLITS:
        try:
            index = dataset_index(split)
            turns = sum(row['turns'] for row in index.entries)
            summaries.append({'split': split, 'episodes': len(index.entries), 'turns': turns,
                              'extraction': sum(row['extraction'] for row in index.entries),
                              'hash': index.digest, 'path': index.path,
                              'mean_turns': turns / len(index.entries) if index.entries else 0})
        except ArtifactError as error:
            issues.append(str(error))
    return summaries, issues


def distributions(index):
    return {name: Counter(value for row in index.entries for value in row[name]).most_common()
            for name in ('reactions', 'actions', 'targets')}


def artifact_id(path):
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20]


def discover():
    """Only expose artifacts inside configured roots, never a path from the request."""
    experiments, reports, issues, seen = [], [], [], set()
    for configured in settings.DIALOGUE_RUN_DIRS:
        root = Path(configured).resolve()
        if not root.is_dir():
            continue
        for directory, subdirs, filenames in os.walk(root, followlinks=False):
            subdirs[:] = sorted(name for name in subdirs if not name.startswith(('.', 'checkpoint-'))
                               and not (Path(directory) / name).is_symlink())
            for filename, collection in (('experiment.json', experiments), ('report.json', reports)):
                if filename not in filenames:
                    continue
                path = Path(directory) / filename
                if path.is_symlink() or path.resolve() in seen or not path.resolve().is_relative_to(root):
                    continue
                seen.add(path.resolve())
                try:
                    value = read_object(path)
                    required = ('task',) if filename == 'experiment.json' else ('metrics',)
                    if any(key not in value for key in required):
                        raise ArtifactError(f'{filename}: missing {", ".join(required)}')
                    evaluation = evaluation_details(value) if filename == 'report.json' else None
                    collection.append({'id': artifact_id(path), 'name': str(Path(directory).relative_to(root.parent)),
                                       'path': Path(directory), 'config': value,
                                       'evaluation': evaluation,
                                       'modified': path.stat().st_mtime})
                except (ValueError, OSError) as error:
                    issues.append(f'{path.parent.name}: {error}')
    for collection in (experiments, reports):
        collection.sort(key=lambda item: (-item['modified'], item['name']))
    return experiments, reports, issues


def metric_rows(metrics):
    rows = []
    for name, item in sorted(metrics.items()):
        if not isinstance(item, dict):
            continue
        value = item.get('value')
        rows.append({'name': name, 'percent': value * 100 if isinstance(value, (int, float)) else None,
                     'numerator': item.get('numerator'), 'denominator': item.get('denominator'),
                     'violation': name.startswith('behaviour.')})
    return rows


def comparison_issues(reports):
    """Check evaluation conditions independently of model identity."""
    issues = []
    configs = [report['config'] | evaluation_options(report['config']) for report in reports]
    fields = ('chain', 'context', 'user_source', 'splits', 'corpus_hash', 'episodes', 'seed', 'initial_state')
    if any(config['user_source'] == 'simulator' for config in configs):
        fields += ('max_turns',)
    for field in fields:
        values = [config.get(field) for config in configs]
        label = EVALUATION_FIELDS[field][0] if field in EVALUATION_FIELDS else field
        if any(value is None for value in values):
            issues.append(f'{label}: условия не записаны во всех отчётах')
        elif any(value != values[0] for value in values[1:]):
            issues.append(f'{label}: разные условия оценки')
    protocols = []
    for report in reports:
        models = report['config'].get('models', {})
        if not models or any(not value.get('generation') for value in models.values()):
            issues.append(f'{report["name"]}: отсутствуют параметры генерации')
        if models and any(not value.get('experiment', {}).get('style') for value in models.values()):
            issues.append(f'{report["name"]}: формат контекста не записан')
        protocols.append({stage: {'generation': value.get('generation'),
                                  'style': value.get('experiment', {}).get('style')}
                          for stage, value in models.items()})
    if any(value != protocols[0] for value in protocols[1:]):
        issues.append('Разные параметры генерации или формат контекста')
    return issues
