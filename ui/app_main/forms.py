"""Validate jobs without loading tokenizers or models into Django."""

import hashlib
import json
import uuid

from django import forms

from experiment_config import TRAIN_DEFAULTS, TRAIN_TASKS, compatible_task, required_stages, validate_evaluation
from . import artifacts, jobs


class StyledForm(forms.Form):
    submission_id = forms.UUIDField(initial=uuid.uuid4, widget=forms.HiddenInput)
    name = forms.CharField(label='Название', max_length=100, required=False)
    local_files_only = forms.BooleanField(label='Использовать только локальные файлы модели', required=False)
    limit_episodes = forms.IntegerField(label='Ограничить число эпизодов в каждом split', required=False, min_value=1, initial=5,
                                       help_text='Пусто — весь датасет. Для короткой проверки можно указать несколько эпизодов.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = ('form-check-input' if isinstance(field.widget, forms.CheckboxInput)
                                           else 'form-select' if isinstance(field.widget, forms.Select) else 'form-control')


class TrainForm(StyledForm):
    model = forms.CharField(label='Базовая модель', max_length=200, initial=TRAIN_DEFAULTS['model'])
    task = forms.ChoiceField(label='Задача', choices=[(x, x) for x in TRAIN_TASKS], initial='ubar')
    epochs = forms.FloatField(label='Общее число эпох', min_value=.01, initial=TRAIN_DEFAULTS['epochs'])
    batch_size = forms.IntegerField(label='Batch size', min_value=1, initial=TRAIN_DEFAULTS['batch_size'])
    accumulation = forms.IntegerField(label='Gradient accumulation', min_value=1, initial=TRAIN_DEFAULTS['accumulation'])
    learning_rate = forms.FloatField(label='Learning rate', min_value=1e-9, max_value=1, initial=TRAIN_DEFAULTS['learning_rate'])
    warmup_ratio = forms.FloatField(label='Warmup ratio', min_value=0, max_value=1, initial=TRAIN_DEFAULTS['warmup_ratio'])
    max_length = forms.IntegerField(label='Контекст (токены)', min_value=32, initial=TRAIN_DEFAULTS['max_length'])
    style = forms.ChoiceField(label='Формат', choices=[('plain', 'Plain'), ('chatml', 'ChatML')], initial='plain')
    loss_scope = forms.ChoiceField(label='Loss', choices=[('targets', 'Целевые сегменты'), ('all', 'Все токены UBAR')], initial='targets')
    seed = forms.IntegerField(label='Seed', min_value=0, max_value=4294967295, initial=TRAIN_DEFAULTS['seed'])
    cpu = forms.BooleanField(label='Обучать на CPU', required=False)

    def clean(self):
        cleaned = super().clean()
        if cleaned.get('loss_scope') == 'all' and cleaned.get('task') != 'ubar':
            self.add_error('loss_scope', 'Все токены поддерживаются только для UBAR.')
        if cleaned.get('model') == 'gpt2' and cleaned.get('max_length', 0) > 1024:
            self.add_error('max_length', 'У классической GPT-2 контекст ограничен 1024 токенами.')
        return cleaned

    def job_config(self):
        return {key: value for key, value in self.cleaned_data.items() if key not in ('name', 'submission_id')} | jobs.data_config(('train', 'validation'))


class BenchmarkForm(StyledForm):
    chain = forms.ChoiceField(label='Цепочка', choices=list(artifacts.EVALUATION_FIELDS['chain'][1].items()), initial='full',
                              help_text='Какие этапы предсказывает модель на каждом ходе.')
    context = forms.ChoiceField(label='Контекст', choices=list(artifacts.EVALUATION_FIELDS['context'][1].items()), initial='reference',
                                help_text='Собственная предыстория доступна для полной цепочки; ошибки накапливаются между ходами.')
    user_source = forms.ChoiceField(label='Сообщения пользователя', choices=list(artifacts.EVALUATION_FIELDS['user_source'][1].items()), initial='dataset',
                                    help_text='Симулятор требует собственную предысторию модели и реагирует на её действия.')
    split = forms.ChoiceField(label='Split', choices=[('validation', 'validation'), ('test', 'test')], initial='validation')
    model = forms.ChoiceField(label='Общая модель для этапов', required=False)
    dst = forms.ChoiceField(label='Модель состояния B (DST / extraction)', required=False)
    policy = forms.ChoiceField(label='Модель действия A', required=False)
    response = forms.ChoiceField(label='Модель ответа R', required=False)
    device = forms.ChoiceField(label='Устройство', choices=[('cpu', 'CPU'), ('cuda', 'CUDA GPU')], initial='cpu')
    max_new_tokens = forms.IntegerField(label='Лимит генерации', min_value=1, max_value=1024, initial=192)
    max_turns = forms.IntegerField(label='Максимум ходов с симулятором', min_value=1, initial=17,
                                  help_text='Только для симулятора. Для сообщений из датасета проверяются все записанные ходы.')
    seed = forms.IntegerField(label='Seed', min_value=0, max_value=4294967295, initial=42)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog = jobs.checkpoints()
        for name in ('model', 'dst', 'policy', 'response'):
            self.fields[name].choices = [('', '— не задана —')] + [(key, item['label']) for key, item in self.catalog.items()]

    def clean(self):
        cleaned = super().clean()
        if any(key not in cleaned for key in ('chain', 'context', 'user_source')):
            return cleaned
        try:
            validate_evaluation(cleaned['chain'], cleaned['context'], cleaned['user_source'])
        except ValueError:
            self.add_error(None, 'Собственная предыстория требует полной цепочки DST → policy → response. '
                                 'Сообщения симулятора доступны только с собственной предысторией.')
            return cleaned
        styles = set()
        for stage in required_stages(cleaned['chain']):
            field = 'dst' if stage == 'extraction' else stage
            key = cleaned.get(field) if field != 'history' else None
            item = self.catalog.get(key or cleaned.get('model'))
            if item is None:
                self.add_error(None, f'Выберите модель для этапа {stage}.')
            elif not compatible_task(item['task'], stage):
                self.add_error(None, f'{item["task"]} не поддерживает этап {stage}.')
            elif item['task'] == 'ubar':
                styles.add(item['run']['config'].get('style', 'plain'))
        if len(styles) > 1:
            self.add_error(None, 'Этапы UBAR должны использовать одинаковый формат.')
        return cleaned

    def job_config(self):
        config = {key: value for key, value in self.cleaned_data.items() if key not in ('name', 'submission_id')}
        for name in ('model', 'dst', 'policy', 'response'):
            config[name] = str(self.catalog[config[name]]['path']) if config[name] else None
        return config | jobs.data_config((config['split'],))


class ResumeForm(StyledForm):
    checkpoint = forms.ChoiceField(label='Checkpoint с состоянием оптимизатора')
    cpu = forms.BooleanField(label='Продолжить на CPU', required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog = jobs.checkpoints(resumable=True)
        self.fields['checkpoint'].choices = [(key, item['label']) for key, item in self.catalog.items()]

    def clean(self):
        cleaned = super().clean()
        item = self.catalog.get(cleaned.get('checkpoint'))
        if item is None:
            return cleaned
        config = item['run']['config']
        try:
            state = artifacts.read_object(item['path'] / 'trainer_state.json')
            if (state.get('epoch') or 0) >= config['epochs']:
                self.add_error('checkpoint', 'Этот checkpoint уже достиг заданного числа эпох.')
            for split in ('train', 'validation'):
                index = artifacts.dataset_index(split)
                limit = cleaned.get('limit_episodes')
                digest = index.digest
                if limit:
                    rows = [index.record(i) for i in range(min(limit, len(index.entries)))]
                    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if digest != config.get(split + '_hash'):
                    self.add_error(None, f'{split}: данные не совпадают с исходным обучением. Проверьте ограничение числа эпизодов.')
        except (artifacts.ArtifactError, KeyError) as error:
            self.add_error(None, str(error))
        return cleaned

    def job_config(self):
        item = self.catalog[self.cleaned_data['checkpoint']]
        manifest = item['run']['config']
        config = {key: manifest[key] for key in TRAIN_DEFAULTS if key not in ('model', 'accumulation')}
        config.update(model=manifest['base_model'], accumulation=manifest['gradient_accumulation_steps'],
                      resume=str(item['path']), cpu=self.cleaned_data['cpu'],
                      local_files_only=self.cleaned_data['local_files_only'],
                      limit_episodes=self.cleaned_data['limit_episodes'])
        return config | jobs.data_config(('train', 'validation'))

