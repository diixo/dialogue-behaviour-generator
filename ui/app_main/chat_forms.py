"""Chat forms only inspect local checkpoint metadata."""

import uuid

from django import forms
from dialogue_protocol import UBAR_CONTEXT

from . import jobs


class ChatForm(forms.Form):
    submission_id = forms.UUIDField(initial=uuid.uuid4, widget=forms.HiddenInput)
    model = forms.ChoiceField(label='Обученная модель')
    device = forms.ChoiceField(label='Устройство', choices=[('cpu', 'CPU'), ('cuda', 'CUDA GPU')], initial='cpu')
    max_new_tokens = forms.IntegerField(label='Лимит генерации', min_value=1, max_value=1024, initial=192)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog = {key: item for key, item in jobs.checkpoints().items()
                        if item['task'] in ('ubar', 'history')}
        self.fields['model'].choices = [(key, item['label']) for key, item in self.catalog.items()]
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-select' if isinstance(field.widget, forms.Select) else 'form-control'

    def chat_config(self):
        item = self.catalog[self.cleaned_data['model']]
        return {'model': str(item['path']), 'model_label': item['label'], 'task': item['task'],
                'trained_dialogue_start': item['run']['config'].get('dialogue_start', 'assistant'),
                'device': self.cleaned_data['device'], 'max_new_tokens': self.cleaned_data['max_new_tokens']}

    def clean_model(self):
        key = self.cleaned_data['model']
        item = self.catalog[key]
        if item['task'] == 'ubar' and item['run']['config'].get('ubar_context') != UBAR_CONTEXT:
            raise forms.ValidationError(
                'Эта UBAR-модель обучена на прежнем формате контекста. '
                'Обучите новый эксперимент UBAR с памятью memory_v1.')
        return key


class MessageForm(forms.Form):
    submission_id = forms.UUIDField(initial=uuid.uuid4, widget=forms.HiddenInput)
    previous_job = forms.UUIDField(required=False, widget=forms.HiddenInput)
    message = forms.CharField(label='Ваше сообщение', max_length=2000,
                              widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control',
                                                           'placeholder': 'Введите сообщение…'}))
