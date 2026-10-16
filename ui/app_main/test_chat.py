import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import uuid

from django.db import IntegrityError, transaction
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from . import artifacts, jobs
from .models import ChatSession, Job


class ChatTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'models' / 'ubar'
        self.model.mkdir(parents=True)
        for name, content in [('experiment.json', {'task': 'ubar'}), ('config.json', {})]:
            (self.model / name).write_text(json.dumps(content), encoding='utf-8')
        (self.model / 'model.safetensors').write_bytes(b'fixture')
        override = override_settings(DIALOGUE_RUN_DIRS=[self.root / 'models'],
                                     DIALOGUE_JOBS_DIR=self.root / 'control', DIALOGUE_OUTPUT_DIR=self.root / 'runs')
        override.enable()
        self.addCleanup(override.disable)
        self.addCleanup(artifacts._json.cache_clear)

    def create_chat(self, message='Hi!'):
        data = {'submission_id': uuid.uuid4(), 'model': artifacts.artifact_id(self.model),
                'device': 'cpu', 'max_new_tokens': 192}
        with patch('app_main.jobs.subprocess.Popen') as launch:
            first = self.client.post(reverse('app_main:chats'), data)
            repeat = self.client.post(reverse('app_main:chats'), data)
        launch.assert_not_called()
        self.assertEqual(first.status_code, 302)
        self.assertEqual(repeat.url, first.url)
        chat = ChatSession.objects.get(pk=data['submission_id'])
        self.assertEqual(chat.jobs.count(), 0)
        if message is not None:
            self.assertEqual(self.send(chat, message=message).status_code, 302)
        return chat

    def complete(self, job, response='Model reply', closed=False):
        job.status = Job.Status.COMPLETED
        job.result = {'status': 'completed', 'response': response, 'message': job.config.get('message'),
                      'state': {'marker': 'predicted'},
                      'action': {'type': 'CLOSE' if closed else 'ACK', 'target': None},
                      'closed': closed, 'block': 'generated-block', 'stages': {}}
        job.save(update_fields=['status', 'result'])

    def send(self, chat, previous=None, message='Hello', identifier=None):
        return self.client.post(reverse('app_main:chat_send', args=[chat.pk]), {
            'submission_id': identifier or uuid.uuid4(), 'previous_job': previous.pk if previous else '', 'message': message})

    def test_creation_waits_for_first_user_message_and_gets_are_read_only(self):
        self.assertEqual(self.client.get(reverse('app_main:chats')).status_code, 200)
        self.assertFalse(ChatSession.objects.exists())
        chat = self.create_chat(message=None)
        page = self.client.get(reverse('app_main:chat', args=[chat.pk]))
        self.assertContains(page, 'name="message"')
        self.assertEqual(chat.jobs.count(), 0)
        self.assertEqual(self.send(chat, message='').status_code, 200)
        self.assertEqual(chat.jobs.count(), 0)
        self.assertEqual(self.send(chat, message='It is evening.').status_code, 302)
        job = chat.jobs.get()
        self.assertEqual(job.kind, 'chat')
        self.assertEqual(job.config['message'], 'It is evening.')
        self.assertEqual(job.config['turns'], [])
        self.assertNotIn('data_hashes', job.config)
        self.assertContains(self.client.get(reverse('app_main:chat', args=[chat.pk])), 'В очереди')
        self.assertEqual(chat.jobs.count(), 1)
        self.assertRedirects(self.client.get(reverse('app_main:job', args=[job.pk])), reverse('app_main:chat', args=[chat.pk]))
        incompatible = self.root / 'models' / 'dst'
        incompatible.mkdir()
        for name, value in [('experiment.json', {'task': 'dst'}), ('config.json', {})]:
            (incompatible / name).write_text(json.dumps(value))
        (incompatible / 'model.safetensors').write_bytes(b'fixture')
        response = self.client.post(reverse('app_main:chats'), {'submission_id': uuid.uuid4(),
                                    'model': artifacts.artifact_id(incompatible), 'device': 'cpu', 'max_new_tokens': 192})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ChatSession.objects.count(), 1)

    def test_send_serializes_own_history_and_rejects_duplicate_pending_or_stale_turns(self):
        chat = self.create_chat()
        first = chat.jobs.get()
        self.complete(first)
        identifier = uuid.uuid4()
        self.assertEqual(self.send(chat, first, identifier=identifier).status_code, 302)
        self.assertEqual(self.send(chat, first, identifier=identifier).status_code, 302)
        second = chat.jobs.get(pk=identifier)
        self.assertEqual(second.config['turns'], [first.result])
        page = self.client.get(reverse('app_main:chat', args=[chat.pk]))
        self.assertContains(page, 'B · предсказанное состояние')
        self.assertContains(page, 'A · предсказанное действие')
        self.assertNotContains(page, 'Память после действия')
        self.assertEqual(second.config['message'], 'Hello')
        self.assertEqual(self.send(chat, first).status_code, 200)
        self.assertEqual(chat.jobs.count(), 2)
        self.complete(second)
        self.assertEqual(self.send(chat, first).status_code, 200)
        self.assertEqual(chat.jobs.count(), 2)
        self.assertEqual(self.send(chat, second, 'Another message').status_code, 302)
        self.assertEqual(chat.jobs.count(), 3)

    def test_failures_do_not_advance_history_and_model_output_is_escaped(self):
        chat = self.create_chat()
        opening = chat.jobs.get()
        self.complete(opening, '<script>alert(1)</script>')
        self.send(chat, opening, '<img src=x onerror=alert(1)>')
        failed = chat.jobs.exclude(pk=opening.pk).get()
        failed.status = Job.Status.FAILED
        failed.error = 'invalid state'
        failed.result = {'status': 'failed', 'stages': {'dst': {'text': '<script>bad()</script>', 'tokens': 3, 'stopped': False}}}
        failed.save()
        response = self.client.get(reverse('app_main:chat', args=[chat.pk]))
        self.assertContains(response, '&lt;script&gt;alert(1)&lt;/script&gt;')
        self.assertContains(response, '&lt;script&gt;bad()&lt;/script&gt;')
        self.assertNotContains(response, '<script>alert(1)</script>')
        self.send(chat, opening, 'Try again')
        pending = chat.jobs.get(status='queued')
        self.assertEqual(pending.config['turns'], [opening.result])

    def test_close_stops_conversation_and_new_chat_has_independent_memory(self):
        chat = self.create_chat()
        first = chat.jobs.get()
        self.complete(first, closed=True)
        self.assertEqual(self.send(chat, first).status_code, 200)
        self.assertEqual(chat.jobs.count(), 1)
        page = self.client.get(reverse('app_main:chat', args=[chat.pk]))
        self.assertContains(page, 'Модель выбрала CLOSE')
        self.assertNotContains(page, 'id="chat-message-form"')
        other = self.create_chat()
        self.assertEqual(other.jobs.get().config['turns'], [])

    def test_stop_requires_post_csrf_and_opening_can_be_retried(self):
        chat = self.create_chat()
        url = reverse('app_main:chat_stop', args=[chat.pk])
        self.assertEqual(self.client.get(url).status_code, 405)
        self.assertEqual(Client(enforce_csrf_checks=True).post(url).status_code, 403)
        self.assertEqual(self.client.post(url).status_code, 302)
        self.assertEqual(chat.jobs.get().status, 'cancelled')
        self.assertEqual(self.send(chat, message='').status_code, 200)
        self.assertFalse(chat.jobs.filter(status='queued').exists())
        self.assertEqual(self.send(chat, message='Hi!').status_code, 302)
        self.assertEqual(chat.jobs.filter(status='queued').count(), 1)
        status = self.client.get(reverse('app_main:chat_status', args=[chat.pk]))
        self.assertTrue(status.json()['active'])
        self.assertEqual(status['Cache-Control'], 'no-store')

    def test_database_prevents_two_active_jobs_for_one_chat(self):
        chat = self.create_chat()
        with self.assertRaises(IntegrityError), transaction.atomic():
            Job.objects.create(kind='chat', name='duplicate', config={}, output='unused', chat_session=chat)
