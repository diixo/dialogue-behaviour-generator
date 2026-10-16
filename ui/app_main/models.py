from django.db import models

import uuid


class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    config = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


class Job(models.Model):
    class Status(models.TextChoices):
        QUEUED = 'queued', 'В очереди'
        RUNNING = 'running', 'Выполняется'
        STOPPING = 'stopping', 'Останавливается'
        STOPPED = 'stopped', 'Остановлено с checkpoint'
        COMPLETED = 'completed', 'Завершено'
        CANCELLED = 'cancelled', 'Отменено'
        FAILED = 'failed', 'Ошибка'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    kind = models.CharField(max_length=16, choices=[('train', 'Обучение'), ('benchmark', 'Бенчмарк'), ('chat', 'Чат')])
    chat_session = models.ForeignKey(ChatSession, null=True, blank=True, on_delete=models.CASCADE, related_name='jobs')
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED, db_index=True)
    config = models.JSONField()
    output = models.TextField()
    progress = models.JSONField(default=dict)
    result = models.JSONField(default=dict)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True)
    finished_at = models.DateTimeField(null=True)

    class Meta:
        ordering = ['-created_at']
        constraints = [models.UniqueConstraint(fields=['chat_session'],
                       condition=models.Q(status__in=['queued', 'running', 'stopping']),
                       name='one_active_job_per_chat')]

    @property
    def active(self):
        return self.status in (self.Status.QUEUED, self.Status.RUNNING, self.Status.STOPPING)


class Worker(models.Model):
    id = models.PositiveSmallIntegerField(primary_key=True, default=1, editable=False)
    heartbeat = models.DateTimeField()
