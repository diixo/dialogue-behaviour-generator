from django.db import models

import uuid


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
    kind = models.CharField(max_length=16, choices=[('train', 'Обучение'), ('benchmark', 'Бенчмарк')])
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

    @property
    def active(self):
        return self.status in (self.Status.QUEUED, self.Status.RUNNING, self.Status.STOPPING)


class Worker(models.Model):
    id = models.PositiveSmallIntegerField(primary_key=True, default=1, editable=False)
    heartbeat = models.DateTimeField()
