from django.core.management.base import BaseCommand, CommandError

from app_main.jobs import run_worker
from job_runtime import LockBusy


class Command(BaseCommand):
    help = 'Run the local training/benchmark queue, one model process at a time.'

    def add_arguments(self, parser):
        parser.add_argument('--once', action='store_true', help='Process at most one queued job and exit')

    def handle(self, *args, **options):
        self.stdout.write('Worker ready. Ctrl+C requests a safe stop of the current job.')
        try:
            run_worker(once=options['once'])
        except LockBusy as error:
            raise CommandError('Другой worker уже работает с этой очередью.') from error
        except KeyboardInterrupt:
            self.stdout.write('Worker stopped.')
