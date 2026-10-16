"""Small, framework-independent control channel for background model processes."""

import json
import math
import os
from pathlib import Path
import time


class LockBusy(RuntimeError):
    pass


class FileLock:
    """OS releases this exclusive lock even if its process crashes."""
    def __init__(self, path):
        self.path, self.file = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        if self.file.tell() == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            raise LockBusy(str(self.path)) from error
        return self

    def __exit__(self, *args):
        self.file.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


def lock_available(path):
    try:
        with FileLock(path):
            return True
    except LockBusy:
        return False


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: finite(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(value) for value in item]
        return item
    temporary.write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, default=str,
                                    allow_nan=False) + '\n', encoding='utf-8')
    # A reader can briefly hold a file open on Windows.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.03)


class JobCancelled(Exception):
    pass


class JobControl:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.progress = {}
        self.stopped = False

    def stop_requested(self):
        return (self.directory / 'stop.request').exists()

    def check_stop(self):
        if self.stop_requested():
            raise JobCancelled('Остановлено по запросу пользователя.')

    def emit(self, **values):
        self.progress.update(values, updated_at=time.time())
        atomic_json(self.directory / 'progress.json', self.progress)

    def benchmark_progress(self, **values):
        self.check_stop()
        total = values['total_episodes']
        self.emit(phase='benchmark', percent=100 * values['completed_episodes'] / total, **values)


def training_callback(control):
    # Transformers stays out of the Django web process.
    from transformers import TrainerCallback

    class Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            job.emit(phase='training', epochs=args.num_train_epochs, epoch=state.epoch or 0,
                     step=state.global_step, percent=100 * (state.epoch or 0) / args.num_train_epochs)

        def on_step_end(self, args, state, control, **kwargs):
            # The HF argument named control is the TrainerControl, not JobControl.
            job.emit(phase='training', epoch=state.epoch or 0, step=state.global_step,
                     epochs=args.num_train_epochs,
                     percent=min(100, 100 * (state.epoch or 0) / args.num_train_epochs))
            if job.stop_requested() and (state.epoch or 0) < args.num_train_epochs:
                job.stopped = True
                # The Trainer closes the interrupted epoch next; evaluate/save once there.
                control.should_save = False
                control.should_evaluate = False
                control.should_training_stop = True
                job.emit(phase='saving')
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if job.stopped:
                control.should_evaluate = True
                control.should_save = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                job.emit(**{key: value for key, value in logs.items()
                            if key in ('loss', 'eval_loss', 'learning_rate', 'grad_norm')})

        def on_save(self, args, state, control, **kwargs):
            job.emit(checkpoint=str(Path(args.output_dir) / f'checkpoint-{state.global_step}'))

    job = control
    return Callback()
