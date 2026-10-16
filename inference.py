"""Shared model inference and strict output parsers for training experiments."""

from dataclasses import dataclass
import json
from pathlib import Path

from dialogue_protocol import (SEGMENTS, UBAR_CONTEXT, encode, task_messages, ubar_prompt,
                               parse_state, parse_extraction, validate_state)
from dialogue_format import get_format


@dataclass
class Request:
    task: str
    history: list
    state: dict | None = None
    action: dict | None = None
    # UBAR memory is B before the last assistant action, plus that action itself.
    previous_state: dict | None = None
    previous_action: dict | None = None
    # Only generated segments of the current turn; no previous user/assistant text.
    prefix: str = ''


@dataclass
class Prediction:
    text: str
    stopped: bool = True
    tokens: int = 0
    error: str | None = None


class HFPredictor:
    def __init__(self, model_path, *, device='cpu', local_files_only=False, max_new_tokens=384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.path = str(model_path)
        path = Path(model_path)
        manifest = path / 'experiment.json'
        if not manifest.exists() and path.name.startswith('checkpoint-'):
            manifest = path.parent / 'experiment.json'
        self.config = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
        self.task = self.config.get('task')
        if self.task == 'ubar' and self.config.get('ubar_context') != UBAR_CONTEXT:
            raise ValueError(
                'UBAR checkpoint uses an incompatible context format. '
                'Train a new --task ubar experiment with memory_v1; old session checkpoints require retraining.')
        self.style = self.config.get('style', 'plain')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_length = min(self.config.get('max_length', 1024),
                              getattr(self.model.config, 'max_position_embeddings', 1024))
        if self.tokenizer.eos_token_id is None or max_new_tokens <= 0:
            raise ValueError('Inference requires EOS and a positive generation budget')

    def __call__(self, request):
        actual_task = request.task
        if request.task == 'dst' and self.task == 'extraction':
            actual_task = 'extraction'
        if self.task not in (None, 'ubar', actual_task):
            raise ValueError(f'Checkpoint trained for {self.task}, requested {actual_task}')
        budget = min(self.max_new_tokens, {'policy': 32, 'response': 128, 'history': 128}.get(
            actual_task, self.max_new_tokens))
        if self.task == 'ubar':
            if actual_task not in SEGMENTS:
                raise ValueError(f'UBAR has no {actual_task} segment')
            if (len(request.history) != 1 or request.history[0].get('role') != 'user' or
                    request.previous_state is None):
                raise ValueError('UBAR requires previous_state and only the current user message')
            start, end = SEGMENTS[actual_task]
            stop_ids = encode(self.tokenizer, end)
            if len(stop_ids) != 1:
                raise ValueError('UBAR checkpoint is missing segment tokens')
            stop_id = stop_ids[0]
            prompt = ubar_prompt(request.history[0]['content'], request.previous_state,
                                 request.previous_action, self.style) + request.prefix + start
        else:
            history = list(request.history)
            fmt = get_format(self.style)
            def render():
                return fmt.render(task_messages(actual_task, history, request.state, request.action), '')['prompt']
            prompt = render()
            while len(encode(self.tokenizer, prompt)) + budget > self.max_length and len(history) > 2:
                history = history[2:]
                prompt = render()
            stop_id = self.tokenizer.eos_token_id
        ids = encode(self.tokenizer, prompt)
        if len(ids) + budget > self.max_length:
            raise ValueError('Context and requested generation budget exceed model context')
        inputs = self.torch.tensor([ids], device=self.device)
        with self.torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=self.torch.ones_like(inputs),
                do_sample=False, max_new_tokens=budget, eos_token_id=stop_id,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id)
        generated = output[0, len(ids):].tolist()
        stopped = bool(generated and generated[-1] == stop_id)
        text = self.tokenizer.decode(generated[:-1] if stopped else generated, skip_special_tokens=False)
        suffix = get_format(self.style).completion_suffix
        if self.task != 'ubar' and suffix and text.endswith(suffix):
            text = text[:-len(suffix)]
        if request.task == 'dst' and actual_task == 'extraction' and stopped:
            try:
                text = json.dumps(parse_extraction(text)['state'], ensure_ascii=False)
            except (ValueError, TypeError, KeyError) as error:
                # Preserve malformed raw output for the benchmark's error report.
                return Prediction(text.strip(), stopped, len(generated), f'Invalid extraction: {error}')
        return Prediction(text.strip(), stopped, len(generated))


class PredictorRouter:
    def __init__(self, default=None, *, dst=None, policy=None, response=None, **kwargs):
        paths = {'dst': dst or default, 'extraction': dst or default,
                 'policy': policy or default, 'response': response or default, 'history': default}
        cache = {}
        self.models = {}
        for task, path in paths.items():
            if path is not None:
                if path not in cache:
                    cache[path] = HFPredictor(path, **kwargs)
                self.models[task] = cache[path]
        styles = {model.style for model in cache.values() if model.task == 'ubar'}
        if len(styles) > 1:
            raise ValueError('UBAR stages must use the same style')
        self.style = next(iter(styles), 'plain')
        self.metadata = {task: {'path': model.path, 'experiment': model.config,
                               'generation': {'decoding': 'greedy', 'device': model.device,
                                              'max_new_tokens': model.max_new_tokens,
                                              'policy_cap': 32, 'response_cap': 128,
                                              'context_limit': model.max_length}}
                         for task, model in self.models.items()}

    def validate_chain(self, chain):
        from experiment_config import required_stages
        for task in required_stages(chain):
            if task not in self.models:
                raise ValueError(f'No checkpoint configured for {task}')
            trained = self.models[task].task
            compatible = trained in (None, task) or (task == 'dst' and trained == 'extraction') or (
                trained == 'ubar' and task in SEGMENTS)
            if not compatible:
                raise ValueError(f'Checkpoint trained for {trained}, requested {task}')

    def __call__(self, request):
        if request.task not in self.models:
            raise ValueError(f'No checkpoint configured for {request.task}')
        return self.models[request.task](request)
