"""Data loading, task/session datasets, tokenization and batch collation.

Chat formatting is selected at runtime. UBAR uses one memory snapshot and one
user message per example; previous dialogue text is never part of its input.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from compile_sft import compile_row, serialize_action
from dialogue_format import get_format
from dialogue_protocol import (SEGMENTS, SPECIAL_TOKENS, State, encode, segment, state_text,
                               task_messages, ubar_header, ubar_user)
from generator import compile_views
from experiment_config import TRAIN_TASKS

TASKS = TRAIN_TASKS


def load_json(path):
    """Read either a JSON array or newline-delimited JSON records."""
    path = Path(path)
    with path.open(encoding='utf-8') as source:
        if path.suffix == '.jsonl':
            return [json.loads(line) for line in source if line.strip()]
        return json.load(source)


def read_episodes(data_dir, split):
    path = Path(data_dir) / f'{split}.episodes.jsonl'
    if not path.exists():
        path = path.with_suffix('.json')
    episodes = load_json(path)
    seen = set()
    for episode in episodes:
        key = episode['episode_id']
        if key in seen:
            raise ValueError(f'Duplicate episode_id: {key}')
        seen.add(key)
        if episode['split'] != split:
            raise ValueError(f'Episode {key} belongs to {episode["split"]}, not {split}')
        turns = episode['turns']
        if not turns or [turn['index'] for turn in turns] != list(range(len(turns))):
            raise ValueError(f'Non-contiguous turns in {key}')
    return episodes


def corpus_hash(episodes):
    return hashlib.sha256(json.dumps(episodes, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def task_records(episodes, task):
    view = 'extraction' if task == 'dst' else task
    for episode in episodes:
        for row in compile_views(episode)[view]:
            if task == 'dst':
                history = row['history_before'] + [{'role': 'user', 'content': row['utterance']}]
                yield {'episode_id': row['episode_id'], 'turn': row['turn'],
                       'messages': task_messages(task, history, row['state_before']),
                       'completion': state_text(row['target_state'])}
            else:
                yield compile_row(row, view)


def require_eos_token_id(tokenizer):
    if tokenizer.eos_token_id is None:
        raise ValueError('Set tokenizer.eos_token_id: every completion must end with EOS')
    return tokenizer.eos_token_id


def tensor_item(ids, labels):
    return {'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.ones(len(ids), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)}


def tokenize_example(tokenizer, example, *, max_length=None, name='Example'):
    """Supervise completion and native EOS, masking the prompt without truncation."""
    eos = require_eos_token_id(tokenizer)
    prompt = encode(tokenizer, example['prompt'])
    completion = encode(tokenizer, example['completion'])
    # A format suffix is part of the answer, not a replacement for the model EOS.
    if not completion or completion[-1] != eos:
        completion.append(eos)
    ids = prompt + completion
    if max_length is not None and len(ids) > max_length:
        raise ValueError(
            f'{name} has {len(ids)} tokens, exceeding max_length={max_length}. '
            'Choose a longer context or explicitly shorten the conversation before tokenization.'
        )
    return tensor_item(ids, [-100] * len(prompt) + completion)


class DialogueDataset(Dataset):
    view = None

    def __init__(self, tokenizer, split='train', *, data_dir=None, style='plain', max_length=None):
        self.tokenizer = tokenizer
        self.format = get_format(style)
        if max_length is not None and max_length <= 0:
            raise ValueError('max_length must be positive')
        self.max_length = max_length
        root = Path(data_dir) if data_dir is not None else Path(__file__).parent / 'data'
        path = root / self.view / f'{split}.jsonl'
        if not path.exists():
            path = path.with_suffix('.json')
        self.data = load_json(path)

    def __len__(self):
        return len(self.data)

    def format_example(self, index):
        """Preview text before tokenization and the required final EOS ID."""
        row = self.data[index]
        if 'messages' in row and 'completion' in row:
            if row['view'] != self.view:
                raise ValueError(f"Expected {self.view} example, got {row['view']}")
            example = row
        else:
            example = compile_row(row, self.view)
        return self.format.render(example['messages'], example['completion'])

    def __getitem__(self, index):
        return tokenize_example(self.tokenizer, self.format_example(index),
                                max_length=self.max_length, name=f'Example {index}')


class ExtractionDataset(DialogueDataset):
    view = 'extraction'


class HistoryDataset(DialogueDataset):
    view = 'history'


class PolicyDataset(DialogueDataset):
    view = 'policy'


class ResponseDataset(DialogueDataset):
    view = 'response'


class DialogueChainDataset(Dataset):
    """Join extraction with its linked policy/response/history within an episode.

    Each item holds four supervised tasks, not a generated inference rollout.
    User-first episodes include extraction for the first message as well.
    """

    def __init__(self, tokenizer, split='train', **kwargs):
        classes = (ExtractionDataset, PolicyDataset, ResponseDataset, HistoryDataset)
        self.datasets = {cls.view: cls(tokenizer, split, **kwargs) for cls in classes}
        indices = {}
        for view, dataset in self.datasets.items():
            indices[view] = {}
            for index, row in enumerate(dataset.data):
                key = (row['episode_id'], row['turn'])
                if key in indices[view]:
                    raise ValueError(f'Duplicate {view} record: {key}')
                indices[view][key] = index
        self.links = []
        for index, row in enumerate(self.datasets['extraction'].data):
            if row['next_turn'] != row['turn']:
                raise ValueError('Extraction must link to the assistant response in the same turn')
            key = (row['episode_id'], row['next_turn'])
            linked = {'extraction': index}
            for view in ('policy', 'response', 'history'):
                if key not in indices[view]:
                    raise ValueError(f'Missing {view} record for extraction: {key}')
                linked[view] = indices[view][key]
            self.links.append(linked)

    def __len__(self):
        return len(self.links)

    def __getitem__(self, index):
        linked = self.links[index]
        row = self.datasets['extraction'].data[linked['extraction']]
        return {
            'episode_id': row['episode_id'], 'turn': row['turn'], 'next_turn': row['next_turn'],
            **{view: self.datasets[view][offset] for view, offset in linked.items()},
        }


class TaskDataset(Dataset):
    """Individual supervised tasks compiled in memory from canonical episodes."""

    def __init__(self, episodes, tokenizer, task, style='plain', max_length=1024):
        self.items = []
        self.metadata = []
        fmt = get_format(style)
        require_eos_token_id(tokenizer)
        for record in task_records(episodes, task):
            rendered = fmt.render(record['messages'], record['completion'])
            self.items.append(tokenize_example(
                tokenizer, rendered, max_length=max_length,
                name=f'{task} {record["episode_id"]}:{record["turn"]}'))
            self.metadata.append({'episode_id': record['episode_id'], 'turn': record['turn']})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class SessionDataset(TaskDataset):
    """UBAR turns conditioned on the previous B/A, without previous utterances."""

    def __init__(self, episodes, tokenizer, style='plain', max_length=1024, loss_scope='targets'):
        if loss_scope not in ('targets', 'all'):
            raise ValueError('loss_scope must be targets or all')
        eos = require_eos_token_id(tokenizer)
        for token in SPECIAL_TOKENS:
            ids = encode(tokenizer, token)
            if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
                raise ValueError('Register UBAR special tokens before building sessions')
        self.items, self.metadata = [], []
        for episode in episodes:
            previous_state, previous_action = State().snapshot(), None
            for index, turn in enumerate(episode['turns']):
                header = encode(tokenizer, ubar_header(style, previous_state, previous_action))
                user = encode(tokenizer, ubar_user(turn['user'], style))
                output = encode(tokenizer,
                    segment('dst', state_text(turn['state_before'])) +
                    segment('policy', serialize_action(turn['action'])) +
                    segment('response', turn['assistant']))
                ids = header + user + output + [eos]
                if len(ids) > max_length:
                    raise ValueError(f'UBAR turn {episode["episode_id"]}:{index} exceeds max_length')
                labels = ([-100] * len(header) +
                          (user if loss_scope == 'all' else [-100] * len(user)) + output + [eos])
                self.items.append(tensor_item(ids, labels))
                self.metadata.append({'episode_id': episode['episode_id'], 'turn': index})
                previous_state, previous_action = turn['state_before'], turn['action']


def make_dataset(episodes, tokenizer, task, **kwargs):
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    if task == 'ubar':
        return SessionDataset(episodes, tokenizer, **kwargs)
    loss_scope = kwargs.pop('loss_scope', 'targets')
    if loss_scope != 'targets':
        raise ValueError('loss_scope=all is only supported for ubar')
    return TaskDataset(episodes, tokenizer, task, **kwargs)


def collate_batch(batch, padding_value, label_padding_value=-100):
    """Right-pad by position, keeping real EOS tokens supervised even if PAD == EOS."""
    if not batch:
        raise ValueError('Cannot collate an empty batch')
    if padding_value is None:
        raise ValueError('Set a padding token ID before collating')
    return {
        'input_ids': pad_sequence([item['input_ids'] for item in batch], batch_first=True,
                                  padding_value=padding_value),
        'attention_mask': pad_sequence([item['attention_mask'] for item in batch], batch_first=True,
                                       padding_value=0),
        'labels': pad_sequence([item['labels'] for item in batch], batch_first=True,
                               padding_value=label_padding_value),
    }


def chain_collate_batch(batch, padding_value, label_padding_value=-100):
    """Return four sub-batches for separate task losses in one optimizer update."""
    return {view: collate_batch([item[view] for item in batch], padding_value, label_padding_value)
            for view in ('extraction', 'policy', 'response', 'history')}


extraction_collate_batch = collate_batch
history_collate_batch = collate_batch
policy_collate_batch = collate_batch
response_collate_batch = collate_batch
