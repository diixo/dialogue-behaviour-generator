"""Experiment datasets built in memory from canonical episodes, without rewriting data.

UBAR is a project adaptation: no database segment; model-output loss by default.
Long sessions use whole-turn windows with the previous turn as masked context.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from compile_sft import PERSONA, build_messages, compile_row, serialize_action
from data import load_json
from dialogue_format import get_format
from generator import compile_views

TASKS = ('dst', 'extraction', 'policy', 'response', 'history', 'ubar')
SEGMENTS = {
    'dst': ('<|state|>', '<|end_state|>'),
    'policy': ('<|action|>', '<|end_action|>'),
    'response': ('<|response|>', '<|end_response|>'),
}
SPECIAL_TOKENS = [token for pair in SEGMENTS.values() for token in pair]


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


def state_text(state):
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def task_messages(task, history, state=None, action=None):
    if task != 'dst':
        return build_messages(history, task, state=state, action=action)
    if state is None:
        raise ValueError('dst requires state before the latest user message')
    instruction = (
        'Update the dialogue state using the latest user message and conversation. '
        'Return only the complete state as a JSON object. Preserve previous facts and refusals; '
        'do not invent unobserved information. State before the latest user message: '
    )
    return [{'role': 'system', 'content': instruction + state_text(state)}] + list(history)


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


def ubar_header(style='plain'):
    fmt = get_format(style)
    instruction = PERSONA + ' Track dialogue state, choose an action, and write a response in that order.'
    return fmt.message_template.format(role='system', content=instruction)


def ubar_user(text, style='plain'):
    return get_format(style).message_template.format(role='user', content=text)


def segment(task, text):
    start, end = SEGMENTS[task]
    return start + text + end


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def tensor_item(ids, labels):
    return {'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.ones(len(ids), dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)}


class TaskDataset(Dataset):
    def __init__(self, episodes, tokenizer, task, style='plain', max_length=1024):
        self.items = []
        self.metadata = []
        fmt = get_format(style)
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer requires eos_token_id')
        for record in task_records(episodes, task):
            rendered = fmt.render(record['messages'], record['completion'])
            prompt = encode(tokenizer, rendered['prompt'])
            target = encode(tokenizer, rendered['completion'])
            if not target or target[-1] != tokenizer.eos_token_id:
                target.append(tokenizer.eos_token_id)
            if len(prompt) + len(target) > max_length:
                raise ValueError(f'{task} {record["episode_id"]}:{record["turn"]} exceeds '
                                 f'max_length={max_length}; increase context or shorten input explicitly')
            self.items.append(tensor_item(prompt + target, [-100] * len(prompt) + target))
            self.metadata.append({'episode_id': record['episode_id'], 'turn': record['turn']})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class SessionDataset(TaskDataset):
    def __init__(self, episodes, tokenizer, style='plain', max_length=1024, loss_scope='targets'):
        if loss_scope not in ('targets', 'all'):
            raise ValueError('loss_scope must be targets or all')
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer requires eos_token_id')
        for token in SPECIAL_TOKENS:
            ids = encode(tokenizer, token)
            if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
                raise ValueError('Register UBAR special tokens before building sessions')
        self.items, self.metadata = [], []
        header = encode(tokenizer, ubar_header(style))
        eos = tokenizer.eos_token_id
        for episode in episodes:
            window, labels = list(header), [-100] * len(header)
            previous = []
            start_turn = 0
            for index, turn in enumerate(episode['turns']):
                user = [] if index == 0 else encode(
                    tokenizer, ubar_user(episode['turns'][index - 1]['user'], style))
                output = encode(tokenizer,
                    segment('dst', state_text(turn['state_before'])) +
                    segment('policy', serialize_action(turn['action'])) +
                    segment('response', turn['assistant']))
                block = user + output
                block_labels = (user if loss_scope == 'all' else [-100] * len(user)) + output
                if len(window) + len(block) + 1 > max_length:
                    if len(window) == len(header):
                        raise ValueError(f'UBAR turn {episode["episode_id"]}:{index} exceeds max_length')
                    # Window boundaries are not conversation endings: no fabricated EOS here.
                    self.items.append(tensor_item(window, labels))
                    self.metadata.append({'episode_id': episode['episode_id'],
                                          'first_turn': start_turn, 'last_turn': index - 1})
                    window = header + previous
                    labels = [-100] * len(window)
                    start_turn = index
                    if len(window) + len(block) + 1 > max_length:
                        raise ValueError('UBAR requires room for two complete turns; increase max_length')
                window += block
                labels += block_labels
                previous = block
            self.items.append(tensor_item(window + [eos], labels + [eos]))
            self.metadata.append({'episode_id': episode['episode_id'],
                                  'first_turn': start_turn, 'last_turn': len(episode['turns']) - 1})


def make_dataset(episodes, tokenizer, task, **kwargs):
    if task not in TASKS:
        raise ValueError(f'Unknown task: {task}')
    if task == 'ubar':
        return SessionDataset(episodes, tokenizer, **kwargs)
    loss_scope = kwargs.pop('loss_scope', 'targets')
    if loss_scope != 'targets':
        raise ValueError('loss_scope=all is only supported for ubar')
    return TaskDataset(episodes, tokenizer, task, **kwargs)
