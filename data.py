"""SFT datasets over the generated views; formatting is selected at runtime."""

import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from compile_sft import compile_row
from dialogue_format import get_format


def load_json(path):
    """Read either a JSON array or newline-delimited JSON records."""
    path = Path(path)
    with path.open(encoding='utf-8') as source:
        if path.suffix == '.jsonl':
            return [json.loads(line) for line in source if line.strip()]
        return json.load(source)


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
        example = self.format_example(index)
        prompt = self.tokenizer.encode(example['prompt'], add_special_tokens=False)
        completion = self.tokenizer.encode(example['completion'], add_special_tokens=False)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError('Set tokenizer.eos_token_id: every completion must end with EOS')
        # A format suffix is part of the answer, not a replacement for the model EOS.
        # Keep EOS in labels so the last answer token learns to predict it.
        if not completion or completion[-1] != eos_token_id:
            completion.append(eos_token_id)
        input_ids = prompt + completion
        if self.max_length is not None and len(input_ids) > self.max_length:
            raise ValueError(
                f'Example {index} has {len(input_ids)} tokens, exceeding max_length={self.max_length}. '
                'Choose a longer context or explicitly shorten the conversation before tokenization.'
            )
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
            'labels': torch.tensor([-100] * len(prompt) + completion, dtype=torch.long),
        }


class ExtractionDataset(DialogueDataset):
    view = 'extraction'


class HistoryDataset(DialogueDataset):
    view = 'history'


class PolicyDataset(DialogueDataset):
    view = 'policy'


class ResponseDataset(DialogueDataset):
    view = 'response'


class DialogueChainDataset(Dataset):
    """Join extraction[t] with policy/response/history[t+1] within an episode.

    Each item holds four supervised tasks, not a generated inference rollout.
    Opening assistant turns remain available in the individual datasets.
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
            if row['next_turn'] != row['turn'] + 1:
                raise ValueError('Extraction must link to the next assistant turn')
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
