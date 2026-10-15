import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from compile_sft import compile_row
from data import (ExtractionDataset, HistoryDataset, PolicyDataset, ResponseDataset,
                  DialogueChainDataset, collate_batch, chain_collate_batch)
from dialogue_format import DialogueFormat
from generator import compile_views, generate_episode


class CharacterTokenizer:
    """Local deterministic tokenizer; tests require no model download."""
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError('Unexpected automatic special tokens')
        return [ord(character) + 1 for character in text]


DATASETS = (ExtractionDataset, HistoryDataset, PolicyDataset, ResponseDataset)
STYLES = ('plain', 'chatml', DialogueFormat(completion_suffix='STOP'), DialogueFormat(completion_suffix=''))


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = CharacterTokenizer()
        views = compile_views(generate_episode(random.Random(42), 'example', 'train'))
        for view, rows in views.items():
            for directory, suffix, records in [
                ('data', '.jsonl', rows), ('json', '.json', rows),
                ('sft', '.jsonl', [compile_row(row, view) for row in rows]),
                ('sft_json', '.json', [compile_row(row, view) for row in rows]),
            ]:
                path = self.root / directory / view / f'train{suffix}'
                path.parent.mkdir(parents=True, exist_ok=True)
                content = json.dumps(records) if suffix == '.json' else '\n'.join(json.dumps(row) for row in records)
                path.write_text(content, encoding='utf-8')

    def dataset(self, cls=HistoryDataset, directory='data', **kwargs):
        return cls(self.tokenizer, data_dir=self.root / directory, **kwargs)

    def test_all_views_load_raw_and_sft_json_and_jsonl_identically(self):
        for cls in DATASETS:
            with self.subTest(view=cls.view):
                expected = self.dataset(cls)
                for directory in ('json', 'sft', 'sft_json'):
                    actual = self.dataset(cls, directory)
                    self.assertEqual(len(actual), len(expected))
                    self.assertEqual(actual.format_example(0), expected.format_example(0))
                    for key in expected[0]:
                        self.assertTrue(torch.equal(actual[0][key], expected[0][key]))

    def test_only_completion_and_eos_are_supervised(self):
        for cls in DATASETS:
            for style in STYLES:
                with self.subTest(view=cls.view, style=style):
                    dataset = self.dataset(cls, style=style)
                    rendered = dataset.format_example(0)
                    item = dataset[0]
                    boundary = len(self.tokenizer.encode(rendered['prompt']))
                    completion = self.tokenizer.encode(rendered['completion']) + [0]
                    self.assertEqual(item['labels'][:boundary].tolist(), [-100] * boundary)
                    self.assertEqual(item['labels'][boundary:].tolist(), completion)
                    self.assertEqual(item['input_ids'][boundary:].tolist(), completion)
                    self.assertEqual(item['attention_mask'].tolist(), [1] * len(item['input_ids']))

    def test_padding_does_not_mask_real_eos_when_pad_equals_eos(self):
        dataset = self.dataset()
        samples = [dataset[0], dataset[1]]
        batch = collate_batch(samples, padding_value=0)
        self.assertEqual(batch['input_ids'].shape[0], 2)
        self.assertLess(len(samples[0]['input_ids']), batch['input_ids'].shape[1])
        for index, sample in enumerate(samples):
            length = len(sample['input_ids'])
            self.assertEqual(batch['labels'][index, length - 1].item(), 0)
            self.assertTrue(torch.equal(batch['labels'][index, :length], sample['labels']))
            self.assertTrue(torch.all(batch['labels'][index, length:] == -100))
            self.assertTrue(torch.all(batch['attention_mask'][index, length:] == 0))
            self.assertTrue(torch.all(batch['input_ids'][index, length:] == 0))

    def test_style_is_runtime_only_and_suffix_is_followed_by_eos(self):
        path = self.root / 'data/history/train.jsonl'
        original = path.read_bytes()
        for style, suffix in [('chatml', '<|im_end|>'), (DialogueFormat(completion_suffix='STOP'), 'STOP')]:
            with self.subTest(style=style):
                dataset = self.dataset(style=style)
                sample = dataset[0]
                expected = self.tokenizer.encode(suffix) + [self.tokenizer.eos_token_id]
                self.assertEqual(sample['labels'][-len(expected):].tolist(), expected)
                self.assertEqual(sample['input_ids'][-1].item(), self.tokenizer.eos_token_id)
                self.assertEqual(dataset.data, self.dataset().data)
        self.assertEqual(path.read_bytes(), original)

    def test_overlong_example_is_rejected_instead_of_losing_target_or_history(self):
        with self.assertRaisesRegex(ValueError, 'exceeding max_length'):
            self.dataset(max_length=10)[0]

    def test_missing_eos_is_rejected_for_every_style(self):
        self.tokenizer.eos_token_id = None
        for style in STYLES:
            with self.subTest(style=style), self.assertRaisesRegex(ValueError, 'eos_token_id'):
                self.dataset(style=style)[0]

    def test_existing_final_eos_is_not_duplicated(self):
        class EOSAwareTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=False):
                chunks = text.split('<eos>')
                tokens = super().encode(chunks[0], add_special_tokens=add_special_tokens)
                for chunk in chunks[1:]:
                    tokens.append(self.eos_token_id)
                    tokens.extend(super().encode(chunk, add_special_tokens=add_special_tokens))
                return tokens

        self.tokenizer = EOSAwareTokenizer()
        dataset = self.dataset(style=DialogueFormat(completion_suffix='<eos>'))
        sample = dataset[0]
        expected = self.tokenizer.encode(dataset.format_example(0)['completion'])
        supervised = sample['labels'][sample['labels'] != -100].tolist()
        self.assertEqual(supervised, expected)
        self.assertEqual(supervised[-1], self.tokenizer.eos_token_id)
        self.assertEqual(supervised.count(self.tokenizer.eos_token_id), 1)

    def test_max_length_includes_the_required_eos(self):
        dataset = self.dataset(style=DialogueFormat(completion_suffix='STOP'))
        example = dataset.format_example(0)
        text_length = sum(len(self.tokenizer.encode(text)) for text in example.values())
        with self.assertRaisesRegex(ValueError, 'exceeding max_length'):
            self.dataset(style=dataset.format, max_length=text_length)[0]
        sample = self.dataset(style=dataset.format, max_length=text_length + 1)[0]
        self.assertEqual(sample['labels'][-1].item(), self.tokenizer.eos_token_id)

    def test_shifted_causal_loss_trains_eos_but_ignores_padding(self):
        dataset = self.dataset(style='chatml')
        samples = [dataset[0], dataset[1]]
        batch = collate_batch(samples, padding_value=self.tokenizer.eos_token_id)
        # Decoder logits at position t predict labels[t + 1].
        targets = batch['labels'][:, 1:]
        vocab_size = int(batch['input_ids'].max()) + 1
        logits = torch.zeros(*targets.shape, vocab_size, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        for index, sample in enumerate(samples):
            eos_prediction_position = len(sample['input_ids']) - 2
            self.assertEqual(targets[index, eos_prediction_position].item(), self.tokenizer.eos_token_id)
            self.assertLess(logits.grad[index, eos_prediction_position, self.tokenizer.eos_token_id].item(), 0)
        self.assertTrue(torch.all(logits.grad[targets == -100] == 0))

    def test_chain_joins_by_episode_and_next_turn_even_when_rows_are_shuffled(self):
        other = compile_views(generate_episode(random.Random(19), 'other', 'train'))
        for view, rows in other.items():
            path = self.root / 'data' / view / 'train.jsonl'
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] + rows
            random.Random(view).shuffle(rows)
            path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        chains = DialogueChainDataset(self.tokenizer, data_dir=self.root / 'data')
        self.assertEqual(len(chains), len(chains.datasets['extraction']))
        self.assertEqual(len(chains.datasets['history']), len(chains) + 2)
        closes = 0
        for index, linked in enumerate(chains.links):
            item = chains[index]
            extraction = chains.datasets['extraction'].data[linked['extraction']]
            self.assertEqual(item['next_turn'], item['turn'] + 1)
            for view in ('policy', 'response', 'history'):
                row = chains.datasets[view].data[linked[view]]
                self.assertEqual((row['episode_id'], row['turn']), (item['episode_id'], item['next_turn']))
                self.assertEqual(row['history'], extraction['history_before'] + [{'role':'user', 'content':extraction['utterance']}])
                if view != 'history':
                    self.assertEqual(row['state'], extraction['target_state'])
                self.assertEqual(item[view]['labels'][-1].item(), self.tokenizer.eos_token_id)
            closes += chains.datasets['response'].data[linked['response']]['action']['type'] == 'CLOSE'
        self.assertEqual(closes, 2)
        batch = chain_collate_batch([chains[0], chains[1]], padding_value=0)
        self.assertEqual(set(batch), {'extraction', 'policy', 'response', 'history'})
        for sub_batch in batch.values():
            self.assertEqual(sub_batch['input_ids'].shape[0], 2)
            self.assertTrue(torch.all(sub_batch['labels'][sub_batch['attention_mask'] == 0] == -100))

    def test_chain_can_load_structured_sft_records(self):
        chains = DialogueChainDataset(self.tokenizer, data_dir=self.root / 'sft')
        self.assertEqual(chains[0]['turn'], 0)
        self.assertEqual(chains[0]['next_turn'], 1)
        self.assertEqual(len(chains), len(self.dataset(ExtractionDataset)))

    def test_chain_rejects_missing_next_turn(self):
        path = self.root / 'data/policy/train.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        path.write_text('\n'.join(json.dumps(row) for row in rows if row['turn'] != 1), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Missing policy record'):
            DialogueChainDataset(self.tokenizer, data_dir=self.root / 'data')

    def test_chain_rejects_duplicate_keys(self):
        path = self.root / 'data/history/train.jsonl'
        content = path.read_text(encoding='utf-8')
        path.write_text(content + '\n' + content.splitlines()[0], encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate history record'):
            DialogueChainDataset(self.tokenizer, data_dir=self.root / 'data')


if __name__ == '__main__':
    unittest.main()
