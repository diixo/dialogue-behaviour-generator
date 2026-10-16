import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from compile_sft import VIEWS, build_messages, compile_row, parse_action, run, serialize_action
from dialogue_format import DialogueFormat, format_example
from generator import compile_views, generate_episode


class FormattingTests(unittest.TestCase):
    def setUp(self):
        self.views = compile_views(generate_episode(random.Random(42), 'example', 'train'))

    def test_all_tasks_keep_content_without_format_markers(self):
        for view in VIEWS:
            with self.subTest(view=view):
                row = self.views[view][0]
                original = copy.deepcopy(row)
                example = compile_row(row, view)
                self.assertEqual(row, original)
                self.assertNotIn('prompt', example)
                self.assertNotIn('<|im_', json.dumps(example))
                self.assertNotIn('hidden_user', example)
                if view == 'extraction':
                    self.assertEqual(json.loads(example['completion']), {
                        'observation': row['target_observation'], 'state': row['target_state'],
                    })
                    self.assertEqual(example['next_turn'], row['turn'])
                    self.assertIn(json.dumps(row['state_before'], sort_keys=True), example['messages'][0]['content'])
                    self.assertEqual(example['messages'][-1], {'role': 'user', 'content': row['utterance']})
                    self.assertEqual(example['messages'][1:-1], row['history_before'])
                elif view == 'policy':
                    self.assertEqual(example['completion'], serialize_action(row['target_action']))
                    self.assertEqual(example['messages'][1:], row['history'])
                else:
                    self.assertEqual(example['completion'], row['target_response'])
                    self.assertEqual(example['messages'][1:], row['history'])
                    if view == 'response':
                        self.assertIn(serialize_action(row['action']), example['messages'][0]['content'])

    def test_extracted_state_builds_next_policy_and_response_inputs(self):
        row = self.views['extraction'][0]
        predicted = json.loads(compile_row(row, 'extraction')['completion'])
        history = row['history_before'] + [{'role': 'user', 'content': row['utterance']}]
        policy = self.views['policy'][row['next_turn']]
        response = self.views['response'][row['next_turn']]
        self.assertEqual(build_messages(history, 'policy', state=predicted['state']),
                         compile_row(policy, 'policy')['messages'])
        action_text = compile_row(policy, 'policy')['completion']
        self.assertEqual(build_messages(history, 'response', state=predicted['state'], action=action_text),
                         compile_row(response, 'response')['messages'])
        baseline = compile_row(self.views['history'][row['next_turn']], 'history')
        self.assertEqual(build_messages(history, 'history'), baseline['messages'])
        self.assertNotIn('Current dialogue state:', baseline['messages'][0]['content'])

    def test_chain_prompts_require_explicit_state_and_action(self):
        for view in ('extraction', 'policy', 'response'):
            with self.subTest(view=view), self.assertRaisesRegex(ValueError, 'requires a dialogue state'):
                build_messages([], view)
        with self.assertRaisesRegex(ValueError, 'requires a dialogue action'):
            build_messages([], 'response', state=self.views['response'][0]['state'])

    def test_policy_completion_can_be_used_as_response_action(self):
        for text in ('ASK(time_of_day)', 'CLARIFY(mood)', 'FOLLOW_UP(books)', 'SWITCH(python)', 'ANSWER', 'ACK', 'CLOSE'):
            self.assertEqual(serialize_action(parse_action(' ' + text + '\n')), text)
        for text in ('ASK', 'CLOSE(mood)', 'FOLLOW_UP(mood)', 'UNKNOWN', 'ASK(activity) extra'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'Invalid dialogue action'):
                parse_action(text)

    def test_styles_preserve_content_and_do_not_mutate_examples(self):
        example = {
            'messages': [{'role': 'user', 'content': 'Hello {name}!\nПривет!'}],
            'completion': 'Hello!',
        }
        original = copy.deepcopy(example)
        self.assertEqual(format_example(example), {
            'prompt': 'user: Hello {name}!\nПривет!\nassistant: ', 'completion': 'Hello!',
        })
        self.assertEqual(format_example(example, 'chatml'), {
            'prompt': '<|im_start|>user\nHello {name}!\nПривет!<|im_end|>\n<|im_start|>assistant\n',
            'completion': 'Hello!<|im_end|>',
        })
        custom = DialogueFormat('[{role}]\n{content}\n', '[assistant]\n', '\nEND')
        self.assertEqual(format_example(example, custom), {
            'prompt': '[user]\nHello {name}!\nПривет!\n[assistant]\n',
            'completion': 'Hello!\nEND',
        })
        self.assertEqual(example, original)
        with self.assertRaisesRegex(ValueError, 'Unknown dialogue style'):
            format_example(example, 'missing')

    def test_exported_json_and_jsonl_match_without_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for view in VIEWS:
                source = root / f'{view}.jsonl'
                source.write_text('\n'.join(json.dumps(row) for row in self.views[view]) + '\n\n', encoding='utf-8')
                dest, json_dest = root / 'out' / f'{view}.jsonl', root / 'json' / f'{view}.json'
                run(source, dest, view, json_dest)
                records = [json.loads(line) for line in dest.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(records, json.loads(json_dest.read_text(encoding='utf-8')))
                self.assertEqual(len(records), len(self.views[view]))
                self.assertNotIn('<|im_', dest.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
