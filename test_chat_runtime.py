"""Interactive state continuity, stage failures and cancellation, without downloads."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from chat_runtime import generate_chat_turn, main
from dialogue_protocol import State, segment, state_text, ubar_user
from inference import Prediction
from job_runtime import JobCancelled, JobControl
from run_job import execute


class ScriptedPredictor:
    task = 'ubar'
    style = 'plain'

    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        output = next(self.outputs)
        return output if isinstance(output, Prediction) else Prediction(output, tokens=5)


class ChatRuntimeTests(TestCase):
    def opening(self):
        return generate_chat_turn(ScriptedPredictor(state_text(State().snapshot()),
                                  'ASK(time_of_day)', 'What time is it for you?'), [], 'Hi!')

    def test_first_user_message_is_extracted_before_action_and_response(self):
        state = State().snapshot()
        state['known']['time_of_day'] = 'evening'
        state['statuses']['time_of_day'] = 'known'
        message = 'It is evening here.'
        model = ScriptedPredictor(state_text(state), 'ASK(activity)', 'What are you doing?')
        first = generate_chat_turn(model, [], message)
        self.assertEqual(first['status'], 'completed')
        self.assertEqual([r.task for r in model.requests], ['dst', 'policy', 'response'])
        self.assertEqual(model.requests[0].history, [{'role': 'user', 'content': message}])
        self.assertEqual(model.requests[0].state, State().snapshot())
        self.assertEqual(model.requests[0].session, [ubar_user(message)])
        self.assertEqual(model.requests[1].state, state)
        self.assertEqual(first['state'], state)
        self.assertNotIn('memory', first)
        for message in (None, '', '  '):
            with self.assertRaises(ValueError):
                generate_chat_turn(ScriptedPredictor(), [], message)

    def test_model_history_and_memory_carry_forward_without_reference_data(self):
        first = self.opening()
        before = deepcopy(first)
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(first['state'], State().snapshot())
        state = deepcopy(first['state'])
        state['known']['time_of_day'] = 'evening'
        state['statuses']['time_of_day'] = 'known'
        state['pending'] = None
        state['attempts']['time_of_day'] = 4
        model = ScriptedPredictor(state_text(state), 'ASK(activity)', 'What are you doing?')
        message = 'It is evening here, after a long day.'
        second = generate_chat_turn(model, [first], message)
        self.assertEqual([request.task for request in model.requests], ['dst', 'policy', 'response'])
        self.assertEqual(model.requests[0].history, [
            {'role': 'user', 'content': first['message']},
            {'role': 'assistant', 'content': first['response']}, {'role': 'user', 'content': message}])
        self.assertEqual(model.requests[0].state, first['state'])
        self.assertEqual(model.requests[0].session, [first['block'], ubar_user(message)])
        self.assertEqual(model.requests[1].state, state)
        self.assertIn(segment('dst', state_text(state)), model.requests[1].session[-1])
        self.assertEqual(second['state'], state)
        self.assertEqual(first, before)

        state = deepcopy(second['state'])
        state['wants_exit'] = True
        state['pending'] = None
        closed = generate_chat_turn(ScriptedPredictor(state_text(state), 'CLOSE', 'Goodbye!'),
                                    [first, second], 'I have to go.')
        self.assertTrue(closed['closed'])
        self.assertEqual(closed['state'], state)
        with self.assertRaises(ValueError):
            generate_chat_turn(ScriptedPredictor(), [first, second, closed], 'One more question')

    def test_model_segments_are_preserved_and_legacy_committed_memory_is_ignored(self):
        state = State().snapshot()
        state_output = json.dumps(state, indent=2)
        action_output = ' ASK(activity) '
        response = '  A model reply with its own formatting.\n'
        model = ScriptedPredictor(state_output, action_output, response)
        first = generate_chat_turn(model, [], 'Hi!')
        expected = ubar_user('Hi!') + segment('dst', state_output)
        self.assertEqual(model.requests[1].session, [expected])
        expected += segment('policy', action_output)
        self.assertEqual(model.requests[2].session, [expected])
        self.assertEqual(model.requests[2].action, {'type': 'ASK', 'target': 'activity'})
        expected += segment('response', response)
        self.assertEqual(first['block'], expected)
        self.assertEqual(first['response'], response)
        # Older saved chats may contain a state modified by commit_action.
        first['memory'] = deepcopy(state)
        first['memory']['pending'] = 'activity'
        first['memory']['attempts']['activity'] = 1
        before = deepcopy(first)
        model = ScriptedPredictor(state_output, 'ACK', 'Got it.')
        result = generate_chat_turn(model, [first], 'Reading.')
        self.assertEqual(model.requests[0].state, state)
        self.assertEqual(model.requests[0].session, [expected, ubar_user('Reading.')])
        self.assertEqual(result['state'], state)
        self.assertEqual(first, before)

    def test_model_action_alone_controls_closure_and_behaviour_is_not_corrected(self):
        state = State().snapshot()
        state['wants_exit'] = True
        state['known']['mood'] = 'happy'
        state['statuses']['mood'] = 'known'
        model = ScriptedPredictor(state_text(state), 'ASK(mood)', 'How are you feeling?')
        result = generate_chat_turn(model, [], 'I am happy, but I have to go.')
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(result['closed'])
        self.assertEqual(result['action'], {'type': 'ASK', 'target': 'mood'})
        self.assertEqual(result['state'], state)
        self.assertEqual(model.requests[-1].state, state)
        self.assertEqual(model.requests[-1].action, result['action'])
        model = ScriptedPredictor(state_text(State().snapshot()), 'CLOSE', 'Bye!')
        result = generate_chat_turn(model, [], 'Tell me more.')
        self.assertTrue(result['closed'])
        self.assertEqual(result['state'], State().snapshot())

    def test_invalid_or_unfinished_generation_never_advances_memory(self):
        first = self.opening()
        model = ScriptedPredictor('{bad json')
        failed = generate_chat_turn(model, [first], 'Hello')
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(failed['stages']['dst']['text'], '{bad json')
        self.assertNotIn('memory', failed)
        self.assertNotIn('block', failed)
        for output in (Prediction('Incomplete', stopped=False), Prediction(''), Prediction('Error', error='bad output')):
            with self.subTest(output=output):
                failed = generate_chat_turn(ScriptedPredictor(state_text(State().snapshot()),
                                            'ASK(time_of_day)', output), [], 'Hi!')
                self.assertEqual(failed['status'], 'failed')
                self.assertNotIn('memory', failed)

    def test_cancellation_between_stages_prevents_next_generation(self):
        with TemporaryDirectory() as directory:
            control = JobControl(directory)
            model = ScriptedPredictor('ASK(time_of_day)')
            def stop_after_prediction(request):
                result = model(request)
                (Path(directory) / 'stop.request').touch()
                return result
            class CancellingPredictor:
                task = 'ubar'
                style = 'plain'
                __call__ = staticmethod(stop_after_prediction)
            with self.assertRaises(JobCancelled):
                generate_chat_turn(CancellingPredictor(), [], 'Hi!', control=control)
            self.assertEqual(len(model.requests), 1)

    def test_worker_chat_dispatch_needs_no_dataset(self):
        with TemporaryDirectory() as directory:
            with patch('inference.HFPredictor', return_value=ScriptedPredictor(
                    state_text(State().snapshot()), 'CLOSE', 'Bye!')) as load:
                result = execute({'kind': 'chat', 'config': {'model': 'local-checkpoint', 'device': 'cpu',
                                  'max_new_tokens': 192, 'turns': [], 'message': 'Bye!'}}, JobControl(directory))
            self.assertEqual(result['status'], 'completed')
            load.assert_called_once_with('local-checkpoint', device='cpu', local_files_only=True, max_new_tokens=192)


class StandaloneChatTests(TestCase):
    def test_portable_files_work_without_site_packages_or_repository(self):
        root = Path(__file__).parent
        with TemporaryDirectory() as directory:
            for name in ('chat_runtime.py', 'inference.py', 'dialogue_protocol.py', 'dialogue_format.py'):
                (Path(directory) / name).write_bytes((root / name).read_bytes())
            code = '''
import sys
sys.path.insert(0, sys.argv[1])
from chat_runtime import generate_chat_turn
from dialogue_protocol import State, state_text
from inference import Prediction
class Model:
    task = 'ubar'
    style = 'plain'
    def __call__(self, request):
        return Prediction({'dst': state_text(State().snapshot()), 'policy': 'CLOSE', 'response': 'Bye!'}[request.task])
for style in ('plain', 'chatml'):
    model = Model()
    model.style = style
    result = generate_chat_turn(model, [], 'Hi!')
    assert result['status'] == 'completed' and result['closed']
for name in ('torch', 'transformers', 'django', 'accelerate', 'data', 'generator', 'compile_sft',
             'trainer', 'benchmark', 'experiment_config', 'job_runtime'):
    assert name not in sys.modules, name
print('portable OK')
'''
            result = subprocess.run([sys.executable, '-I', '-S', '-c', code, directory], cwd=directory,
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('portable OK', result.stdout)
            help_result = subprocess.run([sys.executable, '-S', 'chat_runtime.py', '--help'], cwd=directory,
                                         capture_output=True, text=True, timeout=20)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn('--model', help_result.stdout)
            self.assertIn('--device', help_result.stdout)

    def test_cli_loads_once_resets_and_keeps_failed_turns_out_of_history(self):
        model = ScriptedPredictor(state_text(State().snapshot()), 'CLOSE', 'Bye!',
                                  '{bad json', state_text(State().snapshot()), 'ACK', 'A fresh reply.')
        output, errors = io.StringIO(), io.StringIO()
        with TemporaryDirectory() as directory, patch('chat_runtime.HFPredictor', return_value=model) as load, \
                patch('builtins.input', side_effect=['Hi!', 'ignored after CLOSE', '/reset', ' ', 'fail', 'retry', '/exit']), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            main(['--model', directory])
            load.assert_called_once_with(directory, device='cpu', local_files_only=True, max_new_tokens=192)
        self.assertIn('Bye!', output.getvalue())
        self.assertIn('A fresh reply.', output.getvalue())
        self.assertTrue(errors.getvalue())
        self.assertEqual(len(model.requests), 7)
        self.assertEqual(model.requests[4].history, [{'role': 'user', 'content': 'retry'}])
        self.assertEqual(model.requests[4].session, [ubar_user('retry')])
        self.assertEqual(model.requests[4].state, State().snapshot())

    def test_cli_handles_eof_interrupt_and_device_options(self):
        with TemporaryDirectory() as directory:
            for interruption in (EOFError, KeyboardInterrupt):
                model = ScriptedPredictor()
                with self.subTest(interruption=interruption), \
                        patch('chat_runtime.HFPredictor', return_value=model) as load, \
                        patch('builtins.input', side_effect=interruption), contextlib.redirect_stdout(io.StringIO()):
                    main(['--model', directory, '--device', 'cuda:0', '--max-new-tokens', '256'])
                    load.assert_called_once_with(directory, device='cuda:0', local_files_only=True, max_new_tokens=256)
                    self.assertEqual(model.requests, [])

    def test_cli_missing_dependencies_and_invalid_arguments_are_actionable(self):
        with TemporaryDirectory() as directory:
            for args in (['--model', str(Path(directory) / 'missing')], ['--max-new-tokens', '0']):
                with self.subTest(args=args), patch('chat_runtime.HFPredictor') as load, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
            errors = io.StringIO()
            with patch('chat_runtime.HFPredictor', side_effect=ImportError('torch')), \
                    contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                main(['--model', directory])
            self.assertEqual(error.exception.code, 2)
            self.assertIn('python -m pip install -r requirements-chat.txt', errors.getvalue())
