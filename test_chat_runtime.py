"""Interactive state continuity, stage failures and cancellation, without downloads."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from chat_runtime import generate_chat_turn
from data import segment, state_text, ubar_user
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
        return generate_chat_turn(ScriptedPredictor('ASK(time_of_day)', 'What time is it for you?'), [])

    def test_model_history_and_memory_carry_forward_without_reference_data(self):
        first = self.opening()
        before = deepcopy(first)
        self.assertEqual(first['status'], 'completed')
        self.assertEqual(first['memory']['pending'], 'time_of_day')
        self.assertEqual(first['memory']['attempts']['time_of_day'], 1)
        state = deepcopy(first['memory'])
        state['known']['time_of_day'] = 'evening'
        state['statuses']['time_of_day'] = 'known'
        state['pending'] = None
        model = ScriptedPredictor(state_text(state), 'ASK(activity)', 'What are you doing?')
        message = 'It is evening here, after a long day.'
        second = generate_chat_turn(model, [first], message)
        self.assertEqual([request.task for request in model.requests], ['dst', 'policy', 'response'])
        self.assertEqual(model.requests[0].history, [
            {'role': 'assistant', 'content': first['response']}, {'role': 'user', 'content': message}])
        self.assertEqual(model.requests[0].state, first['memory'])
        self.assertEqual(model.requests[0].session, [first['block'], ubar_user(message)])
        self.assertEqual(model.requests[1].state, state)
        self.assertIn(segment('dst', state_text(state)), model.requests[1].session[-1])
        self.assertEqual(second['memory']['pending'], 'activity')
        self.assertEqual(second['memory']['known']['time_of_day'], 'evening')
        self.assertEqual(first, before)

        state = deepcopy(second['memory'])
        state['wants_exit'] = True
        state['pending'] = None
        closed = generate_chat_turn(ScriptedPredictor(state_text(state), 'CLOSE', 'Goodbye!'),
                                    [first, second], 'I have to go.')
        self.assertTrue(closed['closed'])
        self.assertEqual(closed['memory']['phase'], 'closed')
        with self.assertRaises(ValueError):
            generate_chat_turn(ScriptedPredictor(), [first, second, closed], 'One more question')

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
                failed = generate_chat_turn(ScriptedPredictor('ASK(time_of_day)', output), [])
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
                generate_chat_turn(CancellingPredictor(), [], control=control)
            self.assertEqual(len(model.requests), 1)

    def test_worker_chat_dispatch_needs_no_dataset(self):
        with TemporaryDirectory() as directory:
            with patch('inference.HFPredictor', return_value=ScriptedPredictor('CLOSE', 'Bye!')) as load:
                result = execute({'kind': 'chat', 'config': {'model': 'local-checkpoint', 'device': 'cpu',
                                  'max_new_tokens': 192, 'turns': [], 'message': None}}, JobControl(directory))
            self.assertEqual(result['status'], 'completed')
            load.assert_called_once_with('local-checkpoint', device='cpu', local_files_only=True, max_new_tokens=192)
