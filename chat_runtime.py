"""Chat with a trained UBAR or history checkpoint: python chat_runtime.py --model outputs.

Only torch and transformers are required to load the model. Importing this module
or using --help needs only the standard library and the lightweight runtime files.
UbarChat predicts state, action and reply; HistoryChat predicts only a reply.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

from dialogue_protocol import State, parse_action, parse_state, segment
from inference import HFPredictor, Request


class _Chat:
    """Shared turn bookkeeping; each implementation defines its model requests."""

    def __init__(self, predictor):
        if predictor.task != self.task:
            raise ValueError(f'{type(self).__name__} требует checkpoint с task={self.task}.')
        self.predictor = predictor

    def generate_turn(self, turns, message=None, control=None):
        """Generate a turn without mutating the caller's saved conversation."""
        if turns and turns[-1]['closed']:
            raise ValueError('Модель завершила разговор. Начните новый диалог.')
        if not message or not message.strip():
            raise ValueError('Введите сообщение пользователя, чтобы продолжить диалог.')
        history = []
        for turn in (turns if self.task == 'history' else []):
            if turn['message'] is not None:
                history.append({'role': 'user', 'content': turn['message']})
            history.append({'role': 'assistant', 'content': turn['response']})
        history.append({'role': 'user', 'content': message})
        stages = {}
        try:
            result = self._generate(history, turns, stages, control)
        except (ValueError, TypeError, KeyError) as error:
            return {'status': 'failed', 'error': str(error), 'stages': stages}
        return {'status': 'completed', 'message': message, **result, 'stages': stages}

    def _predict(self, request, stages, control):
        if control:
            control.check_stop()
            control.emit(phase='chat_' + request.task)
        prediction = self.predictor(deepcopy(request))
        stages[request.task] = {'text': prediction.text, 'tokens': prediction.tokens,
                                'stopped': prediction.stopped}
        if control:
            control.check_stop()
        if not prediction.stopped:
            marker = 'маркер завершения сегмента' if self.task == 'ubar' else 'EOS'
            raise ValueError(f'{request.task}: модель не выдала {marker}.')
        if prediction.error:
            raise ValueError(f'{request.task}: {prediction.error}')
        if request.task in ('response', 'history') and not prediction.text.strip():
            raise ValueError(f'{request.task}: модель вернула пустой ответ.')
        return prediction.text


class UbarChat(_Chat):
    """Predict B → A → R from the latest B/A memory and the new user message."""

    task = 'ubar'

    def _generate(self, history, turns, stages, control):
        # Never commit or repair predicted B: the model sees the actual last A
        # and learns to account for its consequences when predicting the next B.
        previous_state = deepcopy(turns[-1]['state']) if turns else State().snapshot()
        previous_action = deepcopy(turns[-1]['action']) if turns else None
        current = ''

        def predict(task, state, action=None):
            return self._predict(Request(task, history, state, action,
                                         previous_state, previous_action, current), stages, control)

        state_output = predict('dst', previous_state)
        state = parse_state(state_output)
        current += segment('dst', state_output)
        action_output = predict('policy', state)
        action = parse_action(action_output)
        current += segment('policy', action_output)
        response = predict('response', state, action)
        return {'response': response, 'state': state, 'action': action,
                'closed': action['type'] == 'CLOSE'}


class HistoryChat(_Chat):
    """Generate one plain reply from user/assistant messages, ending in a user message."""

    task = 'history'

    def reply(self, history, control=None):
        """Return a reply string; invalid, unfinished or empty output raises ValueError."""
        return self._reply(history, {}, control)

    def _reply(self, history, stages, control):
        if not history or len(history) % 2 != 1:
            raise ValueError('История должна начинаться и заканчиваться сообщением пользователя.')
        for index, message in enumerate(history):
            role = 'assistant' if index % 2 else 'user'
            if (not isinstance(message, dict) or message.get('role') != role or
                    not isinstance(message.get('content'), str) or not message['content'].strip()):
                raise ValueError('История должна содержать непустые сообщения user и assistant по очереди.')
        messages = [{'role': row['role'], 'content': row['content']} for row in history]
        return self._predict(Request('history', messages), stages, control)

    def _generate(self, history, turns, stages, control):
        # EOS finishes the reply, not the conversation; there is no CLOSE action.
        return {'response': self._reply(history, stages, control), 'closed': False}


def create_chat(predictor):
    """Select the chat protocol from the checkpoint's experiment.json task."""
    implementations = {'ubar': UbarChat, 'history': HistoryChat}
    if predictor.task not in implementations:
        raise ValueError('Для чата нужен checkpoint с experiment.json (task=ubar или task=history).')
    return implementations[predictor.task](predictor)


def generate_chat_turn(predictor, turns, message=None, control=None):
    """Compatibility entry point shared by queued jobs and existing callers."""
    return create_chat(predictor).generate_turn(turns, message, control)


def terminal_chat(predictor):
    """Keep one model loaded; failed turns never enter the conversation history."""
    chat = create_chat(predictor)
    turns = []
    print('Напишите сообщение. /reset — новый диалог, /exit — выход.')
    while True:
        message = input('Вы: ').strip()
        if message == '/exit':
            return
        if message == '/reset':
            turns.clear()
            print('Начат новый диалог.')
            continue
        if not message:
            continue
        if turns and turns[-1]['closed']:
            print('Модель завершила разговор. Введите /reset для нового диалога.')
            continue
        result = chat.generate_turn(turns, message)
        if result['status'] != 'completed':
            print(f'Ошибка генерации: {result["error"]}', file=sys.stderr)
            continue
        turns.append(result)
        print(f'Модель: {result["response"]}')
        if result['closed']:
            print('Модель выбрала CLOSE. /reset — новый диалог, /exit — выход.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='outputs', help='Local trained UBAR or history model/checkpoint directory')
    parser.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu, cuda or cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=192,
                        help='Per-segment budget; policy/reply caps are 32/128')
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error('--max-new-tokens must be positive')
    if not Path(args.model).is_dir():
        parser.error(f'Каталог модели не найден: {args.model}')
    try:
        print(f'Загрузка модели: {args.model} ({args.device})…', flush=True)
        predictor = HFPredictor(args.model, device=args.device, local_files_only=True,
                                max_new_tokens=args.max_new_tokens)
        terminal_chat(predictor)
    except (EOFError, KeyboardInterrupt):
        print()
    except ImportError as error:
        parser.exit(2, f'Не удалось загрузить зависимости: {error}\n'
                       'Установите их: python -m pip install -r requirements-chat.txt\n')
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Ошибка чата: {error}\n')


if __name__ == '__main__':
    main()
