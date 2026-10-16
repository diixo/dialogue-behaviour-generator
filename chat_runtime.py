"""Standalone chat with a trained UBAR checkpoint: python chat_runtime.py --model outputs.

Only torch and transformers are required to load the model. Importing this module
or using --help needs only the standard library and the lightweight runtime files.
State, action and reply come exclusively from model predictions.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

from dialogue_protocol import State, parse_action, parse_state, segment, ubar_user
from inference import HFPredictor, Request


def generate_chat_turn(predictor, turns, message=None, control=None):
    if predictor.task != 'ubar':
        raise ValueError('Для чата с цепочкой B → A → R нужна модель UBAR.')
    if turns and turns[-1]['closed']:
        raise ValueError('Модель завершила разговор. Начните новый диалог.')
    if not message or not message.strip():
        raise ValueError('Введите сообщение пользователя, чтобы продолжить диалог.')

    history, blocks = [], []
    for turn in turns:
        if turn['message'] is not None:
            history.append({'role': 'user', 'content': turn['message']})
        history.append({'role': 'assistant', 'content': turn['response']})
        blocks.append(turn['block'])
    # Previous B stays exactly as predicted. The model sees its prior A/R in the
    # session and must infer their consequences itself when generating the next B.
    previous_state = deepcopy(turns[-1]['state']) if turns else State().snapshot()
    current = ubar_user(message, predictor.style)
    history.append({'role': 'user', 'content': message})
    stages = {}

    def predict(task, state, action=None):
        if control:
            control.check_stop()
            control.emit(phase='chat_' + task)
        prediction = predictor(Request(task, deepcopy(history), deepcopy(state), deepcopy(action),
                                       blocks + [current]))
        stages[task] = {'text': prediction.text, 'tokens': prediction.tokens,
                        'stopped': prediction.stopped}
        if control:
            control.check_stop()
        if not prediction.stopped:
            raise ValueError(f'{task}: модель не выдала маркер завершения сегмента.')
        if prediction.error:
            raise ValueError(f'{task}: {prediction.error}')
        return prediction.text

    try:
        state_output = predict('dst', previous_state)
        state = parse_state(state_output)
        current += segment('dst', state_output)
        action_output = predict('policy', state)
        action = parse_action(action_output)
        current += segment('policy', action_output)
        response = predict('response', state, action)
        if not response.strip():
            raise ValueError('response: модель вернула пустой ответ.')
        current += segment('response', response)
    except (ValueError, TypeError, KeyError) as error:
        return {'status': 'failed', 'error': str(error), 'stages': stages}

    return {'status': 'completed', 'message': message, 'response': response,
            'state': state, 'action': action,
            'block': current, 'closed': action['type'] == 'CLOSE', 'stages': stages}


def terminal_chat(predictor):
    """Keep one model loaded; failed turns never enter the conversation history."""
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
        result = generate_chat_turn(predictor, turns, message)
        if result['status'] != 'completed':
            print(f'Ошибка генерации: {result["error"]}', file=sys.stderr)
            continue
        turns.append(result)
        print(f'Модель: {result["response"]}')
        if result['closed']:
            print('Модель выбрала CLOSE. /reset — новый диалог, /exit — выход.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='outputs', help='Local trained UBAR model/checkpoint directory')
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
        if predictor.task != 'ubar':
            parser.error('Нужен обученный UBAR checkpoint с experiment.json (task=ubar).')
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
