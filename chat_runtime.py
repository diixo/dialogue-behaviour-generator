"""One interactive UBAR turn, using only the model's own conversation history."""

from copy import deepcopy

from compile_sft import parse_action, serialize_action
from data import segment, state_text, ubar_user
from generator import Action, State, commit_action
from inference import Request, parse_state


def generate_chat_turn(predictor, turns, message=None, control=None):
    if predictor.task != 'ubar':
        raise ValueError('Для чата с цепочкой B → A → R нужна модель UBAR.')
    if turns and turns[-1]['closed']:
        raise ValueError('Модель завершила разговор. Начните новый диалог.')
    if bool(turns) != bool(message and message.strip()):
        raise ValueError('Первая реплика принадлежит модели; затем требуется сообщение пользователя.')

    history, blocks = [], []
    for turn in turns:
        if turn['message'] is not None:
            history.append({'role': 'user', 'content': turn['message']})
        history.append({'role': 'assistant', 'content': turn['response']})
        blocks.append(turn['block'])
    memory = deepcopy(turns[-1]['memory']) if turns else State().snapshot()
    current = ubar_user(message, predictor.style) if message is not None else ''
    if message is not None:
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
        state = parse_state(predict('dst', memory)) if turns else memory
        current += segment('dst', state_text(state))
        action = parse_action(predict('policy', state))
        current += segment('policy', serialize_action(action))
        response = predict('response', state, action).strip()
        if not response:
            raise ValueError('response: модель вернула пустой ответ.')
        current += segment('response', response)
        updated = State.from_snapshot(state)
        commit_action(updated, Action(**action))
    except (ValueError, TypeError, KeyError) as error:
        return {'status': 'failed', 'error': str(error), 'stages': stages}

    return {'status': 'completed', 'message': message, 'response': response,
            'state': state, 'action': action, 'memory': updated.snapshot(),
            'block': current, 'closed': action['type'] == 'CLOSE', 'stages': stages}
