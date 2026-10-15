"""Export structured SFT examples without tokenizer or chat-format markers."""

import argparse
import json
import re
from pathlib import Path

PERSONA = (
    "You are a friendly, curious and respectful conversational assistant. "
    "Follow the user's lead. Do not repeat answered questions. "
    "Respect refusals and requests to end."
)
VIEWS = ('policy', 'response', 'history', 'extraction')
SPLITS = ('train', 'validation', 'test')


def serialize_action(action):
    return action['type'] + (f"({action['target']})" if action.get('target') else '')


def parse_action(text):
    """Convert a decoded policy completion into the action consumed by response."""
    match = re.fullmatch(r'([A-Z_]+)(?:\(([a-z_]+)\))?', text.strip())
    if match is None:
        raise ValueError(f'Invalid dialogue action: {text!r}')
    kind, target = match.groups()
    slots = {'time_of_day', 'activity', 'mood'}
    topics = {'books', 'work', 'music', 'relaxing', 'python'}
    allowed = {'ASK': slots, 'CLARIFY': slots, 'FOLLOW_UP': topics, 'SWITCH': topics,
               'ANSWER': {None}, 'ACK': {None}, 'CLOSE': {None}}
    if kind not in allowed or target not in allowed[kind]:
        raise ValueError(f'Invalid dialogue action: {text!r}')
    return {'type': kind, 'target': target}


def build_messages(history, view, *, state=None, action=None):
    """Build training or inference inputs using observed/predicted state and action.

    For extraction, history includes the latest user message and state is the memory
    after committing the preceding assistant action, before observing that message.
    For policy/response, pass the state returned by extraction. No targets are read.
    """
    if view not in VIEWS:
        raise ValueError(f'Unsupported view: {view}')
    if view != 'history' and state is None:
        raise ValueError(f'{view} requires a dialogue state')
    if view == 'policy':
        system = PERSONA + ' Predict the next dialogue action using the state and conversation.'
    elif view == 'response':
        if action is None:
            raise ValueError('response requires a dialogue action')
        if isinstance(action, str):
            action = parse_action(action)
        system = PERSONA + ' Write the next assistant message for this dialogue action: '
        system += serialize_action(action) + '.'
    elif view == 'history':
        system = PERSONA
    else:
        system = (
            'Read the latest user message and update the supplied dialogue state. '
            'Return a JSON object with observation and state. '
            'The observation contains reaction, facts, new_topic, refused, user_question and exit. '
            'The state contains all fields of the supplied state with accumulated known facts, '
            'slot statuses, question attempts, topic, used follow-ups and dialogue flags. '
            'Preserve previous facts and refusals; do not invent unobserved information.'
        )
    if view != 'history':
        label = 'State before the latest user message' if view == 'extraction' else 'Current dialogue state'
        system += f'\n{label}: ' + json.dumps(state, ensure_ascii=False, sort_keys=True)
    return [{'role': 'system', 'content': system}] + [dict(message) for message in history]


def compile_row(row, view):
    """Keep message boundaries as structure; choose text formatting at runtime."""
    if view not in VIEWS:
        raise ValueError(f'Unsupported view: {view}')
    history_key = 'history_before' if view == 'extraction' else 'history'
    history = list(row[history_key])
    state = None if view == 'history' else row['state_before' if view == 'extraction' else 'state']
    if view == 'extraction':
        history.append({'role': 'user', 'content': row['utterance']})
        completion = json.dumps({'observation': row['target_observation'], 'state': row['target_state']},
                                ensure_ascii=False, sort_keys=True)
    elif view == 'policy':
        completion = serialize_action(row['target_action'])
    else:
        completion = row['target_response']
    example = {
        'episode_id': row['episode_id'],
        'turn': row['turn'],
        'messages': build_messages(history, view, state=state, action=row.get('action')),
        'completion': completion,
        'view': view,
    }
    if view == 'extraction':
        example['next_turn'] = row['next_turn']
    return example


def run(src, dest, view, json_dest=None):
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with src.open(encoding='utf-8') as source, dest.open('w', encoding='utf-8', newline='\n') as output:
        for line in source:
            if not line.strip():
                continue
            record = compile_row(json.loads(line), view)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            records.append(record)
    json_text = json.dumps(records, ensure_ascii=False, indent=2) + '\n'
    dest.with_suffix('.json').write_text(json_text, encoding='utf-8', newline='\n')
    if json_dest is not None:
        json_dest = Path(json_dest)
        json_dest.parent.mkdir(parents=True, exist_ok=True)
        json_dest.write_text(json_text, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--out', default='data/sft')
    parser.add_argument('--json-out', help='Export an additional JSON copy here; adjacent JSON files are always written')
    args = parser.parse_args()
    for split in SPLITS:
        for view in VIEWS:
            json_dest = Path(args.json_out) / view / f'{split}.json' if args.json_out else None
            run(Path(args.data) / view / f'{split}.jsonl',
                Path(args.out) / view / f'{split}.jsonl', view, json_dest)
    print('Structured SFT compilation complete.')
