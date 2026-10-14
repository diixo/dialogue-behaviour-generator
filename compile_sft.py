"""Export structured SFT examples without tokenizer or chat-format markers."""

import argparse
import json
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


def compile_row(row, view):
    """Keep message boundaries as structure; choose text formatting at runtime."""
    if view not in VIEWS:
        raise ValueError(f'Unsupported view: {view}')
    history_key = 'history_before' if view == 'extraction' else 'history'
    history = [dict(message) for message in row[history_key]]
    if view == 'policy':
        system = PERSONA + ' Predict the next dialogue action, not a natural language reply.'
        completion = serialize_action(row['target_action'])
    elif view == 'response':
        system = PERSONA + ' Write the next assistant message for this dialogue action: '
        system += serialize_action(row['action']) + '.'
        completion = row['target_response']
    elif view == 'history':
        system = PERSONA
        completion = row['target_response']
    else:
        system = (
            'Extract the observation from the latest user message using the conversation context. '
            'Return a JSON object with reaction, facts, new_topic, refused, user_question and exit.'
        )
        history.append({'role': 'user', 'content': row['utterance']})
        completion = json.dumps(row['target_observation'], ensure_ascii=False, sort_keys=True)
    return {
        'episode_id': row['episode_id'],
        'turn': row['turn'],
        'messages': [{'role': 'system', 'content': system}] + history,
        'completion': completion,
        'view': view,
    }


def run(src, dest, view, json_dest=None):
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    records = [] if json_dest is not None else None
    with src.open(encoding='utf-8') as source, dest.open('w', encoding='utf-8') as output:
        for line in source:
            if not line.strip():
                continue
            record = compile_row(json.loads(line), view)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            if records is not None:
                records.append(record)
    if json_dest is not None:
        json_dest = Path(json_dest)
        json_dest.parent.mkdir(parents=True, exist_ok=True)
        json_dest.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data')
    parser.add_argument('--out', default='data/sft')
    parser.add_argument('--json-out', help='Also export matching JSON arrays to this directory')
    args = parser.parse_args()
    for split in SPLITS:
        for view in VIEWS:
            json_dest = Path(args.json_out) / view / f'{split}.json' if args.json_out else None
            run(Path(args.data) / view / f'{split}.jsonl',
                Path(args.out) / view / f'{split}.jsonl', view, json_dest)
    print('Structured SFT compilation complete.')
