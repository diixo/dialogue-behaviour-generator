"""Export structured SFT examples without tokenizer or chat-format markers."""

import argparse
import json
from pathlib import Path

from dialogue_protocol import PERSONA, VIEWS, build_messages, parse_action, serialize_action

SPLITS = ('train', 'validation', 'test')


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
