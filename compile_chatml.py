"""Compile dataset views into ChatML supervised completion records (stdlib only).

The prompt is masked in SFT; compute loss on completion only. ChatML markers
are plain text here; ensure tokenizer supports them or add explicit special tokens.
"""
import argparse
import json
from pathlib import Path

PERSONA = 'You are a friendly, curious and respectful conversational assistant. Follow the user\'s lead. Do not repeat answered questions. Respect refusals and requests to end.'


def serialize_action(action):
    return action['type'] + (f"({action['target']})" if action.get('target') else '')


def messages_to_chatml(messages):
    return ''.join(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n" for msg in messages)


def compile_row(row, view):
    history = list(row['history'])
    if view == 'policy':
        system = PERSONA + ' Predict the next dialogue action, not a natural language reply.'
        completion = serialize_action(row['target_action'])
    elif view == 'response':
        system = PERSONA + ' Write the next assistant message for the action supplied by the user.'
        history.append({'role': 'user', 'content': f"[NEXT_ACTION] {serialize_action(row['action'])}"})
        completion = row['target_response']
    elif view == 'history':
        system = PERSONA
        completion = row['target_response']
    else:
        raise ValueError(f'Unsupported view: {view}')
    prompt = messages_to_chatml([{'role': 'system', 'content': system}] + history) + '<|im_start|>assistant\n'
    return {'episode_id': row['episode_id'], 'turn': row['turn'], 'prompt': prompt,
            'completion': completion + '<|im_end|>', 'view': view}


def run(src, dest, view):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding='utf-8') as f, dest.open('w', encoding='utf-8') as o:
        for line in f:
            o.write(json.dumps(compile_row(json.loads(line), view), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='data/chatml')
    args = p.parse_args()
    for split in ('train', 'validation', 'test'):
        for view in ('policy', 'response', 'history'):
            run(Path(args.data) / view / f'{split}.jsonl', Path(args.out) / view / f'{split}.jsonl', view)
    print('ChatML compilation complete.')
