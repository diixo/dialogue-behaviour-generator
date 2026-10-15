"""Evaluate generated behaviour, independently of teacher-forced validation loss.

Examples (train checkpoints with trainer.py first):
  python benchmark.py --model runs/ubar --mode full --output runs/eval-ubar
  python benchmark.py --model runs/ubar --mode rollout --output runs/rollout-ubar
  python benchmark.py --model runs/dst --mode dst --output runs/eval-dst
  python benchmark.py --dst-model runs/dst --policy-model runs/policy \
      --response-model runs/response --mode full --output runs/eval-modular

Replay uses reference history and resets memory at each transition. response uses
reference B,A; policy_response uses reference B; full predicts B,A,R after each
user message. Initial B is supplied in all chains. rollout carries predicted
memory, actions and replies forward; its simulator responds to actions, not text.
Template metrics are finite-corpus proxies, not semantic scores for paraphrases.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random

from compile_sft import parse_action, serialize_action
from data import corpus_hash, read_episodes, segment, state_text, ubar_user
from generator import (ACKS, ASK_TEXT, CLOSES, FOLLOWUP, SLOTS, Action, State,
                       UserSimulator, commit_action, compile_views, observe,
                       select_action, write_jsonl)
from inference import Prediction, PredictorRouter, Request, parse_extraction, parse_state

MODES = ('dst', 'extraction', 'policy', 'response', 'history', 'policy_response', 'full', 'rollout')
ANSWERS = ('I do not have a time of day myself, but I would be happy to chat.',
           'I am here to help and chat with you.')


def normalize(text):
    return ' '.join(text.casefold().split())


def response_variants(action):
    kind, target = action['type'], action.get('target')
    if kind in ('ASK', 'CLARIFY'):
        return ASK_TEXT[target]
    if kind in ('FOLLOW_UP', 'SWITCH'):
        return FOLLOWUP[target]
    return {'ACK': ACKS, 'CLOSE': CLOSES, 'ANSWER': ANSWERS}[kind]


KNOWN_RESPONSES = {normalize(text) for variants in [*ASK_TEXT.values(), *FOLLOWUP.values(),
                                                   ACKS, CLOSES, ANSWERS] for text in variants}


def action_dict(action):
    return {'type': action.type, 'target': action.target}


def new_record(episode, turn, reaction, expected_action):
    return {'episode_id': episode['episode_id'], 'turn': turn,
            'tags': {'reaction': reaction, 'action': expected_action['type'],
                     'target': expected_action.get('target') or 'none',
                     'depth': '0-2' if turn < 3 else '3-5' if turn < 6 else '6+'},
            'stages': {}, 'metrics': {}, 'errors': []}


def predict(predictor, request, record, parser):
    task = request.task
    entry = {'input': {'history': request.history, 'state': request.state, 'action': request.action}}
    record['stages'][task] = entry
    metrics = record['metrics']
    metrics[f'{task}.available'] = 1
    metrics[f'{task}.stopped'] = metrics[f'{task}.valid'] = 0
    try:
        result = predictor(deepcopy(request))
        if not isinstance(result, Prediction):
            raise TypeError('Predictor must return Prediction')
        entry.update(text=result.text, tokens=result.tokens, stopped=result.stopped)
        metrics[f'{task}.stopped'] = int(result.stopped)
        if not result.stopped:
            entry['failure'] = 'missing_end_token'
            record['errors'].append(f'{task}.missing_end_token')
            return None
        if result.error:
            raise ValueError(result.error)
        value = parser(result.text)
        entry['parsed'] = value
        metrics[f'{task}.valid'] = 1
        return value
    except (ValueError, TypeError, KeyError) as error:
        entry['failure'] = str(error)
        category = 'invalid_output' if 'text' in entry else 'generation_error'
        record['errors'].append(f'{task}.{category}')
        return None


def blocked(record, task):
    record['stages'][task] = {'failure': 'blocked_by_previous_stage'}
    for metric in ('available', 'stopped', 'valid'):
        record['metrics'][f'{task}.{metric}'] = 0
    record['errors'].append(f'{task}.blocked')


def score_state(record, predicted, reference):
    record['metrics']['state.exact'] = int(predicted == reference)
    for field, expected in reference.items():
        correct = predicted is not None and predicted[field] == expected
        record['metrics'][f'state.{field}'] = int(correct)
        if predicted is not None and not correct:
            record['errors'].append(f'state.{field}')
    for slot in SLOTS:
        for field in ('known', 'statuses'):
            correct = predicted is not None and predicted[field][slot] == reference[field][slot]
            record['metrics'][f'slot.{slot}.{field}'] = int(correct)
            if predicted is not None and not correct:
                record['errors'].append(f'slot.{slot}.{field}')


def score_action(record, predicted, reference, truth):
    record['metrics']['action.exact'] = int(predicted == reference)
    if predicted is None:
        return
    kind, target = predicted['type'], predicted['target']
    asks = kind in ('ASK', 'CLARIFY')
    violations = {
        'repeat_known': asks and truth['known'][target] is not None,
        'repeat_refused': asks and truth['statuses'][target] == 'declined',
        'attempt_limit': asks and truth['attempts'][target] >= 2,
        'ignored_exit': truth['wants_exit'] and kind != 'CLOSE',
        'ignored_question': truth['user_question_pending'] and not truth['wants_exit'] and kind != 'ANSWER',
        'premature_close': kind == 'CLOSE' and reference['type'] != 'CLOSE',
        'wrong_followup': kind in ('FOLLOW_UP', 'SWITCH') and (
            target != truth['topic'] or target in truth['followup_used']),
    }
    for name, value in violations.items():
        # Violation rates are conditional on a parseable action, alongside action coverage.
        record['metrics'][f'behaviour.{name}'] = int(bool(value))
        if value:
            record['errors'].append(f'behaviour.{name}')
    if predicted != reference:
        record['errors'].append('action.mismatch')


def parse_response(text):
    if not text.strip():
        raise ValueError('Empty response')
    return text.strip()


def score_response(record, text, expected_action, conditioned_action, reference=None):
    normalized = normalize(text) if text is not None else None
    metrics = record['metrics']
    metrics['response.known_template'] = int(normalized in KNOWN_RESPONSES)
    for name, action in [('expected_action_template', expected_action),
                         ('conditioned_action_template', conditioned_action)]:
        metrics[f'response.{name}'] = int(action is not None and normalized in {
            normalize(variant) for variant in response_variants(action)})
    if reference is not None:
        metrics['response.reference_exact'] = int(normalized == normalize(reference))
    if text is not None:
        if normalized not in KNOWN_RESPONSES:
            record['errors'].append('response.unrecognized_template_review')
        elif not metrics['response.expected_action_template']:
            record['errors'].append('response.wrong_action_template')


def replay(episodes, predictor, mode, style):
    records = []
    for episode in episodes:
        views = compile_views(episode)
        extractions = {row['next_turn']: row for row in views['extraction']}
        previous_blocks = []
        for turn, policy_row in zip(episode['turns'], views['policy']):
            index = turn['index']
            gold_state, gold_action = policy_row['state'], policy_row['target_action']
            history = policy_row['history']
            user_block = '' if index == 0 else ubar_user(episode['turns'][index - 1]['user'], style)
            gold_block = (user_block + segment('dst', state_text(gold_state)) +
                          segment('policy', serialize_action(gold_action)) + segment('response', turn['assistant']))
            if mode in ('dst', 'extraction') and index == 0:
                previous_blocks.append(gold_block)
                continue
            source = extractions.get(index)
            reaction = source['target_observation']['reaction'] if source else 'OPENING'
            record = new_record(episode, index, reaction, gold_action)
            record['reference'] = {'state': gold_state, 'action': gold_action, 'response': turn['assistant']}
            state, action, response = gold_state, gold_action, None
            current = user_block
            if mode in ('dst', 'extraction', 'full') and source is not None:
                task = 'extraction' if mode == 'extraction' else 'dst'
                result = predict(predictor, Request(task, history, source['state_before'],
                                 session=previous_blocks + [current]), record,
                                 parse_extraction if task == 'extraction' else parse_state)
                state = result['state'] if result is not None and task == 'extraction' else result
                score_state(record, state, gold_state)
                if task == 'extraction':
                    obs = result['observation'] if result else None
                    record['reference']['observation'] = source['target_observation']
                    record['metrics']['observation.exact'] = int(obs == source['target_observation'])
                    for field in source['target_observation']:
                        record['metrics'][f'observation.{field}'] = int(
                            obs is not None and obs[field] == source['target_observation'][field])
                    record['metrics']['extraction.consistent'] = int(result is not None and
                        observe(State.from_snapshot(source['state_before']), obs).snapshot() == state)
                    if obs is not None and obs != source['target_observation']:
                        record['errors'].append('observation.mismatch')
            if state is not None:
                current += segment('dst', state_text(state))
            if mode in ('policy', 'policy_response', 'full'):
                if state is None:
                    blocked(record, 'policy')
                    action = None
                else:
                    action = predict(predictor, Request('policy', history, state,
                                     session=previous_blocks + [current]), record, parse_action)
                score_action(record, action, gold_action, gold_state)
            if action is not None:
                current += segment('policy', serialize_action(action))
            if mode in ('response', 'history', 'policy_response', 'full'):
                task = 'history' if mode == 'history' else 'response'
                if action is None or state is None:
                    blocked(record, task)
                else:
                    response = predict(predictor, Request(task, history,
                        None if task == 'history' else state, None if task == 'history' else action,
                        [] if task == 'history' else previous_blocks + [current]), record, parse_response)
                score_response(record, response, gold_action, action, turn['assistant'])
            records.append(record)
            # Every replay transition starts from the recorded context, even after failures.
            previous_blocks.append(gold_block)
    return records, []


def rollout(episodes, predictor, style, seed, max_turns):
    records, outcomes = [], []
    for episode in episodes:
        truth, memory = State(), State()
        simulator = UserSimulator(random.Random(f'{seed}:{episode["episode_id"]}'), episode['hidden_user'])
        history, previous_blocks, current = [], [], ''
        reaction, terminal = 'OPENING', 'horizon_limit'
        episode_records = []
        for index in range(max_turns):
            expected = action_dict(select_action(truth))
            record = new_record(episode, index, reaction, expected)
            record['reference'] = {'state': truth.snapshot(), 'action': expected}
            episode_records.append(record)
            state = memory.snapshot()
            if index:
                state = predict(predictor, Request('dst', history, state,
                                session=previous_blocks + [current]), record, parse_state)
                score_state(record, state, truth.snapshot())
            if state is None:
                blocked(record, 'policy')
                action = None
            else:
                memory = State.from_snapshot(state)
                current += segment('dst', state_text(state))
                action = predict(predictor, Request('policy', history, state,
                                 session=previous_blocks + [current]), record, parse_action)
            score_action(record, action, expected, truth.snapshot())
            if action is None:
                blocked(record, 'response')
                response = None
            else:
                current += segment('policy', serialize_action(action))
                response = predict(predictor, Request('response', history, state, action,
                                   previous_blocks + [current]), record, parse_response)
            score_response(record, response, expected, action)
            if response is None:
                terminal = 'invalid_generation'
                break
            previous_blocks.append(current + segment('response', response))
            history.append({'role': 'assistant', 'content': response})
            actual_action = Action(**action)
            commit_action(memory, actual_action)
            commit_action(truth, actual_action)
            if actual_action.type == 'CLOSE':
                terminal = 'appropriate_close' if expected['type'] == 'CLOSE' else 'premature_close'
                break
            user, observation = simulator.response(actual_action, truth)
            observe(truth, observation)
            history.append({'role': 'user', 'content': user})
            record['environment'] = {'user': user, 'observation': observation}
            reaction = observation['reaction']
            current = ubar_user(user, style)
        records.extend(episode_records)
        outcomes.append({'episode_id': episode['episode_id'], 'turns': len(episode_records),
                         'terminal': terminal,
                         'policy_perfect': all(r['metrics'].get('action.exact') == 1 for r in episode_records),
                         'response_template_perfect': all(r['metrics'].get(
                             'response.expected_action_template') == 1 for r in episode_records)})
    return records, outcomes


def aggregate(records):
    values = defaultdict(list)
    for record in records:
        for key, value in record['metrics'].items():
            values[key].append(value)
    return {key: {'value': sum(items) / len(items), 'numerator': sum(items), 'denominator': len(items)}
            for key, items in sorted(values.items())}


def summarize(records):
    groups = defaultdict(list)
    errors = Counter()
    suggestions = Counter()
    for record in records:
        errors.update(set(record['errors']))
        for tag, value in record['tags'].items():
            groups[f'{tag}={value}'].append(record)
        for error in set(record['errors']):
            if not error.endswith('.blocked'):
                suggestions[(error, record['tags']['reaction'], record['tags']['target'])] += 1
    requests = [{'error': error, 'reaction': reaction, 'target': target, 'count': count,
                 'next_step': 'Review paraphrase manually before labelling it an error' if
                 error == 'response.unrecognized_template_review' else
                 'Inspect failing examples; consider targeted training examples after checking labels and code'}
                for (error, reaction, target), count in suggestions.most_common()]
    return dict(errors.most_common()), {
        tag: {'examples': len(items), 'metrics': aggregate(items)} for tag, items in sorted(groups.items())
    }, requests


def run_benchmark(episodes, predictor, *, mode='full', seed=42, max_turns=17):
    if mode not in MODES or max_turns <= 0 or not episodes:
        raise ValueError('Choose a valid mode, positive horizon and nonempty episodes')
    if len({episode['episode_id'] for episode in episodes}) != len(episodes):
        raise ValueError('Duplicate evaluation episode IDs')
    style = getattr(predictor, 'style', 'plain')
    if mode == 'rollout':
        records, outcomes = rollout(episodes, predictor, style, seed, max_turns)
    else:
        records, outcomes = replay(episodes, predictor, mode, style)
    errors, groups, requests = summarize(records)
    report = {'schema_version': 1, 'mode': mode, 'seed': seed, 'max_turns': max_turns,
              'splits': sorted({ep['split'] for ep in episodes}), 'corpus_hash': corpus_hash(episodes),
              'episodes': len(episodes), 'evaluated_turns': len(records),
              'models': getattr(predictor, 'metadata', {}),
              'context': 'predicted_history_and_memory' if mode == 'rollout' else 'reference_history_per_transition',
              'initial_state': 'provided', 'metrics': aggregate(records),
              'errors': errors, 'groups': groups, 'data_review_candidates': requests,
              'rollout_outcomes': outcomes,
              'limitations': ['Shared-template corpus; this does not measure unseen-language generalization.',
                             'Response template matches are proxies; unknown paraphrases need manual review.',
                             'Simulator reacts to predicted actions, not the semantic content of replies.',
                             'Behaviour violation rates condition on valid actions; inspect stage coverage.',
                             'Data review candidates are diagnostics, not proof that more data will fix an error.']}
    if outcomes:
        report['rollout_summary'] = {'termination_counts': dict(Counter(o['terminal'] for o in outcomes)),
                                    'policy_perfect_episodes': sum(o['policy_perfect'] for o in outcomes),
                                    'response_template_perfect_episodes': sum(
                                        o['response_template_perfect'] for o in outcomes)}
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data')
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--mode', choices=MODES, default='full')
    parser.add_argument('--model', help='Default trained checkpoint for every requested stage')
    parser.add_argument('--dst-model')
    parser.add_argument('--policy-model')
    parser.add_argument('--response-model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=384, help='Upper bound; action/reply caps are 32/128')
    parser.add_argument('--max-turns', type=int, default=17)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-episodes', type=int)
    args = parser.parse_args()
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        parser.error('--limit-episodes must be positive')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty to preserve previous reports')
    predictor = PredictorRouter(args.model, dst=args.dst_model, policy=args.policy_model,
                                response=args.response_model, device=args.device,
                                local_files_only=args.local_files_only, max_new_tokens=args.max_new_tokens)
    predictor.validate_mode(args.mode)
    episodes = read_episodes(args.data, args.split)[:args.limit_episodes]
    report, records = run_benchmark(episodes, predictor, mode=args.mode, seed=args.seed, max_turns=args.max_turns)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_jsonl(output / 'predictions.jsonl', records)
    print(json.dumps({'report': str(output / 'report.json'), 'metrics': report['metrics']}, indent=2))


if __name__ == '__main__':
    main()
