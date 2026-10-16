"""Evaluate generated behaviour, independently of teacher-forced validation loss.

Examples (train checkpoints with trainer.py first):
  python benchmark.py --model runs/ubar --chain full --output runs/eval-ubar
  python benchmark.py --model runs/ubar --context predicted --output runs/session-ubar
  python benchmark.py --model runs/ubar --context predicted --user-source simulator --output runs/rollout-ubar
  python benchmark.py --model runs/dst --chain dst --output runs/eval-dst
  python benchmark.py --dst-model runs/dst --policy-model runs/policy \
      --response-model runs/response --chain full --output runs/eval-modular

Chain: response uses reference B,A; policy_response uses reference B; full predicts
B,A,R after each user message, starting from empty memory for the first message.
Context: reference resets history/memory at each transition; predicted carries
the model's memory, actions and replies forward (full chain only).
User source: dataset replays recorded messages; simulator supplies subsequent user
messages in response to predicted actions, not text (predicted context only).
Template metrics are finite-corpus proxies, not semantic scores for paraphrases.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random

from compile_sft import parse_action, serialize_action
from data import corpus_hash, read_episodes, segment
from dialogue_protocol import state_text
from generator import (ACKS, ASK_TEXT, CLOSES, FOLLOWUP, SLOTS, Action, State,
                       UserSimulator, commit_action, compile_views, observe,
                       select_action, write_jsonl)
from inference import Prediction, PredictorRouter, Request, parse_extraction, parse_state
from experiment_config import EVAL_CHAINS, EVAL_CONTEXTS, EVAL_USER_SOURCES, validate_evaluation
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


def uses_ubar(predictor, task):
    model = getattr(predictor, 'models', {}).get(task, predictor)
    return getattr(model, 'task', None) == 'ubar'


def predict(predictor, request, record, parser):
    task = request.task
    if uses_ubar(predictor, task):
        request = replace(request, history=[request.history[-1]],
                          state=request.previous_state if task == 'dst' else request.state)
    entry = {'input': deepcopy({'history': request.history, 'state': request.state, 'action': request.action})}
    if uses_ubar(predictor, task):
        entry['input'].update(previous_state=deepcopy(request.previous_state),
                              previous_action=deepcopy(request.previous_action), prefix=request.prefix)
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


def replay(episodes, predictor, chain, progress=None):
    records = []
    for episode_number, episode in enumerate(episodes):
        views = compile_views(episode)
        extractions = {row['next_turn']: row for row in views['extraction']}
        previous_state, previous_action = State().snapshot(), None
        for turn, policy_row in zip(episode['turns'], views['policy']):
            if progress:
                progress(completed_episodes=episode_number, total_episodes=len(episodes),
                         episode_id=episode['episode_id'], turn=turn['index'])
            index = turn['index']
            gold_state, gold_action = policy_row['state'], policy_row['target_action']
            history = policy_row['history']
            source = extractions[index]
            reaction = source['target_observation']['reaction']
            record = new_record(episode, index, reaction, gold_action)
            record['reference'] = {'state': gold_state, 'action': gold_action, 'response': turn['assistant']}
            state, action, response = gold_state, gold_action, None
            current = ''
            if chain in ('dst', 'extraction', 'full'):
                task = 'extraction' if chain == 'extraction' else 'dst'
                result = predict(predictor, Request(task, history, source['state_before'],
                                 previous_state=previous_state, previous_action=previous_action), record,
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
                # Preserve generated text exactly as chat does. Reference stages
                # use the canonical serialization only when that stage is bypassed.
                state_output = (record['stages']['dst']['text'] if 'dst' in record['stages']
                                else state_text(state))
                current += segment('dst', state_output)
            if chain in ('policy', 'policy_response', 'full'):
                if state is None:
                    blocked(record, 'policy')
                    action = None
                else:
                    action = predict(predictor, Request('policy', history, state,
                                     previous_state=previous_state, previous_action=previous_action,
                                     prefix=current), record, parse_action)
                score_action(record, action, gold_action, gold_state)
            if action is not None:
                action_output = (record['stages']['policy']['text'] if 'policy' in record['stages']
                                 else serialize_action(action))
                current += segment('policy', action_output)
            if chain in ('response', 'history', 'policy_response', 'full'):
                task = 'history' if chain == 'history' else 'response'
                if action is None or state is None:
                    blocked(record, task)
                else:
                    response = predict(predictor, Request(task, history,
                        None if task == 'history' else state, None if task == 'history' else action,
                        None if task == 'history' else previous_state,
                        None if task == 'history' else previous_action,
                        '' if task == 'history' else current), record, parse_response)
                score_response(record, response, gold_action, action, turn['assistant'])
            records.append(record)
            # Every replay transition starts from the recorded context, even after failures.
            previous_state, previous_action = gold_state, gold_action
    return records, []


def predict_chain(predictor, history, memory, previous_state, previous_action, record):
    """Generate B/A/R using only supplied model context; references are for scoring."""
    reference = record['reference']
    state = predict(predictor, Request('dst', history, memory,
                    previous_state=previous_state, previous_action=previous_action), record, parse_state)
    score_state(record, state, reference['state'])
    if state is None:
        blocked(record, 'policy')
        action = None
    else:
        current = segment('dst', record['stages']['dst']['text'])
        action = predict(predictor, Request('policy', history, state,
                         previous_state=previous_state, previous_action=previous_action,
                         prefix=current), record, parse_action)
    score_action(record, action, reference['action'], reference['state'])
    if action is None:
        blocked(record, 'response')
        response = None
    else:
        current += segment('policy', record['stages']['policy']['text'])
        response = predict(predictor, Request('response', history, state, action,
                           previous_state, previous_action, current), record, parse_response)
    score_response(record, response, reference['action'], action, reference.get('response'))
    return state, action, response


def session(episodes, predictor, progress=None):
    """Fixed recorded users, generated history; unavailable remaining turns count as failures."""
    records, outcomes = [], []
    for episode_number, episode in enumerate(episodes):
        memory = State()
        previous_state, previous_action = State().snapshot(), None
        history, episode_records = [], []
        terminal, generated_turns = None, 0
        for turn in episode['turns']:
            if progress:
                progress(completed_episodes=episode_number, total_episodes=len(episodes),
                         episode_id=episode['episode_id'], turn=turn['index'])
            expected = {key: turn['action'][key] for key in ('type', 'target')}
            record = new_record(episode, turn['index'], turn['observation']['reaction'], expected)
            record['reference'] = {'state': turn['state_before'], 'action': expected,
                                   'response': turn['assistant']}
            episode_records.append(record)
            if terminal is not None:
                # Keep the complete reference denominator; never restart from gold memory.
                for stage in ('dst', 'policy', 'response'):
                    blocked(record, stage)
                    record['stages'][stage]['failure'] = 'session_ended:' + terminal
                record['errors'].append('session.after_' + terminal)
                score_state(record, None, turn['state_before'])
                score_action(record, None, expected, turn['state_before'])
                score_response(record, None, expected, None, turn['assistant'])
                continue
            history.append({'role': 'user', 'content': turn['user']})
            state, action, response = predict_chain(
                predictor, history, memory.snapshot(), previous_state, previous_action, record)
            if response is None:
                terminal = 'invalid_generation'
                continue
            previous_state, previous_action = state, action
            history.append({'role': 'assistant', 'content': response})
            memory = State.from_snapshot(state)
            if not uses_ubar(predictor, 'dst'):
                commit_action(memory, Action(**action))
            generated_turns += 1
            if action['type'] == 'CLOSE':
                terminal = 'appropriate_close' if expected['type'] == 'CLOSE' else 'premature_close'
        records.extend(episode_records)
        outcomes.append({'episode_id': episode['episode_id'], 'turns': len(episode_records),
                         'generated_turns': generated_turns, 'terminal': terminal or 'reference_exhausted',
                         'policy_perfect': all(r['metrics']['action.exact'] == 1 for r in episode_records),
                         'response_template_perfect': all(r['metrics']['response.expected_action_template'] == 1
                                                          for r in episode_records)})
    return records, outcomes


def rollout(episodes, predictor, seed, max_turns, progress=None):
    records, outcomes = [], []
    for episode_number, episode in enumerate(episodes):
        truth, memory = State(), State()
        simulator = UserSimulator(random.Random(f'{seed}:{episode["episode_id"]}'), episode['hidden_user'])
        opening = episode['turns'][0]
        # Gold observation is evaluator-only; the model receives text and empty memory.
        observe(truth, opening['observation'])
        history = [{'role': 'user', 'content': opening['user']}]
        previous_state, previous_action = State().snapshot(), None
        reaction, terminal = opening['observation']['reaction'], 'horizon_limit'
        episode_records = []
        for index in range(max_turns):
            if progress:
                progress(completed_episodes=episode_number, total_episodes=len(episodes),
                         episode_id=episode['episode_id'], turn=index)
            expected = action_dict(select_action(truth))
            record = new_record(episode, index, reaction, expected)
            record['reference'] = {'state': truth.snapshot(), 'action': expected}
            episode_records.append(record)
            state, action, response = predict_chain(
                predictor, history, memory.snapshot(), previous_state, previous_action, record)
            if response is None:
                terminal = 'invalid_generation'
                break
            previous_state, previous_action = state, action
            history.append({'role': 'assistant', 'content': response})
            actual_action = Action(**action)
            memory = State.from_snapshot(state)
            if not uses_ubar(predictor, 'dst'):
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


def run_benchmark(episodes, predictor, *, chain='full', context='reference', user_source='dataset',
                  seed=42, max_turns=17, progress=None):
    validate_evaluation(chain, context, user_source)
    if max_turns <= 0 or not episodes:
        raise ValueError('Choose a positive horizon and nonempty episodes')
    if len({episode['episode_id'] for episode in episodes}) != len(episodes):
        raise ValueError('Duplicate evaluation episode IDs')
    for episode in episodes:
        turns = episode['turns']
        if not turns or any(not isinstance(turn.get('user'), str) or not turn['user'].strip() or
                            'state_before_user' not in turn or not turn.get('observation') for turn in turns):
            raise ValueError(f'{episode["episode_id"]}: benchmark requires user-first episodes; regenerate the dataset')
        if turns[0]['state_before_user'] != State().snapshot():
            raise ValueError(f'{episode["episode_id"]}: memory before the first user message must be empty')
    if user_source == 'simulator':
        records, outcomes = rollout(episodes, predictor, seed, max_turns, progress)
    elif context == 'predicted':
        records, outcomes = session(episodes, predictor, progress)
    else:
        records, outcomes = replay(episodes, predictor, chain, progress)
    if progress:
        progress(completed_episodes=len(episodes), total_episodes=len(episodes), episode_id=None, turn=None)
    errors, groups, requests = summarize(records)
    report = {'schema_version': 2, 'chain': chain, 'context': context, 'user_source': user_source,
              'seed': seed, 'max_turns': max_turns,
              'splits': sorted({ep['split'] for ep in episodes}), 'corpus_hash': corpus_hash(episodes),
              'episodes': len(episodes), 'evaluated_turns': len(records),
              'models': getattr(predictor, 'metadata', {}),
              'initial_state': 'empty_before_first_user',
              'metrics': aggregate(records),
              'errors': errors, 'groups': groups, 'data_review_candidates': requests,
              'rollout_outcomes': outcomes if user_source == 'simulator' else [],
              'limitations': ['Shared-template corpus; this does not measure unseen-language generalization.',
                             'Response template matches are proxies; unknown paraphrases need manual review.',
                             'Behaviour violation rates condition on valid actions; inspect stage coverage.',
                             'Data review candidates are diagnostics, not proof that more data will fix an error.']}
    if user_source == 'simulator':
        report['limitations'].append('Simulator reacts to predicted actions, not the semantic content of replies.')
    if context == 'predicted' and user_source == 'dataset':
        report['session_outcomes'] = outcomes
        report['limitations'].extend([
            'Recorded user messages do not adapt to generated questions; references describe the recorded trajectory.',
            'After invalid generation or CLOSE, remaining recorded turns are unavailable and score zero.',
            'All recorded turns are scored; max_turns applies only to simulator rollout.'])
    if outcomes:
        summary = {'termination_counts': dict(Counter(o['terminal'] for o in outcomes)),
                   'policy_perfect_episodes': sum(o['policy_perfect'] for o in outcomes),
                   'response_template_perfect_episodes': sum(o['response_template_perfect'] for o in outcomes)}
        if user_source == 'dataset':
            summary['generated_turns'] = sum(o['generated_turns'] for o in outcomes)
        report['rollout_summary' if user_source == 'simulator' else 'session_summary'] = summary
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data')
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--chain', choices=EVAL_CHAINS, default='full', help='Stages to evaluate')
    parser.add_argument('--context', choices=EVAL_CONTEXTS, default='reference',
                        help='Previous history/memory; predicted requires --chain full')
    parser.add_argument('--user-source', choices=EVAL_USER_SOURCES, default='dataset',
                        help='User messages; simulator requires --context predicted')
    parser.add_argument('--model', help='Default trained checkpoint for every requested stage')
    parser.add_argument('--dst-model')
    parser.add_argument('--policy-model')
    parser.add_argument('--response-model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=384, help='Upper bound; action/reply caps are 32/128')
    parser.add_argument('--max-turns', type=int, default=17, help='Simulator only; dataset evaluation scores all recorded turns')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-episodes', type=int)
    args = parser.parse_args()
    try:
        validate_evaluation(args.chain, args.context, args.user_source)
    except ValueError as error:
        parser.error(str(error))
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        parser.error('--limit-episodes must be positive')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty to preserve previous reports')
    predictor = PredictorRouter(args.model, dst=args.dst_model, policy=args.policy_model,
                                response=args.response_model, device=args.device,
                                local_files_only=args.local_files_only, max_new_tokens=args.max_new_tokens)
    predictor.validate_chain(args.chain)
    episodes = read_episodes(args.data, args.split)[:args.limit_episodes]
    report, records = run_benchmark(episodes, predictor, chain=args.chain, context=args.context,
                                    user_source=args.user_source, seed=args.seed, max_turns=args.max_turns)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_jsonl(output / 'predictions.jsonl', records)
    print(json.dumps({'report': str(output / 'report.json'), 'metrics': report['metrics']}, indent=2))


if __name__ == '__main__':
    main()
