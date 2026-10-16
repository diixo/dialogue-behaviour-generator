"""Shared dialogue wire format and output schema; standard library only.

Training, evaluation and standalone chat use the same serialization contract.
This module contains no dataset loading, generation rules or model dependencies.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import re
from typing import Optional

from dialogue_format import get_format

SLOTS = ('time_of_day', 'activity', 'mood')
VIEWS = ('policy', 'response', 'history', 'extraction')
SLOT_VALUES = {
    'time_of_day': ('morning', 'afternoon', 'evening', 'night'),
    'activity': ('reading', 'working', 'listening_to_music', 'relaxing'),
    'mood': ('calm', 'happy', 'tired'),
}
TOPIC_NAMES = ('books', 'work', 'music', 'relaxing', 'python')
SEGMENTS = {
    'dst': ('<|state|>', '<|end_state|>'),
    'policy': ('<|action|>', '<|end_action|>'),
    'response': ('<|response|>', '<|end_response|>'),
}
SPECIAL_TOKENS = [token for pair in SEGMENTS.values() for token in pair]

PERSONA = (
    "You are a friendly, curious and respectful conversational assistant. "
    "Follow the user's lead. Do not repeat answered questions. "
    "Respect refusals and requests to end."
)


@dataclass
class State:
    known: dict = field(default_factory=lambda: {k: None for k in SLOTS})
    statuses: dict = field(default_factory=lambda: {k: 'unknown' for k in SLOTS})
    asked: list = field(default_factory=list)
    pending: Optional[str] = None
    attempts: dict = field(default_factory=lambda: {k: 0 for k in SLOTS})
    topic: str = 'daily_routine'
    followup_used: set = field(default_factory=set)
    user_question_pending: bool = False
    wants_exit: bool = False
    phase: str = 'opening'
    last_action: Optional[str] = None

    @classmethod
    def from_snapshot(cls, snapshot):
        """Restore observable dialogue memory, including mutable collection types."""
        if set(snapshot) != set(cls.__dataclass_fields__):
            raise ValueError('Expected a complete dialogue state snapshot')
        values = deepcopy(snapshot)
        values['followup_used'] = set(values['followup_used'])
        return cls(**values)

    def snapshot(self):
        return {'known': dict(self.known),
                'statuses': dict(self.statuses),
                'asked': list(self.asked),
                'pending': self.pending,
                'attempts': dict(self.attempts),
                'topic':self.topic,
                'followup_used': sorted(self.followup_used),
                'user_question_pending': self.user_question_pending,
                'wants_exit': self.wants_exit,
                'phase': self.phase,
                'last_action': self.last_action}


def serialize_action(action):
    return action['type'] + (f"({action['target']})" if action.get('target') else '')


def parse_action(text):
    """Convert a decoded policy completion into the action consumed by response."""
    match = re.fullmatch(r'([A-Z_]+)(?:\(([a-z_]+)\))?', text.strip())
    if match is None:
        raise ValueError(f'Invalid dialogue action: {text!r}')
    kind, target = match.groups()
    slots = set(SLOTS)
    topics = set(TOPIC_NAMES)
    allowed = {'ASK': slots, 'CLARIFY': slots, 'FOLLOW_UP': topics, 'SWITCH': topics,
               'ANSWER': {None}, 'ACK': {None}, 'CLOSE': {None}}
    if kind not in allowed or target not in allowed[kind]:
        raise ValueError(f'Invalid dialogue action: {text!r}')
    return {'type': kind, 'target': target}


def build_messages(history, view, *, state=None, action=None):
    """Build training or inference inputs using observed/predicted state and action.

    For extraction, history includes the latest user message and state is the memory
    before observing that message (empty for the first user message, otherwise
    after committing the preceding assistant action).
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


def state_text(state):
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def task_messages(task, history, state=None, action=None):
    if task != 'dst':
        return build_messages(history, task, state=state, action=action)
    if state is None:
        raise ValueError('dst requires state before the latest user message')
    instruction = (
        'Update the dialogue state using the latest user message and conversation. '
        'Return only the complete state as a JSON object. Preserve previous facts and refusals; '
        'do not invent unobserved information. State before the latest user message: '
    )
    return [{'role': 'system', 'content': instruction + state_text(state)}] + list(history)


def ubar_header(style='plain'):
    fmt = get_format(style)
    instruction = PERSONA + ' Track dialogue state, choose an action, and write a response in that order.'
    return fmt.message_template.format(role='system', content=instruction)


def ubar_user(text, style='plain'):
    return get_format(style).message_template.format(role='user', content=text)


def segment(task, text):
    start, end = SEGMENTS[task]
    return start + text + end


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


REACTIONS = {'DIRECT_ANSWER', 'MULTI_FACT', 'REFUSAL', 'NON_ANSWER', 'COUNTER_QUESTION',
             'TOPIC_SHIFT', 'EXIT', 'ACKNOWLEDGEMENT', 'DIRECT_FOLLOWUP', 'NO_TURN',
             'GREETING', 'INFORM', 'QUESTION'}


def validate_state(state):
    def require(condition):
        if not condition:
            raise ValueError('Invalid dialogue state schema or value')
    require(isinstance(state, dict) and set(state) == set(State().snapshot()))
    for key in ('known', 'statuses', 'attempts'):
        require(isinstance(state[key], dict) and set(state[key]) == set(SLOTS))
    for slot in SLOTS:
        require(state['known'][slot] in [None] + list(SLOT_VALUES[slot]))
        require(state['statuses'][slot] in ('unknown', 'pending', 'known', 'declined', 'ambiguous', 'skipped'))
        require(type(state['attempts'][slot]) is int and state['attempts'][slot] >= 0)
    require(isinstance(state['asked'], list) and all(slot in SLOTS for slot in state['asked']))
    require(state['pending'] is None or state['pending'] in SLOTS)
    require(state['topic'] in ('daily_routine', *TOPIC_NAMES))
    require(isinstance(state['followup_used'], list) and
            all(topic in TOPIC_NAMES for topic in state['followup_used']))
    require(type(state['wants_exit']) is bool and type(state['user_question_pending']) is bool)
    require(state['phase'] in ('opening', 'exploration', 'closed'))
    require(state['last_action'] in (None, 'ASK', 'CLARIFY', 'FOLLOW_UP', 'SWITCH', 'ANSWER', 'ACK', 'CLOSE'))
    return state


def parse_state(text):
    return validate_state(json.loads(text))


def parse_extraction(text):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'observation', 'state'}:
        raise ValueError('Expected observation and state')
    validate_state(value['state'])
    obs = value['observation']
    if not isinstance(obs, dict) or set(obs) != {'reaction', 'facts', 'new_topic', 'refused', 'user_question', 'exit'}:
        raise ValueError('Invalid observation schema')
    if (obs['reaction'] not in REACTIONS or not isinstance(obs['facts'], dict) or
            obs['refused'] not in (None, *SLOTS) or obs['new_topic'] not in (None, *TOPIC_NAMES) or
            type(obs['exit']) is not bool or type(obs['user_question']) is not bool):
        raise ValueError('Invalid observation value')
    for slot, fact in obs['facts'].items():
        if slot not in SLOTS or fact not in list(SLOT_VALUES[slot]):
            raise ValueError('Invalid observed fact')
    return value
