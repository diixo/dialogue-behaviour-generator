"""Dialogue Behaviour Generator v0.3 — standalone, standard-library only.

Usage: python generator.py --out output --train 2000 --validation 200 --test 200 --seed 42
All train/validation/test episodes are generated separately. Current template families
are shared across splits; do not interpret test as unseen-template generalization.
"""
from __future__ import annotations
import argparse
import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SLOTS = ('time_of_day', 'activity', 'mood')

USER_VALUES = {
    'time_of_day': [
        ('morning', ['It is morning here.', 'It is still morning.']),
        ('afternoon', ['It is afternoon.', 'It is the afternoon here.']),
        ('evening', ['It is evening.', 'It is evening here.'])
        ],
    'activity': [
        ('reading', ['I am reading a book.', 'I am reading.']),
        ('working', ['I am working on a project.', 'Just working.']),
        ('listening_to_music', ['I am listening to music.', 'Just listening to some music.']),
        ('relaxing', ['I am relaxing.', 'Just taking it easy.'])
        ],
    'mood': [
        ('calm', ['I feel calm.', 'Pretty calm.']),
        ('happy', ['I feel happy.', 'I am in a good mood.']),
        ('tired', ['I feel tired.', 'A little tired.'])
        ],
}
TOPICS = {'reading':'books', 'working':'work', 'listening_to_music':'music', 'relaxing':'relaxing'}
ASK_TEXT = {
    'time_of_day': ['What time of day is it for you?', 'Is it morning, afternoon, or evening there?'],
    'activity': ['What are you doing right now?', 'What are you up to?'],
    'mood': ['How are you feeling?', 'What kind of mood are you in?'],
}
FOLLOWUP = {
    'books': ['What sort of book are you reading?', 'What is the book about?'],
    'work': ['What kind of project is it?', 'What are you working on?'],
    'music': ['What kind of music are you listening to?', 'What are you listening to?'],
    'relaxing': ['What helps you unwind?', 'How do you usually relax?'],
    'python': ['What are you building in Python?', 'What part of Python are you learning?'],
}
USER_FOLLOWUP = {
    'books': ['It is a mystery novel.', 'It is about a detective.'],
    'work': ['It is a small software project.', 'I am fixing a bug.'],
    'music': ['Mostly jazz.', 'I have some ambient music on.'],
    'relaxing': ['I like to rest quietly.', 'Usually a cup of tea.'],
    'python': ['A little chatbot.', 'I am learning functions.'],
}
ACKS = ['That sounds nice.', 'Thanks for sharing.', 'I see.']
CLOSES = ['Enjoy the rest of your day!', 'Thanks for chatting. Take care!', 'Have a good one!']

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
        return {'known': dict(self.known), 'statuses': dict(self.statuses),
                'asked': list(self.asked), 'pending': self.pending,
                'attempts': dict(self.attempts), 'topic':self.topic,
                'followup_used': sorted(self.followup_used),
                'user_question_pending': self.user_question_pending,
                'wants_exit': self.wants_exit, 'phase': self.phase,
                'last_action': self.last_action}

@dataclass(frozen=True)
class Action:
    type: str
    target: Optional[str] = None
    reason: str = ''
    def json(self): return {'type':self.type,'target':self.target,'reason':self.reason}

class ResponseRealizer:
    def realize(self, action: Action, state: State, rng: random.Random):
        kind, target = action.type, action.target
        if kind == 'ASK': return rng.choice(ASK_TEXT[target])
        if kind == 'FOLLOW_UP': return rng.choice(FOLLOWUP[target])
        if kind == 'SWITCH': return rng.choice(FOLLOWUP[target])
        if kind == 'ANSWER': return rng.choice(['I do not have a time of day myself, but I would be happy to chat.', 'I am here to help and chat with you.'])
        if kind == 'ACK': return rng.choice(ACKS)
        if kind == 'CLOSE': return rng.choice(CLOSES)
        if kind == 'CLARIFY': return rng.choice(ASK_TEXT[target])
        raise ValueError(kind)

class UserSimulator:
    def __init__(self, rng, hidden): self.rng, self.hidden = rng, hidden

    def opening(self):
        """An unsolicited first message; only disclosed facts become observable."""
        typ = self.weighted([('GREETING', .15), ('INFORM', .30), ('MULTI_FACT', .30),
                             ('TOPIC_SHIFT', .10), ('QUESTION', .07), ('REFUSAL', .04), ('EXIT', .04)])
        if typ == 'GREETING':
            return self.rng.choice(['Hi!', 'Hello!', 'Hi, can we chat?']), self.obs(typ)
        if typ in ('INFORM', 'MULTI_FACT'):
            slots = self.rng.sample(SLOTS, 1 if typ == 'INFORM' else self.rng.choice([2, 3]))
            return ' '.join(self.utterance(slot) for slot in slots), self.obs(
                typ, facts={slot: self.hidden[slot] for slot in slots})
        if typ == 'TOPIC_SHIFT':
            return 'Hi, I am learning Python.', self.obs(typ, new_topic='python')
        if typ == 'QUESTION':
            return 'Hello! How are you?', self.obs(typ, user_question=True)
        if typ == 'REFUSAL':
            slot = self.rng.choice(SLOTS)
            topic = {'time_of_day': 'the time of day', 'activity': 'what I am doing', 'mood': 'my mood'}[slot]
            return f'Hi. I would rather not talk about {topic}.', self.obs(typ, refused=slot)
        return 'Sorry, I have to go now.', self.obs(typ, exit=True)

    def response(self, action, state):
        r=self.rng
        if action.type == 'CLOSE': return None, {'reaction':'NO_TURN','facts':{},'new_topic':None,'refused':None,'user_question':False,'exit':False}
        if action.type == 'ANSWER':
            return r.choice(['Thanks for answering!', 'I see, thanks.']), self.obs('ACKNOWLEDGEMENT')
        if action.type in ('FOLLOW_UP','SWITCH'):
            options=[('DIRECT_FOLLOWUP',0.67),('TOPIC_SHIFT',0.12),('EXIT',0.12),('COUNTER_QUESTION',0.09)]
            typ=self.weighted(options)
            if typ=='DIRECT_FOLLOWUP': return r.choice(USER_FOLLOWUP[action.target]),self.obs('DIRECT_FOLLOWUP')
            if typ=='TOPIC_SHIFT': return 'Actually, I started learning Python.',self.obs('TOPIC_SHIFT',new_topic='python')
            if typ=='COUNTER_QUESTION': return 'And what about you?',self.obs('COUNTER_QUESTION',user_question=True)
            return 'I have to go now.',self.obs('EXIT',exit=True)
        if action.type=='ACK':
            typ=self.weighted([('ACKNOWLEDGEMENT',0.45),('TOPIC_SHIFT',0.2),('EXIT',0.35)])
            if typ=='ACKNOWLEDGEMENT': return r.choice(['Thanks!', 'Yeah.', 'Okay.']),self.obs(typ)
            if typ=='TOPIC_SHIFT': return 'By the way, I am learning Python.',self.obs(typ,new_topic='python')
            return 'I need to run now.',self.obs(typ,exit=True)
        slot=action.target
        typ=self.weighted([('DIRECT_ANSWER',0.44),('MULTI_FACT',0.17),('REFUSAL',0.10),('NON_ANSWER',0.09),('COUNTER_QUESTION',0.07),('TOPIC_SHIFT',0.07),('EXIT',0.06)])
        if action.type == 'CLARIFY': typ=self.weighted([('DIRECT_ANSWER',0.7),('REFUSAL',0.2),('NON_ANSWER',0.1)])
        if typ=='DIRECT_ANSWER':
            return self.utterance(slot),self.obs(typ, facts={slot:self.hidden[slot]})
        if typ=='MULTI_FACT':
            others=[s for s in SLOTS if s!=slot and state.statuses[s]=='unknown']
            if not others: return self.utterance(slot),self.obs('DIRECT_ANSWER',facts={slot:self.hidden[slot]})
            second=r.choice(others)
            return self.utterance(slot)+' '+self.utterance(second),self.obs(typ,facts={slot:self.hidden[slot],second:self.hidden[second]})
        if typ=='REFUSAL': return 'I would rather not say.',self.obs(typ,refused=slot)
        if typ=='NON_ANSWER': return 'Hmm, good question.',self.obs(typ)
        if typ=='COUNTER_QUESTION': return 'What about you?',self.obs(typ,user_question=True)
        if typ=='TOPIC_SHIFT': return 'Actually, I have been learning Python.',self.obs(typ,new_topic='python')
        return 'Sorry, I have to go.',self.obs('EXIT',exit=True)
    def utterance(self, slot):
        for value, variants in USER_VALUES[slot]:
            if value==self.hidden[slot]: return self.rng.choice(variants)
        raise KeyError(slot)
    def weighted(self, options): return self.rng.choices([x for x,_ in options],weights=[w for _,w in options])[0]
    @staticmethod
    def obs(reaction, facts=None, new_topic=None, refused=None, user_question=False, exit=False):
        return {'reaction':reaction,'facts':facts or {},'new_topic':new_topic,'refused':refused,'user_question':user_question,'exit':exit}

def observe(state, obs):
    # Clear the pending question after any user response: unanswered stays unknown;
    # one non-answer permits one bounded clarification.
    previous=state.pending
    state.pending=None
    if previous is not None and previous not in obs['facts'] and obs['reaction']=='NON_ANSWER':
        state.statuses[previous]='skipped' if state.attempts[previous]>=2 else 'ambiguous'
    elif previous is not None and previous not in obs['facts'] and state.statuses[previous]=='pending':
        state.statuses[previous]='skipped'
    if obs['refused'] is not None:
        state.statuses[obs['refused']]='declined'
    for slot,value in obs['facts'].items():
        state.known[slot]=value
        state.statuses[slot]='known'
        if slot=='activity': state.topic=TOPICS[value]
    if obs['new_topic']: state.topic=obs['new_topic']
    state.user_question_pending=obs['user_question']
    state.wants_exit=obs['exit']
    return state

def select_action(state: State):
    if state.wants_exit: return Action('CLOSE',reason='user_exit')
    if state.user_question_pending: return Action('ANSWER',reason='user_question')
    # Once a user has introduced a topic, make at most one follow-up in this MVP.
    if state.topic in FOLLOWUP and state.topic not in state.followup_used:
        return Action('FOLLOW_UP',target=state.topic,reason='topic_engagement')
    for slot in SLOTS:
        if state.statuses[slot]=='ambiguous' and state.attempts[slot] < 2:
            return Action('CLARIFY',slot,'ambiguous_answer')
    for slot in SLOTS:
        if state.statuses[slot]=='unknown' and state.attempts[slot] < 2:
            return Action('ASK',slot,'unknown_slot')
    if state.last_action not in ('ACK','CLOSE'):
        return Action('ACK',reason='acknowledge_and_close')
    return Action('CLOSE',reason='goals_complete_or_exhausted')

def commit_action(state, action):
    state.last_action=action.type
    for slot in SLOTS:
        if state.statuses[slot]=='ambiguous' and state.attempts[slot]>=2:
            state.statuses[slot]='skipped'
    if action.type in ('ASK','CLARIFY'):
        state.pending=action.target
        state.asked.append(action.target)
        state.attempts[action.target]+=1
        state.statuses[action.target]='pending'
    elif action.type in ('FOLLOW_UP','SWITCH'):
        state.followup_used.add(action.target)
    elif action.type == 'ANSWER':
        state.user_question_pending=False
    elif action.type == 'CLOSE':
        state.phase='closed'
    else:
        state.phase='exploration'

def episode_messages(turns):
    messages=[]
    for turn in turns:
        messages.append({'role':'user','content':turn['user']})
        messages.append({'role':'assistant','content':turn['assistant']})
    return messages

def generate_episode(rng, eid, split, max_turns=16):
    if max_turns <= 0:
        raise ValueError('max_turns must be positive')
    hidden={k:rng.choice([v for v,_ in USER_VALUES[k]]) for k in SLOTS}
    state=State()
    sim=UserSimulator(rng,hidden)
    realizer=ResponseRealizer()
    turns=[]
    user, observation = sim.opening()
    for i in range(max_turns + 1):
        before_user = state.snapshot()
        observe(state, observation)
        action=select_action(state)
        if i == max_turns and action.type != 'CLOSE':
            action = Action('CLOSE', reason='horizon_limit')
        state_before=state.snapshot()
        assistant=realizer.realize(action,state,rng)
        commit_action(state,action)
        turns.append({'index':i, 'state_before_user':before_user,
                      'state_before':state_before, 'action':action.json(),
                      'assistant':assistant, 'user':user, 'observation':observation,
                      'state_after':state.snapshot()})
        if action.type=='CLOSE':
            break
        user, observation = sim.response(action, state)
    return {'schema_version':2, 'dialogue_start':'user',
            'episode_id':eid,'split':split,'persona':'friendly_curious',
            'scenario':'daily_smalltalk','hidden_user':hidden,'turns':turns,
            'messages':episode_messages(turns),
            'terminal_reason':turns[-1]['action']['reason']}

def compile_views(ep):
    policy=[]; response=[]; history_only=[]; extraction=[]
    history=[]
    for turn in ep['turns']:
        extraction.append({'episode_id':ep['episode_id'], 'turn':turn['index'],
                           'next_turn':turn['index'],
                           'history_before':list(history), 'utterance':turn['user'],
                           'state_before':deepcopy(turn['state_before_user']),
                           'target_observation':deepcopy(turn['observation']),
                           'target_state':deepcopy(turn['state_before'])})
        history.append({'role':'user', 'content':turn['user']})
        a=turn['action']
        common={'episode_id':ep['episode_id'],'turn':turn['index'],'persona':ep['persona'],
                'history':list(history)}
        policy.append({**common,'state':deepcopy(turn['state_before']),
                       'target_action':{'type':a['type'],'target':a['target']}})
        response.append({**common,'state':deepcopy(turn['state_before']),
                         'action':{'type':a['type'],'target':a['target']},
                         'target_response':turn['assistant']})
        history_only.append({**common,'target_response':turn['assistant']})
        history.append({'role':'assistant','content':turn['assistant']})
    return {'policy':policy,'response':response,'history':history_only,'extraction':extraction}

def validate_episode(ep):
    assert ep['turns'] and ep['turns'][-1]['action']['type']=='CLOSE'
    state = State()
    for index, turn in enumerate(ep['turns']):
        assert turn['index'] == index and turn['user']
        assert turn['state_before_user'] == state.snapshot()
        assert all(ep['hidden_user'][slot] == value for slot, value in turn['observation']['facts'].items())
        observe(state, turn['observation'])
        assert turn['state_before'] == state.snapshot()
        action = Action(**turn['action'])
        if action.reason != 'horizon_limit':
            assert action == select_action(state)
        commit_action(state, action)
        assert turn['state_after'] == state.snapshot()
        assert max(state.attempts.values()) <= 2
        assert action.type != 'CLOSE' or index == len(ep['turns']) - 1
    assert ep['messages']==episode_messages(ep['turns'])

def write_jsonl(path, records):
    """Write JSONL and an adjacent, human-readable JSON array."""
    path = Path(path)
    records = list(records)
    with path.open('w',encoding='utf-8',newline='\n') as f:
        for rec in records: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    with path.with_suffix('.json').open('w',encoding='utf-8',newline='\n') as f:
        json.dump(records,f,ensure_ascii=False,indent=2)
        f.write('\n')

def generate_all(out,counts,seed):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    manifest={'version':'0.3.0','dialogue_start':'user','seed':seed,'counts':counts,'notes':[
        'No pretrained model used; template-based oracle simulation.',
        'Current splits share template families; evaluate unseen templates separately.',
        'Policy labels are supervised only, not proof of autonomous conversational ability.'
    ]}
    for split,n in counts.items():
        rng=random.Random(f'{seed}:{split}')
        eps=[generate_episode(rng,f'{split}_{i:06d}',split) for i in range(n)]
        for ep in eps: validate_episode(ep)
        write_jsonl(root/f'{split}.episodes.jsonl',eps)
        views={'policy':[],'response':[],'history':[],'extraction':[]}
        for ep in eps:
            generated=compile_views(ep)
            for key in views: views[key].extend(generated[key])
        for k,records in views.items():
            sub=root/k;sub.mkdir(exist_ok=True)
            write_jsonl(sub/f'{split}.jsonl',records)
        manifest[split+'_rows']={k:len(v) for k,v in views.items()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='output')
    p.add_argument('--train',type=int,default=2000)
    p.add_argument('--validation',type=int,default=200)
    p.add_argument('--test',type=int,default=200)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    print(json.dumps(generate_all(args.out,{'train':args.train,'validation':args.validation,'test':args.test},args.seed),indent=2))
