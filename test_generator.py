import unittest
from unittest.mock import patch
from generator import State, Action, observe, select_action, commit_action, generate_episode, validate_episode
import random

class PolicyTests(unittest.TestCase):
    def test_skip_known(self):
        s=State()
        s.known['time_of_day']='evening'
        s.statuses['time_of_day']='known'
        self.assertEqual(select_action(s).target,'activity')
    def test_exit_has_priority(self):
        s=State();s.wants_exit=True;s.user_question_pending=True
        self.assertEqual(select_action(s).type,'CLOSE')
    def test_declined_slot_not_reasked(self):
        s=State();s.statuses['time_of_day']='declined'
        self.assertEqual(select_action(s).target,'activity')
    def test_clarification_limit(self):
        s=State();s.statuses['time_of_day']='ambiguous';s.attempts['time_of_day']=1
        a=select_action(s)
        self.assertEqual(a.type,'CLARIFY')
        commit_action(s,a)
        observe(s,{'reaction':'NON_ANSWER','facts':{},'new_topic':None,'refused':None,'user_question':False,'exit':False})
        self.assertEqual(select_action(s).target,'activity')
        self.assertEqual(s.statuses['time_of_day'],'skipped')
    def test_whole_episodes(self):
        r=random.Random(12)
        for i in range(100): validate_episode(generate_episode(r,str(i),'test'))

class DatasetTests(unittest.TestCase):
    def test_extraction_updates_memory_and_matches_same_turn(self):
        from generator import compile_views
        reactions = set()
        for seed in range(50):
            ep = generate_episode(random.Random(seed), str(seed), 'train')
            views = compile_views(ep)
            self.assertEqual(len(views['extraction']), len(views['history']))
            self.assertEqual(views['extraction'][0]['state_before'], State().snapshot())
            self.assertEqual(views['extraction'][0]['history_before'], [])
            self.assertNotIn('state', views['history'][0])
            for extracted in views['extraction']:
                observation = extracted['target_observation']
                reactions.add(observation['reaction'])
                restored = State.from_snapshot(extracted['state_before'])
                observe(restored, observation)
                self.assertEqual(restored.snapshot(), extracted['target_state'])
                self.assertIsNone(restored.pending)
                if observation['refused']:
                    self.assertEqual(restored.statuses[observation['refused']], 'declined')
                for slot, value in extracted['state_before']['known'].items():
                    self.assertEqual(restored.known[slot], observation['facts'].get(slot, value))
                next_turn = extracted['turn']
                self.assertEqual(extracted['next_turn'], next_turn)
                context = extracted['history_before'] + [{'role':'user', 'content':extracted['utterance']}]
                for view in ('policy', 'response', 'history'):
                    row = views[view][next_turn]
                    self.assertEqual(row['episode_id'], extracted['episode_id'])
                    self.assertEqual(row['history'], context)
                    if view != 'history':
                        self.assertEqual(row['state'], extracted['target_state'])
                action = select_action(restored)
                self.assertEqual({'type':action.type, 'target':action.target}, views['policy'][next_turn]['target_action'])
                self.assertEqual(views['policy'][next_turn]['target_action'], views['response'][next_turn]['action'])
                self.assertEqual(views['response'][next_turn]['target_response'], views['history'][next_turn]['target_response'])
            self.assertEqual(views['response'][-1]['action']['type'], 'CLOSE')
        self.assertTrue({'REFUSAL', 'NON_ANSWER', 'MULTI_FACT', 'TOPIC_SHIFT', 'COUNTER_QUESTION', 'EXIT'} <= reactions)

    def test_state_restore_does_not_share_memory_with_snapshot(self):
        state = State()
        state.followup_used.add('books')
        snapshot = state.snapshot()
        restored = State.from_snapshot(snapshot)
        commit_action(restored, Action('FOLLOW_UP', 'python'))
        restored.known['mood'] = 'happy'
        self.assertEqual(snapshot, state.snapshot())
        self.assertEqual(restored.followup_used, {'books', 'python'})
        with self.assertRaisesRegex(ValueError, 'complete dialogue state'):
            State.from_snapshot({'known': {}})

    def test_messages_exactly_match_turns(self):
        from generator import episode_messages
        e=generate_episode(random.Random(14),'x','train')
        self.assertEqual(e['messages'],episode_messages(e['turns']))
        self.assertEqual(len(e['messages']),2*len(e['turns']))
        self.assertEqual([m['role'] for m in e['messages']], ['user', 'assistant'] * len(e['turns']))

    def test_first_message_facts_precede_policy_without_revealing_hidden_slots(self):
        from generator import UserSimulator, compile_views
        def opening(sim):
            return sim.utterance('time_of_day'), sim.obs('INFORM', facts={'time_of_day': sim.hidden['time_of_day']})
        with patch.object(UserSimulator, 'opening', opening):
            episode = generate_episode(random.Random(42), 'first', 'train')
        validate_episode(episode)
        views = compile_views(episode)
        state = views['extraction'][0]['target_state']
        self.assertEqual(state['known'], {'time_of_day': episode['hidden_user']['time_of_day'],
                                         'activity': None, 'mood': None})
        self.assertEqual(views['policy'][0]['target_action'], {'type': 'ASK', 'target': 'activity'})
        self.assertEqual(views['policy'][0]['history'], [episode['messages'][0]])
    def test_policy_no_mutation(self):
        s=State();s.statuses['time_of_day']='ambiguous';s.attempts['time_of_day']=2
        before=s.snapshot()
        select_action(s)
        self.assertEqual(before,s.snapshot())
    def test_sft_no_hidden_user_or_format_markers(self):
        from generator import compile_views
        from compile_sft import compile_row
        e=generate_episode(random.Random(19),'x','train')
        rows=compile_views(e)['history']
        formatted=compile_row(rows[-1],'history')
        self.assertEqual(formatted['messages'][1:], rows[-1]['history'])
        self.assertEqual(formatted['completion'], rows[-1]['target_response'])
        self.assertNotIn('hidden_user', formatted)
        self.assertNotIn('<|im_', str(formatted))


if __name__=='__main__': unittest.main()
