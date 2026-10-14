import unittest
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
    def test_messages_exactly_match_turns(self):
        from generator import episode_messages
        e=generate_episode(random.Random(14),'x','train')
        self.assertEqual(e['messages'],episode_messages(e['turns']))
        self.assertEqual(len(e['messages']),2*len(e['turns'])-1)
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
