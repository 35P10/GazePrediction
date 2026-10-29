"""Check direct delegation to the official controller; no network or real weights."""
import copy
from importlib import import_module
import json
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from models import load_predictor
from models.chartist import Chartist
from models.chartist.model import CHARTIST_DIR


class RecordedClient:
    def __init__(self, fenced=False, fenced_answer=False):
        self.calls = []
        self.fenced = fenced
        self.fenced_answer = fenced_answer
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        answer = 'Please answer the question' in kwargs['messages'][-1]['content']
        content = (dict(answer='9.0', reason='observed value') if answer
                   else dict(function='search_text', parameter='', reason='find a label'))
        text = json.dumps(content)
        if (self.fenced and not answer) or (self.fenced_answer and answer):
            text = '```json\n' + text + '\n```'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakePolicy:
    def predict(self, observation, **kwargs):
        return np.array([3, 8]), None


class FakeEnvironment:
    gaze_size = 16

    def __init__(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        return {}

    def step(self, action):
        return {}, 0, True, dict(function='search_text', position=action, text='9.0')


class ChartistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        sys.path.insert(0, str(CHARTIST_DIR))
        cls.released = import_module('model.llm_policy')

    def predictor(self, client=None):
        with patch('stable_baselines3.PPO.load', return_value=FakePolicy()):
            return Chartist(client=client or RecordedClient())

    def test_uses_exact_official_class_methods_and_constructor_defaults(self):
        before = dict(os.environ)
        client = RecordedClient()
        predictor = self.predictor(client)
        self.assertIs(type(predictor.network), self.released.LLMPolicy)
        for method in ('setup', 'action', 'answer', 'update_memory', 'memory_summary',
                       'search_text', 'look_at_data_point', 'read_axis', 'explore'):
            self.assertIs(getattr(predictor.network, method).__func__, getattr(self.released.LLMPolicy, method))
        self.assertEqual(predictor.network.max_steps, 10)
        self.assertEqual(predictor.network.memory_limit, 30)
        self.assertIs(predictor.network.client.client, client)
        self.assertIs(self.released.OpenAI, import_module('openai').OpenAI)
        self.assertEqual(os.environ.get('AALTO_OPENAI_API_KEY'), before.get('AALTO_OPENAI_API_KEY'))

    def test_predict_runs_original_decisions_until_original_limit(self):
        client = RecordedClient()
        predictor = self.predictor(client)
        with patch.multiple(self.released, SearchEnv=FakeEnvironment, MapEnv=FakeEnvironment,
                            ReadEnv=FakeEnvironment, TitleEnv=FakeEnvironment):
            result = predictor.predict(dict(id=1, question='What value?'), meta_json='unused', folder='unused')
        self.assertEqual(len(result['points']), 10)
        self.assertEqual(len(result['actions']), 10)
        self.assertEqual(result['answer']['answer'], '9.0')
        self.assertEqual(predictor.network.steps, 11)
        self.assertEqual(len(client.calls), 21)  # Ten decisions, ten summaries, one final answer.
        self.assertTrue(all(isinstance(action, str) for action in result['actions']))

    def test_predict_logs_every_llm_call_and_final_state(self):
        client = RecordedClient()
        predictor = self.predictor(client)
        with patch.multiple(self.released, SearchEnv=FakeEnvironment, MapEnv=FakeEnvironment,
                            ReadEnv=FakeEnvironment, TitleEnv=FakeEnvironment):
            result = predictor.predict(dict(id=1, question='What value?'), meta_json='unused', folder='unused')
        log = result['log']
        self.assertIs(predictor.last_log, log)
        self.assertEqual([call['request'] for call in log['llm_calls']], client.calls)
        self.assertEqual([call['decision'] for call in log['llm_calls']],
                         [step for step in range(1, 11) for _ in range(2)] + [11])
        self.assertEqual(log['llm_calls'][-1]['response'], json.dumps(dict(answer='9.0', reason='observed value')))
        self.assertEqual(log['gaze_cells'], [[3, 8]] * 10)
        self.assertEqual(len(log['memory']), 10)
        self.assertEqual((log['steps'], log['max_steps'], log['memory_limit']), (11, 10, 30))
        json.dumps(log)

    def test_prompts_name_the_chart_type_from_meta(self):
        client = RecordedClient()
        predictor = self.predictor(client)
        with patch.multiple(self.released, SearchEnv=FakeEnvironment, MapEnv=FakeEnvironment,
                            ReadEnv=FakeEnvironment, TitleEnv=FakeEnvironment):
            predictor.predict(dict(id=1, question='What value?', chart_type='a horizontal bar chart'),
                              meta_json='unused', folder='unused')
        systems = [call['messages'][0]['content'] for call in client.calls]
        self.assertTrue(all('You are looking at a horizontal bar chart' in s for s in systems))
        self.assertFalse(any('scatterplot' in s for s in systems))

    def test_memory_summary_sends_the_summary_prompt(self):
        client = RecordedClient()
        predictor = self.predictor(client)
        with patch.multiple(self.released, SearchEnv=FakeEnvironment, MapEnv=FakeEnvironment,
                            ReadEnv=FakeEnvironment, TitleEnv=FakeEnvironment):
            predictor.predict(dict(id=1, question='What value?'), meta_json='unused', folder='unused')
        summaries = [call for call in predictor.last_log['llm_calls'] if call['requested_model'] == 'gpt-4o-mini'
                     and call['decision'] <= predictor.network.max_steps]
        self.assertEqual(len(summaries), 10)
        for call in summaries:
            self.assertIn('Please summarize the memory into one paragraph', call['request']['messages'][-1]['content'])
            self.assertNotIn('Please decide your next action', call['request']['messages'][-1]['content'])

    def test_fenced_final_answer_is_accepted_at_once(self):
        client = RecordedClient(fenced_answer=True)
        predictor = self.predictor(client)
        with patch.multiple(self.released, SearchEnv=FakeEnvironment, MapEnv=FakeEnvironment,
                            ReadEnv=FakeEnvironment, TitleEnv=FakeEnvironment):
            result = predictor.predict(dict(id=1, question='What value?'), meta_json='unused', folder='unused')
        self.assertEqual(result['answer']['answer'], '9.0')
        final = [call for call in result['log']['llm_calls'] if call['decision'] == 11]
        self.assertEqual(len(final), 1)
        self.assertTrue(final[0]['response'].startswith('```'))
        self.assertEqual(json.loads(final[0]['unfenced_response'])['answer'], '9.0')
        self.assertEqual(len(result['points']), 10)

    def test_every_call_goes_to_gpt_4o_by_default(self):
        client = RecordedClient()
        predictor = self.predictor(client)
        with patch.multiple(self.released, SearchEnv=FakeEnvironment, MapEnv=FakeEnvironment,
                            ReadEnv=FakeEnvironment, TitleEnv=FakeEnvironment):
            result = predictor.predict(dict(id=1, question='What value?'), meta_json='unused', folder='unused')
        self.assertEqual({call['model'] for call in client.calls}, {'gpt-4o'})
        self.assertEqual({call['requested_model'] for call in result['log']['llm_calls']}, {'gpt-4o', 'gpt-4o-mini'})
        self.assertEqual(result['log']['llm_model'], 'gpt-4o')

    def test_llm_model_none_keeps_the_models_named_by_the_code(self):
        client = RecordedClient()
        with patch('stable_baselines3.PPO.load', return_value=FakePolicy()):
            predictor = Chartist(client=client, llm_model=None)
        with patch.multiple(self.released, SearchEnv=FakeEnvironment, MapEnv=FakeEnvironment,
                            ReadEnv=FakeEnvironment, TitleEnv=FakeEnvironment):
            predictor.predict(dict(id=1, question='What value?'), meta_json='unused', folder='unused')
        self.assertEqual({call['model'] for call in client.calls}, {'gpt-4o', 'gpt-4o-mini'})

    def test_does_not_replace_the_official_json_reader(self):
        predictor = self.predictor(RecordedClient(fenced=True))
        with patch.multiple(self.released, SearchEnv=FakeEnvironment, MapEnv=FakeEnvironment,
                            ReadEnv=FakeEnvironment, TitleEnv=FakeEnvironment):
            result = predictor.predict(dict(id=1, question='What value?'), meta_json='unused', folder='unused')
        self.assertEqual(result['points'], [])
        self.assertEqual(result['actions'], [])
        self.assertEqual(predictor.network.steps, 11)

    def test_custom_controller_options_are_not_available(self):
        for settings in (dict(max_decisions=50), dict(memory_limit=30), dict(decision_model='other'),
                         dict(max_response_attempts=5)):
            with self.subTest(settings=settings), self.assertRaises(TypeError):
                load_predictor('chartist', **settings)


if __name__ == '__main__':
    unittest.main()
