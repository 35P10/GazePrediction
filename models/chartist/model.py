"""Load and run the unmodified official Chartist controller and PPO policies."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Iterable, List, Union
from typing_extensions import NotRequired, TypedDict

from unittest.mock import patch

CHARTIST_DIR = Path(__file__).resolve().parent / 'official_reference'
PathLike = Union[str, Path]
# The release defaults to 5 decisions (at most 25 fixations). The paper's 46-49 mean fixations
# (SD 3-7) fit a cap of 10 decisions x 5 fixations, passed through the official constructor.
MAX_STEPS = 10


class ChartistText(TypedDict):
    """OCR text and its [left, top, right, bottom] box in the 320x320 chart."""
    position: List[int]
    text: str
    ocr_score: NotRequired[float]
    semantic_similarity: NotRequired[float]


class ChartistInput(TypedDict):
    """One published meta.json record, passed unchanged to LLMPolicy.setup(meta)."""
    id: int
    image: str
    task: str
    chart_type: str
    question: str
    text_positions: List[ChartistText]


def _plain(value):
    """JSON-compatible copy of controller state (numpy values become Python values)."""
    def convert(item):
        return item.tolist() if hasattr(item, 'tolist') else str(item)
    return json.loads(json.dumps(value, default=convert))


class _LoggedClient:
    """Forwards chat.completions.create to the user's client and records every call."""

    def __init__(self, client, step):
        self.client = client
        self.step = step
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        record = dict(decision=self.step(), request=copy.deepcopy(kwargs))
        self.calls.append(record)
        start = time.perf_counter()
        try:
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as error:
            record.update(seconds=time.perf_counter() - start, error=repr(error))
            raise
        usage = getattr(completion, 'usage', None)
        record.update(seconds=time.perf_counter() - start, served_model=getattr(completion, 'model', None),
                      response=completion.choices[0].message.content,
                      usage=usage.model_dump() if hasattr(usage, 'model_dump') else None)
        return completion


class Chartist:
    """Official Chartist, with local paths, a user-provided LLM connection and 10 decisions."""

    def __init__(self, *, api_key=None, client=None, device='cpu'):
        from openai import OpenAI
        from stable_baselines3 import PPO

        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        sys.path.insert(0, str(CHARTIST_DIR))
        import model.llm_policy
        from model.llm_policy import LLMPolicy

        if Path(model.llm_policy.__file__).resolve() != CHARTIST_DIR / 'model/llm_policy.py':
            raise RuntimeError('A different package named model was imported; use a fresh kernel.')
        if client is None:
            client = OpenAI(api_key=api_key)
        client = _LoggedClient(client, step=lambda: self.network.steps)
        policies = {name + '_model': PPO.load(str(CHARTIST_DIR / 'model' / f'vision_{name}.pt'), device=device)
                    for name in ('search', 'map', 'read', 'title')}

        def connect(**kwargs):
            # Replace only the authors' service connection. Restore its environment changes afterward.
            kwargs['http_client'].close()
            return client

        with patch.dict(os.environ), patch.object(model.llm_policy, 'OpenAI', side_effect=connect):
            self.network = LLMPolicy(**policies, MAX_STEPS=MAX_STEPS)
        self.client = client
        self.device = device
        self.last_log = None

    def predict(self, meta: ChartistInput, *, meta_json: PathLike, folder: PathLike):
        """One prepared condition; points use the model's 320x320 coordinate space.

        The result's log holds every LLM call and the final controller state. It is also kept in
        last_log while the prediction runs, so a failed prediction can still be inspected.
        """
        self.client.calls = []
        stdout = io.StringIO()
        self.last_log = log = dict(max_steps=self.network.max_steps, memory_limit=self.network.memory_limit,
                                   llm_calls=self.client.calls, stdout=stdout)
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(stdout):  # The official controller reports failures with print().
                self.network.setup(meta, meta_json=str(meta_json), folder=str(folder))
                done = False
                while not done:
                    done = self.network.action()
        finally:
            log.update(seconds=time.perf_counter() - start, steps=self.network.steps,
                       gaze_cells=_plain([list(action[:2]) for action in self.network.gaze_history]),
                       memory=_plain(self.network.memory), memory_summary=self.network.memory_in_text,
                       stdout=stdout.getvalue())
        return dict(points=list(zip(self.network.scanpath_xs, self.network.scanpath_ys)),
                    answer=self.network.answer_to_the_task, actions=self.network.action_history, log=log)

    def run(self, inputs: Iterable[ChartistInput], *, meta_json: PathLike, folder: PathLike):
        """Return one prediction per prepared condition, in input order."""
        return [self.predict(meta, meta_json=meta_json, folder=folder) for meta in inputs]


def create_predictor(**settings):
    return Chartist(**settings)
