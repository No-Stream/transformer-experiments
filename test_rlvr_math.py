from __future__ import annotations

import types
from typing import List, Tuple

import rlvr_math as rm


def test_make_task_pairs_deterministic_and_seeded():
    task = rm.TaskConfig(task_mode="ltr", ltr_min_steps=2, ltr_max_steps=3, val_range=10, mul_range=3)
    train_seed = 11
    eval_seed = 99
    pairs_a_train, pairs_a_eval = rm._make_task_pairs(task, n_train=5, n_eval=4, train_seed=train_seed, eval_seed=eval_seed)
    pairs_b_train, pairs_b_eval = rm._make_task_pairs(task, n_train=5, n_eval=4, train_seed=train_seed, eval_seed=eval_seed)

    # Deterministic for same seeds/config
    assert pairs_a_train == pairs_b_train
    assert pairs_a_eval == pairs_b_eval

    # Train/eval should differ because seeds differ
    assert pairs_a_train != pairs_a_eval

    # Changing eval seed should change eval set but not train set
    _, pairs_c_eval = rm._make_task_pairs(task, n_train=5, n_eval=4, train_seed=train_seed, eval_seed=eval_seed + 1)
    assert pairs_c_eval != pairs_a_eval
    assert pairs_a_train == pairs_b_train


def test_parse_and_reward_correct_integer_handles_last_int_and_sign():
    texts = ["answer: 3", "nums -1 then 4", "no ints here", "foo -7 bar 5"]
    gold = [3, 4, 0, 5]
    # parse_answer uses last integer-like token
    assert rm.parse_answer(texts[0]) == 3
    assert rm.parse_answer(texts[1]) == 4
    assert rm.parse_answer(texts[2]) is None
    assert rm.parse_answer(texts[3]) == 5

    rewards = rm.reward_correct_integer(texts, gold)
    assert rewards == [1.0, 1.0, 0.0, 1.0]


def test_measure_baseline_uses_task_and_eval_seed(monkeypatch):
    # Stub model/tokenizer to avoid heavy downloads; only ensure call flow and dataset size.
    class DummyIDs(list):
        @property
        def shape(self):
            return (1, 2)

    class DummyBatch(dict):
        def to(self, device):
            return self

    class DummyTok:
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, **kwargs):
            return "prompt"

        def __call__(self, text, return_tensors="pt"):
            return DummyBatch({"input_ids": DummyIDs([0, 1])})

        def decode(self, ids, skip_special_tokens=True):
            return "0"

    class DummyModel:
        config = types.SimpleNamespace(pad_token_id=None, eos_token_id=None, bos_token_id=None, generation_config=None)

        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[0, 1, 2]]

    def fake_from_pretrained(*args, **kwargs):
        return DummyModel()

    def fake_tok_from_pretrained(*args, **kwargs):
        return DummyTok()

    monkeypatch.setattr(rm, "AutoModelForCausalLM", types.SimpleNamespace(from_pretrained=fake_from_pretrained))
    monkeypatch.setattr(rm, "AutoTokenizer", types.SimpleNamespace(from_pretrained=fake_tok_from_pretrained))

    task_cfg = rm.TaskConfig(task_mode="simple")
    res = rm.measure_baseline_accuracy(
        model_id="dummy",
        task_cfg=task_cfg,
        n_eval=7,
        device="cpu",
        dtype=None,
        load_in_4bit=False,
        eval_seed=321,
    )
    # Should report the requested eval size and task mode
    assert res["n"] == 7
    assert res["task_mode"] == "simple"


def test_train_config_seeds_and_generation_batch_size(monkeypatch):
    # Stub GRPOTrainer to inspect args without running training
    captured = {}

    class DummyTrainer:
        def __init__(self, *args, **kwargs):
            captured["args"] = kwargs["args"]
            self.args = kwargs["args"]
            self.model = types.SimpleNamespace(config=types.SimpleNamespace(), generation_config=types.SimpleNamespace())

        def train(self):
            return

        def save_model(self):
            return

        def save_state(self):
            return

    def fake_trainer(**kwargs):
        return DummyTrainer(**kwargs)

    monkeypatch.setattr(rm, "GRPOTrainer", fake_trainer)

    class DummyArgs:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.world_size = getattr(self, "world_size", 1)
            self.steps_per_generation = getattr(self, "steps_per_generation", 1)
            self.generation_batch_size = getattr(self, "generation_batch_size", None)

    monkeypatch.setattr(rm, "GRPOConfig", DummyArgs)
    class DummyTok:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.bos_token_id = 2
            self.padding_side = "left"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return "prompt"

        def __call__(self, text, return_tensors="pt"):
            return types.SimpleNamespace(input_ids=[[0, 1]], to=lambda device: types.SimpleNamespace(input_ids=[[0, 1]]))

        def decode(self, ids, skip_special_tokens=True):
            return "0"

    monkeypatch.setattr(rm, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTok()))

    cfg = rm.TrainConfig(
        quick_run=True,
        max_steps_quick=123,
        train_seed=7,
        eval_seed=8,
        dtype=None,
        task=rm.TaskConfig(task_mode="simple"),
    )
    rm.train_grpo_integer_math(cfg)

    args = captured["args"]
    assert args.max_steps == 123
    assert args.seed == 7


def test_baseline_smoke_tiny_gpt2(monkeypatch):
    import os

    if not os.getenv("RLVR_SMOKE"):
        import pytest

        pytest.skip("RLVR_SMOKE not set; skipping HF smoke test")

    model_id = "sshleifer/tiny-gpt2"
    try:
        tok = rm.AutoTokenizer.from_pretrained(model_id)
        model = rm.AutoModelForCausalLM.from_pretrained(model_id)
    except Exception:
        import pytest

        pytest.skip("Could not load tiny model/tokenizer")

    res = rm.measure_baseline_accuracy(
        model_id=model_id,
        task_cfg=rm.TaskConfig(task_mode="simple", val_range=10, mul_range=3),
        n_eval=3,
        device="cpu",
        dtype=rm.torch.float32,
        load_in_4bit=False,
        eval_seed=5,
        chat_template="You are a calculator.\nUser: {{user}}\nAssistant:",
    )
    assert res["n"] == 3
