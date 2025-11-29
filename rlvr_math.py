from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
SYSTEM_PROMPT = (
    "You are a calculator. Solve the user's math problem and reply with an integer. "
    "ASCII characters only, no markdown. "
    "Your final answer should be on a new line at the end of your response. "
)

# Regex matches the last integer-looking token in the output (supports negatives)
NUM_RE = re.compile(r"[-+]?\d+")
REWARD_BUFFER: deque[float] = deque(maxlen=4096)
REWARD_EMA: Optional[float] = None
REWARD_ALPHA: float = 0.9


def build_messages(problem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def parse_answer(text: str) -> Optional[int]:
    cleaned = text.strip().replace(",", "").replace("_", "")
    matches = NUM_RE.findall(cleaned)
    if not matches:
        return None
    return int(matches[-1])


def _sample_int(rng: random.Random, low: int, high: int) -> int:
    x = rng.randint(low, high)
    if x == 0:
        x = rng.choice([low, high, 1, -1])
    return x


def gen_synthetic_math(
    n: int = 500,
    seed: int = 0,
    add_sub_range: int = 99999,
    mul_range: int = 50,
) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    items: List[Tuple[str, int]] = []
    for _ in range(n):
        a = _sample_int(rng, -add_sub_range, add_sub_range)
        b = _sample_int(rng, -add_sub_range, add_sub_range)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            prob = f"{a} + {b} = ?"
        elif op == "-":
            ans = a - b
            prob = f"{a} - {b} = ?"
        elif op == "*":
            a2 = _sample_int(rng, -mul_range, mul_range)
            b2 = _sample_int(rng, -mul_range, mul_range)
            ans = a2 * b2
            prob = f"{a2} * {b2} = ?"
        else:
            raise ValueError("Invalid operation.")
        items.append((prob, int(ans)))
    return items


def gen_ltr_arithmetic(
    n: int = 500,
    seed: int = 0,
    min_steps: int = 2,
    max_steps: int = 3,
    add_sub_range: int = 999,
    mul_range: int = 20,
) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    ops = ["+", "-", "*"]
    items: List[Tuple[str, int]] = []
    for _ in range(n):
        steps = rng.randint(min_steps, max_steps)
        ops_selected = [rng.choice(ops) for _ in range(steps)]
        nums: List[int] = []
        for i in range(steps + 1):
            prev_mul = i > 0 and ops_selected[i - 1] == "*"
            nums.append(
                _sample_int(rng, -mul_range, mul_range)
                if prev_mul
                else _sample_int(rng, -add_sub_range, add_sub_range)
            )
        tokens: List[str] = []
        for i in range(steps):
            tokens.append(str(nums[i]))
            tokens.append(ops_selected[i])
        tokens.append(str(nums[-1]))
        expr = " ".join(tokens)
        res = nums[0]
        for i, op in enumerate(ops_selected):
            b = nums[i + 1]
            if op == "+":
                res = res + b
            elif op == "-":
                res = res - b
            elif op == "*":
                res = res * b
        prompt = (
            "Evaluate this expression strictly from left to right (ignore normal operator precedence):\n"
            f"{expr}\n"
            "What is the result?"
        )
        items.append((prompt, int(res)))
    return items


def gen_word_multi_step(
    n: int = 500,
    seed: int = 0,
    min_ops: int = 3,
    max_ops: int = 5,
    value_range: int = 9999,
    mul_range: int = 50,
) -> List[Tuple[str, int]]:
    rng = random.Random(seed)
    items: List[Tuple[str, int]] = []
    verbs = {"+": "add", "-": "subtract", "*": "multiply by"}
    for _ in range(n):
        k = rng.randint(min_ops, max_ops)
        ops = [rng.choice(list(verbs.keys())) for _ in range(k)]
        start = _sample_int(rng, -value_range, value_range)
        vals: List[int] = []
        for op in ops:
            vals.append(
                _sample_int(rng, -mul_range, mul_range)
                if op == "*"
                else _sample_int(rng, -value_range, value_range)
            )
        res = start
        parts = [f"Start with {start}."]
        for op, v in zip(ops, vals):
            if op == "+":
                res = res + v
                parts.append(f"Then add {v}.")
            elif op == "-":
                res = res - v
                parts.append(f"Then subtract {v}.")
            elif op == "*":
                res = res * v
                parts.append(f"Then multiply by {v}.")
        parts.append("What is the result?")
        prompt = " ".join(parts)
        items.append((prompt, int(res)))
    return items


def _pairs_to_rows(pairs: Sequence[Tuple[str, int]]) -> List[Dict[str, int]]:
    return [{"problem": p, "gold": int(g)} for p, g in pairs]


def render_prompts(
    rows: List[Dict[str, int]],
    model_id: str = DEFAULT_MODEL_ID,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dataset:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    def _render(row: Dict[str, int]) -> Dict[str, int]:
        problem = row.get("problem", row.get("0"))
        gold = row.get("gold", row.get("1"))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return {"prompt": prompt, "gold": gold}

    ds = Dataset.from_list(rows)
    return ds.map(_render, remove_columns=list(ds.column_names))


def reward_correct_integer(completions: List[str], gold: List[int], **kwargs) -> List[float]:
    rewards: List[float] = []
    for out, gt in zip(completions, gold):
        pred = parse_answer(out)
        rewards.append(1.0 if pred == gt else 0.0)
    try:
        REWARD_BUFFER.extend(rewards)
        mean_r = (sum(rewards) / len(rewards)) if rewards else 0.0
        global REWARD_EMA
        REWARD_EMA = mean_r if REWARD_EMA is None else (REWARD_ALPHA * REWARD_EMA + (1 - REWARD_ALPHA) * mean_r)
    except Exception:
        pass
    return rewards


def _make_task_pairs(
    task_cfg: TaskConfig,
    n_train: int,
    n_eval: int,
    train_seed: int,
    eval_seed: int,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if task_cfg.task_mode == "simple":
        train_pairs = (
            gen_synthetic_math(
                n=n_train,
                seed=train_seed,
                add_sub_range=task_cfg.val_range,
                mul_range=task_cfg.mul_range,
            )
            if n_train
            else []
        )
        eval_pairs = gen_synthetic_math(
            n=n_eval,
            seed=eval_seed,
            add_sub_range=task_cfg.val_range,
            mul_range=task_cfg.mul_range,
        )
    elif task_cfg.task_mode == "ltr":
        train_pairs = (
            gen_ltr_arithmetic(
                n=n_train,
                seed=train_seed,
                min_steps=task_cfg.ltr_min_steps,
                max_steps=task_cfg.ltr_max_steps,
                add_sub_range=task_cfg.val_range,
                mul_range=task_cfg.mul_range,
            )
            if n_train
            else []
        )
        eval_pairs = gen_ltr_arithmetic(
            n=n_eval,
            seed=eval_seed,
            min_steps=task_cfg.ltr_min_steps,
            max_steps=task_cfg.ltr_max_steps,
            add_sub_range=task_cfg.val_range,
            mul_range=task_cfg.mul_range,
        )
    elif task_cfg.task_mode == "word":
        train_pairs = (
            gen_word_multi_step(
                n=n_train,
                seed=train_seed,
                min_ops=task_cfg.word_min_ops,
                max_ops=task_cfg.word_max_ops,
                value_range=task_cfg.val_range,
                mul_range=task_cfg.mul_range,
            )
            if n_train
            else []
        )
        eval_pairs = gen_word_multi_step(
            n=n_eval,
            seed=eval_seed,
            min_ops=task_cfg.word_min_ops,
            max_ops=task_cfg.word_max_ops,
            value_range=task_cfg.val_range,
            mul_range=task_cfg.mul_range,
        )
    else:
        raise ValueError(f"Unknown TASK_MODE={task_cfg.task_mode}")
    return train_pairs, eval_pairs


def measure_baseline_accuracy(
    model_id: str = DEFAULT_MODEL_ID,
    task_cfg: Optional[TaskConfig] = None,
    n_eval: int = 100,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    load_in_4bit: bool = True,
    eval_seed: int = 123,
    chat_template: Optional[str] = None,
) -> Dict[str, object]:
    task_cfg = task_cfg or TaskConfig()
    device_map = "auto" if device != "cpu" else {"": "cpu"}
    kwargs = dict(
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = getattr(tok, "bos_token_id", None)
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.bos_token_id = getattr(tok, "bos_token_id", None)

    torch.backends.cuda.matmul.allow_tf32 = True
    model.eval()

    _, eval_pairs = _make_task_pairs(
        task_cfg=task_cfg,
        n_train=0,
        n_eval=n_eval,
        train_seed=eval_seed,
        eval_seed=eval_seed,
    )

    correct = 0
    samples: List[Dict[str, object]] = []
    for i, (problem, ans) in enumerate(eval_pairs):
        messages = build_messages(problem)
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            chat_template=chat_template,
        )
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = parse_answer(gen)
        is_ok = pred == ans
        correct += int(is_ok)
        if i < 10:
            samples.append({"problem": problem, "gold": ans, "raw": gen.strip(), "pred": pred, "ok": bool(is_ok)})

    acc = correct / max(1, len(eval_pairs))
    result = {
        "model": model_id,
        "task_mode": task_cfg.task_mode,
        "n": len(eval_pairs),
        "accuracy": acc,
        "samples": samples,
    }
    logger.info(json.dumps(result, indent=2))
    return result


def _bytes_to_gib(x: int) -> float:
    return x / (1024**3)


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, device: int = 0, alpha: float = 0.3, print_every: int = 10):
        self.device = device
        self.alpha = alpha
        self.print_every = max(1, print_every)
        self._last_t: Optional[float] = None
        self._last_step: int = 0
        self._ema_step_t: Optional[float] = None
        self.csv_path: Optional[str] = None

    def _gpu_mem_stats(self) -> Dict[str, float]:
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            alloc_b = torch.cuda.memory_allocated(self.device)
            reserv_b = torch.cuda.memory_reserved(self.device)
            return {
                "free_gib": _bytes_to_gib(free_b),
                "total_gib": _bytes_to_gib(total_b),
                "used_gib": _bytes_to_gib(total_b - free_b),
                "alloc_gib": _bytes_to_gib(alloc_b),
                "reserved_gib": _bytes_to_gib(reserv_b),
            }
        return {"free_gib": 0.0, "total_gib": 0.0, "used_gib": 0.0, "alloc_gib": 0.0, "reserved_gib": 0.0}

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_t = time.time()
        ms = self._gpu_mem_stats()
        self.csv_path = f"{args.output_dir}/mem_log.csv"
        try:
            with open(self.csv_path, "w") as f:
                f.write("time,step,used_gib,alloc_gib,reserved_gib,usage_pct,ema_step_s,approx_tok_s,approx_seq_s\n")
        except Exception:
            self.csv_path = None
        if state.is_world_process_zero:
            logger.info(
                "[mem] start used=%.2fGiB free=%.2fGiB total=%.2fGiB reserved=%.2fGiB",
                ms["used_gib"],
                ms["free_gib"],
                ms["total_gib"],
                ms["reserved_gib"],
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.time()
        step_delta = state.global_step - self._last_step if self._last_step is not None else 0
        dt = now - self._last_t if self._last_t is not None else 0.0
        step_t = (dt / step_delta) if step_delta else None
        if step_t is not None:
            self._ema_step_t = step_t if self._ema_step_t is None else (self.alpha * step_t + (1 - self.alpha) * self._ema_step_t)
        ms = self._gpu_mem_stats()
        if logs is None:
            logs = {}
        usage = (ms["used_gib"] / ms["total_gib"] * 100.0) if ms["total_gib"] else 0.0
        global_batch = args.per_device_train_batch_size * args.world_size
        toks_per_step_est = global_batch * (getattr(args, "num_generations", 1) or 1) * (args.max_completion_length or 1)
        tok_s = (toks_per_step_est / self._ema_step_t) if (self._ema_step_t and toks_per_step_est) else None
        seq_s = (global_batch * (getattr(args, "num_generations", 1) or 1) / self._ema_step_t) if self._ema_step_t else None
        eta_s = ((args.max_steps - state.global_step) * self._ema_step_t) if (self._ema_step_t and args.max_steps) else None
        logs.update(
            {
                "gpu_used_gib": ms["used_gib"],
                "gpu_alloc_gib": ms["alloc_gib"],
                "gpu_reserved_gib": ms["reserved_gib"],
                "gpu_usage_pct": usage,
                "ema_step_s": self._ema_step_t or 0.0,
                "approx_tok_s": tok_s or 0.0,
                "approx_seq_s": seq_s or 0.0,
            }
        )
        if state.is_world_process_zero and state.global_step % self.print_every == 0:
            eta_str = f"ETA~{int(eta_s // 60)}m{int(eta_s % 60)}s" if eta_s else "ETA~na"
            logger.info(
                "[mem][step %d] used=%.2fGiB (%d%%) step_t=%.2fs tok/s~%.0f seq/s~%.1f %s",
                state.global_step,
                ms["used_gib"],
                int(usage),
                self._ema_step_t or 0.0,
                tok_s or 0.0,
                seq_s or 0.0,
                eta_str,
            )
        if self.csv_path:
            try:
                with open(self.csv_path, "a") as f:
                    f.write(
                        f"{int(now)},{state.global_step},{ms['used_gib']:.4f},{ms['alloc_gib']:.4f},{ms['reserved_gib']:.4f},"
                        f"{usage:.2f},{(self._ema_step_t or 0):.4f},{(tok_s or 0):.2f},{(seq_s or 0):.2f}\n"
                    )
            except Exception:
                pass
        self._last_t, self._last_step = now, state.global_step


class QuickEvalCallback(TrainerCallback):
    def __init__(
        self,
        tok,
        eval_pairs: Sequence[Tuple[str, int]],
        n_quick: int = 16,
        device: str = "cuda",
        max_new_tokens: int = 64,
    ):
        self.tok = tok
        self.eval_pairs = list(eval_pairs)[:n_quick] if eval_pairs else []
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._trainer: Optional[GRPOTrainer] = None
        self.prompts: List[str] = []
        for p, _g in self.eval_pairs:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]
            self.prompts.append(self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    def set_trainer(self, trainer: GRPOTrainer) -> None:
        self._trainer = trainer

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if not (state.is_world_process_zero and self._trainer and self.prompts):
            return
        model = self._trainer.model
        tok = self.tok
        n = len(self.prompts)
        correct = 0
        for i in range(n):
            text = self.prompts[i]
            inputs = tok(text, return_tensors="pt").to(args.device if hasattr(args, "device") else self.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )
            gen = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            pred = parse_answer(gen)
            gold = self.eval_pairs[i][1]
            correct += int(pred == gold)
        acc = correct / max(1, n)
        if logs is None:
            logs = {}
        logs["quick_eval_accuracy"] = acc
        logger.info("[qe][step %d] acc_quick=%.1f%% on %d", state.global_step, acc * 100.0, n)


class RewardLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}
        try:
            rb_len = len(REWARD_BUFFER)
            reward_rate = (sum(REWARD_BUFFER) / rb_len) if rb_len else 0.0
            reward_ema = REWARD_EMA if REWARD_EMA is not None else reward_rate
        except Exception:
            reward_rate, reward_ema = 0.0, 0.0
        # Advantage proxy: centered reward scaled by KL if present
        adv_proxy = reward_rate - float(logs.get("kl", 0.0))
        logs.update({"train_reward_rate": reward_rate, "train_reward_ema": reward_ema, "advantage_proxy": adv_proxy})
        if state.is_world_process_zero and state.global_step % args.logging_steps == 0:
            logger.info(
                "[log][step %d] loss=%.4f reward=%.3f ema=%.3f adv~%.3f kl=%.4f",
                state.global_step,
                float(logs.get("loss", 0.0)),
                reward_rate,
                reward_ema,
                adv_proxy,
                float(logs.get("kl", 0.0)),
            )


@dataclass
class TaskConfig:
    task_mode: str = "ltr"
    ltr_min_steps: int = 2
    ltr_max_steps: int = 3
    word_min_ops: int = 2
    word_max_ops: int = 3
    val_range: int = 99
    mul_range: int = 20


@dataclass
class TrainConfig:
    model_id: str = DEFAULT_MODEL_ID
    # NOTE: We keep training in float32 with 4-bit disabled because prior attempts to mix bf16/4bit
    # triggered hidden-state vs lm_head dtype mismatches during generation. Only change these if you
    # are ready to debug dtype consistency end-to-end.
    dtype: torch.dtype = torch.float32
    load_in_4bit: bool = False
    device: str = "cuda"
    train_seed: int = 42
    eval_seed: int = 123
    quick_run: bool = True
    train_samples_quick: int = 1024
    eval_samples_quick: int = 32
    train_samples_full: int = 4000
    eval_samples_full: int = 200
    max_steps_quick: int = 400
    max_steps_full: int = 3000
    max_prompt_tok: int = 128
    max_completion_tok: int = 64
    num_generations: int = 8
    per_device_train_batch: int = 8
    grad_accum_steps: int = 1
    task: TaskConfig = field(default_factory=TaskConfig)
    learning_rate: float = 2e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 200
    save_total_limit: int = 2
    run_name: str = "grpo-math-quick"
    output_dir: str = "qwen3-06b-grpo-math-quick"


def train_grpo_integer_math(cfg: Optional[TrainConfig] = None) -> GRPOTrainer:
    cfg = cfg or TrainConfig()
    random.seed(cfg.train_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    n_train = cfg.train_samples_quick if cfg.quick_run else cfg.train_samples_full
    n_eval = cfg.eval_samples_quick if cfg.quick_run else cfg.eval_samples_full
    max_steps = cfg.max_steps_quick if cfg.quick_run else cfg.max_steps_full

    train_pairs, eval_pairs = _make_task_pairs(
        task_cfg=cfg.task,
        n_train=n_train,
        n_eval=n_eval,
        train_seed=cfg.train_seed,
        eval_seed=cfg.eval_seed,
    )
    train_rows = _pairs_to_rows(train_pairs)
    eval_rows = _pairs_to_rows(eval_pairs)
    train_ds = render_prompts(train_rows, model_id=cfg.model_id)
    eval_ds = render_prompts(eval_rows, model_id=cfg.model_id)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    effective_dtype = cfg.dtype

    args = GRPOConfig(
        output_dir=cfg.output_dir,
        seed=cfg.train_seed,
        tf32=True,
        bf16=(effective_dtype == torch.bfloat16),
        per_device_train_batch_size=cfg.per_device_train_batch,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        max_steps=max_steps,
        run_name=cfg.run_name,
        report_to="none",
        max_prompt_length=cfg.max_prompt_tok,
        max_completion_length=cfg.max_completion_tok,
        num_generations=cfg.num_generations,
        temperature=0.8,
        top_p=0.9,
        beta=0.05,
        epsilon=0.2,
        scale_rewards="batch",  # stdev at batch level, more stable than group
        loss_type="dapo",
        model_init_kwargs=dict(
            torch_dtype=effective_dtype,
            trust_remote_code=True,
            device_map="auto",
            **(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=effective_dtype,
                    )
                }
                if cfg.load_in_4bit
                else {}
            ),
        ),
    )
    args.generation_batch_size = None

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    mem_cb = MemoryMonitorCallback(print_every=max(cfg.logging_steps, 5))
    qe_cb = QuickEvalCallback(tok, eval_pairs, n_quick=min(16, len(eval_pairs)), device=cfg.device, max_new_tokens=cfg.max_completion_tok)
    reward_log_cb = RewardLoggingCallback()

    trainer = GRPOTrainer(
        model=cfg.model_id,
        reward_funcs=reward_correct_integer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=[mem_cb, qe_cb, reward_log_cb],
    )
    qe_cb.set_trainer(trainer)
    model = trainer.model
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = getattr(tok, "bos_token_id", None)
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.bos_token_id = getattr(tok, "bos_token_id", None)
    if trainer.args.generation_batch_size is None:
        trainer.args.generation_batch_size = (
            trainer.args.per_device_train_batch_size * trainer.args.world_size * trainer.args.steps_per_generation
        )

    trainer.train()
    trainer.save_model()
    try:
        trainer.save_state()
    except Exception:
        pass
    return trainer


def load_trainer_logs(output_dir: str):
    p = Path(output_dir) / "trainer_state.json"
    if not p.exists():
        raise FileNotFoundError(f"No trainer_state.json at {p}")
    with open(p) as f:
        st = json.load(f)
    import pandas as pd

    return pd.DataFrame(st.get("log_history", []))


def load_mem_log(output_dir: str):
    p = Path(output_dir) / "mem_log.csv"
    if not p.exists():
        return None
    import pandas as pd

    return pd.read_csv(p)


def plot_losses(df):
    import matplotlib.pyplot as plt
    import pandas as pd

    if df is None or df.empty:
        logger.info("No trainer logs found.")
        return
    plt.figure(figsize=(8, 4))
    if "loss" in df:
        plt.plot(df.get("step", df.index), pd.to_numeric(df["loss"], errors="coerce"), label="train loss", alpha=0.7)
    if "eval_loss" in df:
        plt.plot(df.get("step", df.index), pd.to_numeric(df["eval_loss"], errors="coerce"), label="eval loss", alpha=0.7)
    if "train_reward_ema" in df:
        plt.plot(df.get("step", df.index), pd.to_numeric(df["train_reward_ema"], errors="coerce"), label="reward ema", alpha=0.7)
    if "quick_eval_accuracy" in df:
        plt.plot(df.get("step", df.index), pd.to_numeric(df["quick_eval_accuracy"], errors="coerce"), label="quick eval acc", alpha=0.7)
    plt.xlabel("step")
    plt.ylabel("metric")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_memory(dfm):
    import matplotlib.pyplot as plt
    import pandas as pd

    if dfm is None or dfm.empty:
        logger.info("No mem_log.csv found.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(dfm["step"], dfm["used_gib"], label="used GiB")
    ax[0].plot(dfm["step"], dfm["reserved_gib"], label="reserved GiB", alpha=0.6)
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("GiB")
    ax[0].legend()
    ax[0].grid(True, alpha=0.2)
    usage_col = "gpu_usage_pct" if "gpu_usage_pct" in dfm.columns else ("usage_pct" if "usage_pct" in dfm.columns else None)
    if usage_col:
        ax[1].plot(dfm["step"], dfm[usage_col])
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("% used")
    ax[1].grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()
    if "approx_tok_s" in dfm.columns:
        plt.figure(figsize=(8, 3))
        plt.plot(dfm["step"], dfm["approx_tok_s"])
        plt.xlabel("step")
        plt.ylabel("approx tok/s")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()


def summarize_logs(df, dfm):
    out = {}
    import pandas as pd

    if df is not None and not df.empty and "loss" in df:
        out["final_train_loss"] = float(pd.to_numeric(df["loss"], errors="coerce").dropna().tail(1))
    if df is not None and not df.empty and "eval_loss" in df:
        ev = pd.to_numeric(df["eval_loss"], errors="coerce").dropna()
        if len(ev):
            out["best_eval_loss"] = float(ev.min())
    if df is not None and not df.empty and "quick_eval_accuracy" in df:
        acc = pd.to_numeric(df["quick_eval_accuracy"], errors="coerce").dropna()
        if len(acc):
            out["max_quick_eval_acc"] = float(acc.max())
    if df is not None and not df.empty and "train_reward_ema" in df:
        ema = pd.to_numeric(df["train_reward_ema"], errors="coerce").dropna()
        if len(ema):
            out["reward_ema_last"] = float(ema.iloc[-1])
    if dfm is not None and len(dfm):
        out["peak_used_gib"] = float(dfm["used_gib"].max())
        if "approx_tok_s" in dfm:
            vals = pd.to_numeric(dfm["approx_tok_s"], errors="coerce").replace({0: pd.NA}).dropna()
            out["mean_tok_s"] = float(vals.mean()) if len(vals) else 0.0
    logger.info(json.dumps(out, indent=2))
    return out
