"""Generation backends.

- vllm: fast batched generation on a GPU (use this on Kaggle).
- hf:   plain transformers, slower; a fallback if vLLM will not install or load.
- mock: no model; replays fixed answers so the whole pipeline can be tested on a CPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

# Liquid's recommended sampling for LFM3 (model card).
SAMPLING_PRESETS = {
    "general": {"temperature": 0.1, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.1},
    "math": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.0},
}


@dataclass
class Generation:
    text: str
    n_tokens: int
    finish_reason: str  # "stop" or "length"


class Backend:
    tokenizer = None

    def render(self, messages: list[dict], thinking: bool) -> str:
        """Apply the model's own chat template; enable_thinking switches the mode."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def generate(self, prompts: list[str], sampling: dict, max_new_tokens: int, seed: int) -> list[Generation]:
        raise NotImplementedError

    def _strip_special(self, text: str) -> str:
        # Keep <think> tags (they may be special tokens) but drop end-of-turn / padding markers.
        for tok in getattr(self.tokenizer, "all_special_tokens", []):
            if tok and tok not in ("<think>", "</think>"):
                text = text.replace(tok, "")
        return text


class VLLMBackend(Backend):
    def __init__(self, model: str, dtype: str = "auto", max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.90, tensor_parallel_size: int = 1,
                 enforce_eager: bool = False, seed: int = 0):
        import torch
        from vllm import LLM

        if dtype == "auto":
            # Kaggle's T4 / P100 have no bfloat16; fall back to float16.
            dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       tensor_parallel_size=tensor_parallel_size, enforce_eager=enforce_eager, seed=seed)
        self.tokenizer = self.llm.get_tokenizer()
        self.dtype = dtype

    def generate(self, prompts, sampling, max_new_tokens, seed):
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_new_tokens, seed=seed, skip_special_tokens=False, **sampling)
        outs = self.llm.generate(prompts, params, use_tqdm=True)
        return [Generation(self._strip_special(o.outputs[0].text), len(o.outputs[0].token_ids),
                           o.outputs[0].finish_reason or "stop") for o in outs]


class HFBackend(Backend):
    def __init__(self, model: str, dtype: str = "auto", batch_size: int = 8, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
                "float16" if torch.cuda.is_available() else "float32")
        self.dtype = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, dtype), device_map="auto")
        self.model.eval()
        self.batch_size = batch_size

    def generate(self, prompts, sampling, max_new_tokens, seed):
        torch = self.torch
        torch.manual_seed(seed)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths per batch
        results: list[Generation | None] = [None] * len(prompts)
        eos = self.tokenizer.eos_token_id
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                                   add_special_tokens=False).to(self.model.device)
            do_sample = sampling["temperature"] > 0
            with torch.no_grad():
                out = self.model.generate(
                    **batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
                    temperature=sampling["temperature"] if do_sample else None,
                    top_k=sampling["top_k"] if do_sample else None,
                    top_p=sampling["top_p"] if do_sample else None,
                    repetition_penalty=sampling["repetition_penalty"],
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new = out[:, batch["input_ids"].shape[1]:]
            for row, i in zip(new.tolist(), idx):
                stop = next((k for k, t in enumerate(row) if t == eos), None)
                toks = row if stop is None else row[:stop]
                text = self._strip_special(self.tokenizer.decode(toks, skip_special_tokens=False))
                results[i] = Generation(text, len(toks), "length" if stop is None and len(row) >= max_new_tokens else "stop")
            print(f"  hf batch {start // self.batch_size + 1}/{-(-len(order) // self.batch_size)}", flush=True)
        return results  # type: ignore[return-value]


class MockBackend(Backend):
    """Replays answers keyed by user message. Used to test parsing and scoring end to end."""

    def __init__(self, answers: dict[str, str], reasoning: str = "Let me think step by step about this."):
        self.answers, self.reasoning = answers, reasoning

    def render(self, messages, thinking):
        return ("<think>\n" if thinking else "<think></think>") + "\x00" + messages[-1]["content"]

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompts, sampling, max_new_tokens, seed):
        out = []
        for p in prompts:
            thinking = p.startswith("<think>\n")
            ans = self.answers.get(p.split("\x00", 1)[1], "")
            text = f"{self.reasoning}\n</think>\n\n{ans}" if thinking else ans
            out.append(Generation(text, self.count_tokens(text), "stop"))
        return out


def make_backend(name: str, model: str, **kw) -> Backend:
    if name == "vllm":
        return VLLMBackend(model, **{k: v for k, v in kw.items() if k in (
            "dtype", "max_model_len", "gpu_memory_utilization", "tensor_parallel_size", "enforce_eager", "seed")})
    if name == "hf":
        return HFBackend(model, **{k: v for k, v in kw.items() if k in ("dtype", "batch_size", "seed")})
    raise ValueError(f"unknown backend {name!r}")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
