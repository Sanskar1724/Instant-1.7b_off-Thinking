"""Tests: parsing units, plus an end-to-end run with a mock model.

Correct answers must score ~100%, wrong answers 0%, in both modes.
Run: pytest -q tests
"""
import gzip
import json
import sys
from pathlib import Path

import pytest

from lfm3_instant.benchmarks import ensure_data, get_benchmarks
from lfm3_instant.generate import MockBackend
from lfm3_instant.parsing import (extract_code, extract_final_number, leaked_thinking,
                                  prompt_opens_thinking, split_thinking)
from lfm3_instant.run_eval import parse_args, run

ROOT = Path(__file__).resolve().parents[1]
DATA, VENDOR = ROOT / "data", ROOT / "vendor"
GPT4_IFEVAL = "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_response_data_gpt4_20231107_145030.jsonl"


# ------------------------------------------------------------------ parsing

def test_split_closed_thinking():
    s = split_thinking("<think>plan</think>\n\nAnswer: 4")
    assert (s.reasoning, s.answer, s.status) == ("plan", "Answer: 4", "closed")


def test_split_when_template_prefills_open_tag():
    s = split_thinking("plan here\n</think>\nfinal", prompt_opened_think=True)
    assert (s.reasoning, s.answer) == ("plan here", "final")


def test_split_truncated_thinking_has_no_answer():
    s = split_thinking("still thinking and never closed", prompt_opened_think=True)
    assert s.answer == "" and s.status == "unclosed"


def test_split_plain_answer():
    s = split_thinking("just the answer")
    assert (s.reasoning, s.answer, s.status) == ("", "just the answer", "none")


def test_prompt_opens_thinking():
    assert prompt_opens_thinking("<|im_start|>assistant\n<think>\n")
    assert not prompt_opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not prompt_opens_thinking("<|im_start|>assistant\n")


def test_leak_detection():
    assert leaked_thinking("<think>hmm</think> 5") and not leaked_thinking("5")


def test_extract_code_prefers_block_defining_entry_point():
    ans = "Helper:\n```python\nx = 1\n```\nSolution:\n```python\ndef f(a):\n    return a\n```\n"
    assert extract_code(ans, "f").startswith("def f(a)")


def test_extract_code_unterminated_fence():
    assert extract_code("```python\ndef f():\n    return 1\n", "f").startswith("def f()")


@pytest.mark.parametrize("text,expected", [
    ("so the total is \\boxed{1,234}", 1234),
    ("\\boxed{\\$18}", 18),
    ("work...\n#### 72", 72),
    ("The answer is 3.5 apples", 3.5),
    ("first 10 then 20", 20),
    ("no digits", None),
])
def test_extract_final_number(text, expected):
    assert extract_final_number(text) == expected


# ------------------------------------------------------------------ end to end

# evalplus's sandboxed test runner hangs on Windows; run these on Linux / WSL / Kaggle.
pytestmark_e2e = pytest.mark.skipif(sys.platform == "win32", reason="evalplus hangs on Windows; run on Linux")

def _oracle_answers():
    """Reference answers for each benchmark, keyed by the user message the harness builds."""
    ensure_data(DATA)
    benches = {b.name: b for b in get_benchmarks(["humaneval", "mbpp", "gsm8k", "ifeval"], VENDOR)}
    he = {r["task_id"]: r for r in (json.loads(l) for l in gzip.open(DATA / "HumanEvalPlus.jsonl.gz", "rt"))}
    mb = {r["task_id"]: r for r in (json.loads(l) for l in gzip.open(DATA / "MbppPlus.jsonl.gz", "rt"))}
    gpt4_path = DATA / "ifeval_gpt4_responses.jsonl"
    if not gpt4_path.exists():
        import urllib.request
        urllib.request.urlretrieve(GPT4_IFEVAL, gpt4_path)
    gpt4 = {r["prompt"]: r["response"] for r in (json.loads(l) for l in open(gpt4_path))}

    good, bad = {}, {}
    for name, b in benches.items():
        for it in b.load(DATA, None, 0):
            q = it.messages[-1]["content"]
            if name == "humaneval":
                r = he[it.id]
                good[q] = f"Here you go:\n```python\n{r['prompt']}{r['canonical_solution']}```"
                bad[q] = f"```python\ndef {r['entry_point']}(*a, **k):\n    return None\n```"
            elif name == "mbpp":
                good[q] = f"```python\n{mb[it.id]['canonical_solution']}\n```"
                bad[q] = "```python\nprint('hi')\n```"
            elif name == "gsm8k":
                g = it.meta["gold"]
                good[q] = f"Working it out... \\boxed{{{int(g) if g == int(g) else g}}}"
                bad[q] = "\\boxed{-999999}"
            else:
                good[q] = gpt4.get(it.meta["row"]["prompt"], "")
                bad[q] = ""
    return good, bad


@pytest.fixture(scope="module")
def answers():
    return _oracle_answers()


def _run(tmp_path, backend, limit=40):
    args = parse_args(["--limit", str(limit), "--out", str(tmp_path), "--data-dir", str(DATA),
                       "--vendor-dir", str(VENDOR)])
    args.backend_obj = backend
    s = run(args)
    return {(r["bench"], r["mode"]): r for r in s["rows"]}


@pytestmark_e2e
def test_correct_answers_score_high_in_both_modes(tmp_path, answers):
    rows = _run(tmp_path, MockBackend(answers[0]))
    for mode in ("think", "nothink"):
        assert rows[("humaneval", mode)]["accuracy"] >= 0.97
        assert rows[("mbpp", mode)]["accuracy"] >= 0.97
        assert rows[("gsm8k", mode)]["accuracy"] == 1.0
        assert 0.6 <= rows[("ifeval", mode)]["accuracy"] <= 0.95  # GPT-4 scored ~0.77
        assert rows[("gsm8k", mode)]["leak_rate"] == 0
    # thinking mode must cost more tokens and report reasoning tokens
    assert rows[("gsm8k", "think")]["mean_tokens"] > rows[("gsm8k", "nothink")]["mean_tokens"]
    assert rows[("gsm8k", "think")]["mean_reasoning_tokens"] > 0
    assert rows[("gsm8k", "nothink")]["mean_reasoning_tokens"] == 0
    assert (tmp_path / "summary.md").read_text().count("| humaneval") == 3


@pytestmark_e2e
def test_wrong_answers_score_zero(tmp_path, answers):
    rows = _run(tmp_path, MockBackend(answers[1]), limit=20)
    assert all(r["accuracy"] == 0 for r in rows.values())


@pytestmark_e2e
def test_leaked_thinking_in_nothink_mode_is_flagged_but_still_scored(tmp_path, answers):
    class Leaky(MockBackend):
        def generate(self, prompts, *a, **k):
            gens = super().generate(prompts, *a, **k)
            for g in gens:
                if "</think>" not in g.text:
                    g.text = "<think>oops</think>\n" + g.text
            return gens

    rows = _run(tmp_path, Leaky(answers[0]), limit=10)
    assert rows[("gsm8k", "nothink")]["leak_rate"] == 1.0
    assert rows[("gsm8k", "nothink")]["accuracy"] == 1.0


@pytestmark_e2e
def test_cache_is_reused(tmp_path, answers):
    _run(tmp_path, MockBackend(answers[0]), limit=5)

    class Exploding(MockBackend):
        def generate(self, *a, **k):
            raise AssertionError("should have used the cache")

    _run(tmp_path, Exploding(answers[0]), limit=5)
