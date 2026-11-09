"""Parse raw model outputs: split thinking from the answer, extract code and numbers."""
from __future__ import annotations

import re
from dataclasses import dataclass

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class SplitOutput:
    reasoning: str
    answer: str
    status: str  # "closed" (thinking finished), "unclosed" (thinking never finished), "none" (no thinking)


def prompt_opens_thinking(rendered_prompt: str) -> bool:
    """True if the chat template already opened a <think> block that the model must close."""
    tail = rendered_prompt.rstrip()
    last_open = tail.rfind(THINK_OPEN)
    return last_open != -1 and tail.find(THINK_CLOSE, last_open) == -1


def split_thinking(raw: str, prompt_opened_think: bool = False) -> SplitOutput:
    """Separate the reasoning trace from the final answer.

    If the trace never closes (truncated or runaway thinking), the answer is empty,
    so the sample is scored as a failure rather than graded on half-finished reasoning.
    """
    if THINK_CLOSE in raw:
        pre, post = raw.rsplit(THINK_CLOSE, 1)
        return SplitOutput(pre.replace(THINK_OPEN, "").strip(), post.strip(), "closed")
    if THINK_OPEN in raw or prompt_opened_think:
        return SplitOutput(raw.split(THINK_OPEN, 1)[-1].strip(), "", "unclosed")
    return SplitOutput("", raw.strip(), "none")


def leaked_thinking(raw: str) -> bool:
    """In no-thinking mode, did the model emit thinking tags anyway?"""
    return THINK_OPEN in raw or THINK_CLOSE in raw


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(answer: str, entry_point: str | None = None) -> str:
    """Pull Python code out of a chat answer.

    Prefers the last fenced block that defines `entry_point`, then the longest
    python block, then the whole answer if there are no fences.
    """
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(answer)]
    if not blocks:
        # An unterminated fence (e.g. answer cut off) still often holds usable code.
        m = re.search(r"```[ \t]*[A-Za-z0-9_+-]*[^\n]*\n(.*)", answer, re.DOTALL)
        return (m.group(1) if m else answer).strip("\n")
    py = [b for lang, b in blocks if lang in ("", "python", "py", "python3")] or [b for _, b in blocks]
    if entry_point:
        defining = [b for b in py if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", b)]
        if defining:
            return defining[-1].strip("\n")
    return max(py, key=len).strip("\n")


def leading_imports(code: str) -> str:
    """Import lines from the top-level of a snippet (used to carry HumanEval prompt imports)."""
    return "\n".join(
        line for line in code.splitlines() if re.match(r"^(import|from)\s+\S+", line)
    )


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _boxed_contents(text: str) -> list[str]:
    """Contents of every \\boxed{...}, handling nested braces."""
    out, i = [], 0
    while True:
        i = text.find("\\boxed{", i)
        if i == -1:
            return out
        j, depth = i + len("\\boxed{"), 1
        start = j
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out.append(text[start : j - 1] if depth == 0 else text[start:])
        i = j


def extract_final_number(text: str) -> float | None:
    """Final numeric answer: \\boxed{} first, then '####', then 'answer is', then last number."""
    boxed = _boxed_contents(text)
    if boxed:
        candidate = boxed[-1]
    elif "####" in text:
        candidate = text.rsplit("####", 1)[-1]
    else:
        m = re.findall(r"answer\s*(?:is|:)\s*([^\n]+)", text, flags=re.IGNORECASE)
        candidate = m[-1] if m else text
    nums = _NUM.findall(candidate) or _NUM.findall(text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def numbers_match(pred: float | None, gold: float, tol: float = 1e-4) -> bool:
    return pred is not None and abs(pred - gold) <= tol * max(1.0, abs(gold))
