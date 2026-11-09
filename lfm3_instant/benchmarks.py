"""Benchmarks: loading, prompts and scoring.

Coding is scored with the official evalplus harness (HumanEval+ / MBPP+), and
instruction following with Google's original IFEval checker, so numbers are
comparable with published results.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .parsing import extract_code, extract_final_number, leading_imports, numbers_match

DATA_FILES = {
    "HumanEvalPlus.jsonl.gz": "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
    "MbppPlus.jsonl.gz": "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "ifeval_input_data.jsonl": "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl",
}
IFEVAL_CODE = "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/{}"
IFEVAL_FILES = ["instructions.py", "instructions_registry.py", "instructions_util.py", "evaluation_lib.py"]


def _download(url: str, dest: Path, attempts: int = 5) -> None:
    """Download with retries: GitHub's raw host sometimes returns 5xx / times out."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - network errors come in many types
            if k == attempts - 1:
                raise RuntimeError(f"download failed after {attempts} tries: {url}") from e
            wait = 2 ** k * 3
            print(f"  download error ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def _github_asset_api_url(url: str) -> str | None:
    """Map a github.com/<o>/<r>/releases/download/<tag>/<file> link to its API asset URL.

    The normal release link redirects through a host that some networks time out on;
    the API endpoint serves the same bytes by another route.
    """
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/([^/]+)$", url)
    if not m:
        return None
    owner, repo, tag, fname = m.groups()
    api = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    with urllib.request.urlopen(api, timeout=60) as r:
        assets = json.load(r)["assets"]
    return next((a["url"] for a in assets if a["name"] == fname), None)


def ensure_data(data_dir: Path) -> None:
    for name, url in DATA_FILES.items():
        dest = data_dir / name
        if dest.exists():
            continue
        print(f"downloading {name}")
        try:
            _download(url, dest, attempts=3)
        except RuntimeError:
            api = _github_asset_api_url(url)
            if not api:
                raise
            print(f"  retrying {name} through the GitHub API")
            req = urllib.request.Request(api, headers={"Accept": "application/octet-stream"})
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest.with_suffix(".part"), "wb") as f:
                shutil.copyfileobj(r, f)
            dest.with_suffix(".part").replace(dest)


def ensure_ifeval_code(vendor_dir: Path) -> None:
    pkg = vendor_dir / "instruction_following_eval"
    for name in IFEVAL_FILES:
        if not (pkg / name).exists():
            _download(IFEVAL_CODE.format(name), pkg / name)
    (pkg / "__init__.py").touch()
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))


@dataclass
class Item:
    bench: str
    id: str
    messages: list[dict]
    meta: dict = field(default_factory=dict)


def _jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _subset(rows: list, limit: int | None, seed: int) -> list:
    """Seeded random subset, kept in original order, so runs are reproducible and unbiased."""
    if not limit or limit >= len(rows):
        return rows
    keep = set(random.Random(seed).sample(range(len(rows)), limit))
    return [r for i, r in enumerate(rows) if i in keep]


class Benchmark:
    name: str
    primary: str  # key of the headline metric in per-item scores
    sampling: str  # "general" or "math" (Liquid's recommended presets)

    def load(self, data_dir: Path, limit: int | None, seed: int) -> list[Item]: ...

    def score(self, items: list[Item], answers: list[str], work_dir: Path) -> list[dict]: ...


# ---------------------------------------------------------------- coding (evalplus)

CODE_INSTRUCTION = (
    "Return the complete function implementation, including the signature and any "
    "imports it needs, in a single ```python code block."
)


class _EvalPlus(Benchmark):
    primary = "plus_pass"
    sampling = "general"
    dataset: str  # evalplus dataset name
    file: str
    override_env: str

    def _rows(self, data_dir: Path) -> list[dict]:
        return _jsonl(data_dir / self.file)

    def score(self, items, answers, work_dir):
        data_dir = Path(items[0].meta["data_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        by_id = {it.id: (it, a) for it, a in zip(items, answers)}
        samples = work_dir / f"{self.name}_samples.jsonl"
        with open(samples, "w") as f:
            # evalplus requires every problem; unselected ones get an empty solution and are ignored below.
            for row in self._rows(data_dir):
                sol = ""
                if row["task_id"] in by_id:
                    it, ans = by_id[row["task_id"]]
                    sol = self.to_solution(it, ans)
                f.write(json.dumps({"task_id": row["task_id"], "solution": sol}) + "\n")
        results = work_dir / f"{self.name}_samples_eval_results.json"
        results.unlink(missing_ok=True)  # evalplus silently reuses stale results otherwise
        env = {**os.environ, self.override_env: str((data_dir / self.file).resolve())}
        # Use the running interpreter: a PATH lookup can find another Python's evalplus.
        cmd = [sys.executable, "-m", "evalplus.evaluate"] + [
            self.dataset, "--samples", str(samples), "--parallel", str(max(1, (os.cpu_count() or 2))),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0 or not results.exists():
            raise RuntimeError(f"evalplus failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        ev = json.load(open(results))["eval"]
        out = []
        for it in items:
            r = ev[it.id][0]
            base = r["base_status"] == "pass"
            out.append({"base_pass": base, "plus_pass": base and r["plus_status"] == "pass"})
        return out


class HumanEvalPlus(_EvalPlus):
    name, dataset, file, override_env = "humaneval", "humaneval", "HumanEvalPlus.jsonl.gz", "HUMANEVAL_OVERRIDE_PATH"

    def load(self, data_dir, limit, seed):
        items = []
        for r in _subset(self._rows(data_dir), limit, seed):
            prompt = f"Complete the following Python function.\n\n```python\n{r['prompt'].rstrip()}\n```\n\n{CODE_INSTRUCTION}"
            items.append(Item(self.name, r["task_id"], [{"role": "user", "content": prompt}],
                              {"entry_point": r["entry_point"], "prompt": r["prompt"], "data_dir": str(data_dir)}))
        return items

    def to_solution(self, it, answer):
        code = extract_code(answer, it.meta["entry_point"])
        if f"def {it.meta['entry_point']}" not in code:  # model wrote only the body
            return it.meta["prompt"] + code
        return leading_imports(it.meta["prompt"]) + "\n\n" + code


class MbppPlus(_EvalPlus):
    name, dataset, file, override_env = "mbpp", "mbpp", "MbppPlus.jsonl.gz", "MBPP_OVERRIDE_PATH"

    def load(self, data_dir, limit, seed):
        items = []
        for r in _subset(self._rows(data_dir), limit, seed):
            task = r["prompt"].strip().strip('"').strip()
            prompt = (f"{task}\n\nThe function must be named `{r['entry_point']}` and pass the test above.\n\n"
                      f"{CODE_INSTRUCTION}")
            items.append(Item(self.name, r["task_id"], [{"role": "user", "content": prompt}],
                              {"entry_point": r["entry_point"], "data_dir": str(data_dir)}))
        return items

    def to_solution(self, it, answer):
        return extract_code(answer, it.meta["entry_point"])


# ---------------------------------------------------------------- math

class GSM8K(Benchmark):
    name, primary, sampling = "gsm8k", "correct", "math"

    def load(self, data_dir, limit, seed):
        rows = _jsonl(data_dir / "gsm8k_test.jsonl")
        items = []
        for i, r in enumerate(rows):
            gold = float(r["answer"].rsplit("####", 1)[-1].strip().replace(",", ""))
            prompt = f"{r['question']}\n\nSolve the problem and put the final numeric answer inside \\boxed{{}}."
            items.append(Item(self.name, f"gsm8k/{i}", [{"role": "user", "content": prompt}], {"gold": gold}))
        return _subset(items, limit, seed)

    def score(self, items, answers, work_dir):
        out = []
        for it, a in zip(items, answers):
            pred = extract_final_number(a)
            out.append({"correct": numbers_match(pred, it.meta["gold"]), "pred": pred})
        return out


# ---------------------------------------------------------------- instruction following

class IFEval(Benchmark):
    name, primary, sampling = "ifeval", "prompt_strict", "general"

    def __init__(self, vendor_dir: Path):
        self.vendor_dir = vendor_dir

    def load(self, data_dir, limit, seed):
        rows = _jsonl(data_dir / "ifeval_input_data.jsonl")
        items = [Item(self.name, f"ifeval/{r['key']}", [{"role": "user", "content": r["prompt"]}], {"row": r})
                 for r in rows]
        return _subset(items, limit, seed)

    def score(self, items, answers, work_dir):
        ensure_ifeval_code(self.vendor_dir)
        from instruction_following_eval import evaluation_lib as ev

        out = []
        for it, a in zip(items, answers):
            r = it.meta["row"]
            inp = ev.InputExample(key=r["key"], instruction_id_list=r["instruction_id_list"],
                                  prompt=r["prompt"], kwargs=r["kwargs"])
            strict = ev.test_instruction_following_strict(inp, {r["prompt"]: a})
            loose = ev.test_instruction_following_loose(inp, {r["prompt"]: a})
            out.append({
                "prompt_strict": strict.follow_all_instructions,
                "prompt_loose": loose.follow_all_instructions,
                "inst_strict": sum(strict.follow_instruction_list) / len(strict.follow_instruction_list),
            })
        return out


def get_benchmarks(names: list[str], vendor_dir: Path) -> list[Benchmark]:
    table = {"humaneval": HumanEvalPlus, "mbpp": MbppPlus, "gsm8k": GSM8K}
    out = []
    for n in names:
        if n == "ifeval":
            out.append(IFEval(vendor_dir))
        elif n in table:
            out.append(table[n]())
        else:
            raise ValueError(f"unknown benchmark {n!r}; choose from humaneval, mbpp, gsm8k, ifeval")
    return out
