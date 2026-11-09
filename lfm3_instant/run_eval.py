"""Baseline evaluation: LFM3-1.7B with thinking ON vs OFF.

Example (Kaggle, GPU):
    python -m lfm3_instant.run_eval --model LiquidAI/LFM3-1.7B --limit 100 --out results/baseline
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .benchmarks import ensure_data, get_benchmarks
from .generate import SAMPLING_PRESETS, Backend, Timer, make_backend
from .parsing import leaked_thinking, prompt_opens_thinking, split_thinking

MODES = {"think": True, "nothink": False}


def check_template(backend: Backend, messages: list[dict], out_dir: Path) -> bool:
    """Render one prompt in both modes and save them. Returns False if the switch does nothing."""
    on, off = backend.render(messages, True), backend.render(messages, False)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "template_check.txt").write_text(
        f"===== enable_thinking=True =====\n{on}\n\n===== enable_thinking=False =====\n{off}\n")
    if on == off:
        print("WARNING: the chat template renders identical prompts for thinking on/off; "
              "enable_thinking is being ignored. Results for the two modes will not differ.")
    return on != off


def _load_cache(path: Path, ids: list[str]) -> list[dict] | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    return rows if [r["id"] for r in rows] == ids else None


def run(args) -> dict:
    out = Path(args.out)
    data_dir, vendor_dir = Path(args.data_dir), Path(args.vendor_dir)
    ensure_data(data_dir)
    benches = get_benchmarks(args.benchmarks, vendor_dir)
    items = {b.name: b.load(data_dir, args.limit, args.seed) for b in benches}

    backend = args.backend_obj if getattr(args, "backend_obj", None) else make_backend(
        args.backend, args.model, dtype=args.dtype, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization, tensor_parallel_size=args.tp,
        enforce_eager=args.enforce_eager, batch_size=args.batch_size, seed=args.seed)
    template_ok = check_template(backend, items[benches[0].name][0].messages, out)

    summary = {"model": args.model, "limit": args.limit, "seed": args.seed,
               "max_new_tokens": args.max_new_tokens, "template_switch_works": template_ok, "rows": []}
    for mode in args.modes:
        thinking = MODES[mode]
        for b in benches:
            its = items[b.name]
            mdir = out / mode
            mdir.mkdir(parents=True, exist_ok=True)
            gen_path = mdir / f"{b.name}.gen.jsonl"
            sampling = SAMPLING_PRESETS[b.sampling]
            prompts = [backend.render(it.messages, thinking) for it in its]

            cached = _load_cache(gen_path, [it.id for it in its])
            if cached is None:
                print(f"\n== generating {b.name} [{mode}] n={len(its)}", flush=True)
                with Timer() as t:
                    gens = backend.generate(prompts, sampling, args.max_new_tokens, args.seed)
                with open(gen_path, "w") as f:
                    for it, g in zip(its, gens):
                        f.write(json.dumps({"id": it.id, "text": g.text, "n_tokens": g.n_tokens,
                                            "finish_reason": g.finish_reason, "gen_seconds": t.seconds / len(its)}) + "\n")
                cached = _load_cache(gen_path, [it.id for it in its])
            else:
                print(f"\n== reusing cached generations for {b.name} [{mode}]")

            records = []
            for it, p, g in zip(its, prompts, cached):
                sp = split_thinking(g["text"], prompt_opens_thinking(p))
                records.append({
                    "id": it.id, "bench": b.name, "mode": mode, "raw": g["text"],
                    "reasoning": sp.reasoning, "answer": sp.answer, "think_status": sp.status,
                    "n_tokens": g["n_tokens"], "n_reasoning_tokens": backend.count_tokens(sp.reasoning),
                    "truncated": g["finish_reason"] == "length",
                    "leak": (not thinking) and leaked_thinking(g["text"]),
                    "gen_seconds": g["gen_seconds"],
                })
            scores = b.score(its, [r["answer"] for r in records], mdir / "scoring")
            for r, s in zip(records, scores):
                r["score"] = s
            with open(mdir / f"{b.name}.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r, default=str) + "\n")

            n = len(records)
            toks = [r["n_tokens"] for r in records]
            row = {
                "mode": mode, "bench": b.name, "n": n, "metric": b.primary,
                "accuracy": sum(bool(r["score"][b.primary]) for r in records) / n,
                "mean_tokens": statistics.mean(toks), "median_tokens": statistics.median(toks),
                "mean_reasoning_tokens": statistics.mean(r["n_reasoning_tokens"] for r in records),
                "truncated_rate": sum(r["truncated"] for r in records) / n,
                "unclosed_think_rate": sum(r["think_status"] == "unclosed" for r in records) / n,
                "leak_rate": sum(r["leak"] for r in records) / n,
                "gen_seconds": sum(r["gen_seconds"] for r in records),
            }
            for extra in ("base_pass", "prompt_loose", "inst_strict"):
                if extra in records[0]["score"]:
                    row[extra] = sum(float(r["score"][extra]) for r in records) / n
            summary["rows"].append(row)
            print(f"   {b.name} [{mode}] {b.primary}={row['accuracy']:.3f}  mean tokens={row['mean_tokens']:.0f}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(to_markdown(summary))
    print("\n" + to_markdown(summary))
    return summary


def to_markdown(summary: dict) -> str:
    rows = summary["rows"]
    lines = [f"# Baseline: {summary['model']}  (limit={summary['limit']}, seed={summary['seed']}, "
             f"max_new_tokens={summary['max_new_tokens']})", ""]
    if not summary["template_switch_works"]:
        lines += ["**WARNING: enable_thinking had no effect on the prompt; modes are identical.**", ""]
    lines += ["| benchmark | mode | n | score | mean tokens | median tokens | reasoning tokens | truncated | leaked thinking | gen time (s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['bench']} ({r['metric']}) | {r['mode']} | {r['n']} | {r['accuracy']:.1%} | "
                     f"{r['mean_tokens']:.0f} | {r['median_tokens']:.0f} | {r['mean_reasoning_tokens']:.0f} | "
                     f"{r['truncated_rate']:.1%} | {r['leak_rate']:.1%} | {r['gen_seconds']:.0f} |")
    by = {(r["bench"], r["mode"]): r for r in rows}
    pairs = [b for b in dict.fromkeys(r["bench"] for r in rows) if (b, "think") in by and (b, "nothink") in by]
    if pairs:
        lines += ["", "## Thinking ON vs OFF", "",
                  "| benchmark | score ON | score OFF | gap (points) | tokens ON | tokens OFF | token reduction |",
                  "|---|---|---|---|---|---|---|"]
        for b in pairs:
            on, off = by[(b, "think")], by[(b, "nothink")]
            red = f"{on['mean_tokens'] / off['mean_tokens']:.1f}x" if off["mean_tokens"] else "n/a"
            lines.append(f"| {b} | {on['accuracy']:.1%} | {off['accuracy']:.1%} | "
                         f"{(on['accuracy'] - off['accuracy']) * 100:+.1f} | {on['mean_tokens']:.0f} | "
                         f"{off['mean_tokens']:.0f} | {red} |")
        lines += ["", "LFM3-Instant's goal: keep the OFF token count while closing the score gap."]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--modes", nargs="+", default=["think", "nothink"], choices=list(MODES))
    p.add_argument("--benchmarks", nargs="+", default=["humaneval", "mbpp", "gsm8k", "ifeval"])
    p.add_argument("--limit", type=int, default=None, help="problems per benchmark (seeded random subset)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=8192, help="same cap for both modes, for fairness")
    p.add_argument("--out", default="results/baseline")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--vendor-dir", default="vendor")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--tp", type=int, default=1, help="tensor parallel GPUs (2 on Kaggle's T4 x2)")
    p.add_argument("--enforce-eager", action="store_true", help="try this if vLLM fails to capture CUDA graphs")
    p.add_argument("--batch-size", type=int, default=8, help="hf backend only")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
