# Instant-1.7B

Making [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) fast **without its thinking step**, while keeping as much of its thinking-mode accuracy as possible.

LFM3-1.7B is a very strong small model, but it thinks by default, and that costs many tokens on phones and laptops, exactly where a 1.7B model is meant to run. Thinking can be switched off (`enable_thinking=False`), but no one has published how much quality that loses. This project:

1. **Measures it.** An honest thinking-ON vs thinking-OFF baseline on HumanEval+, MBPP+, GSM8K and IFEval (accuracy, tokens, truncation, leaked thinking, speed).
2. **Fixes it** (next). Fine-tunes the model on its own verified answers so it answers directly, with fewer tokens, and checks the result at Q4 GGUF too.

> Status: evaluation harness done and tested; baseline run on Kaggle in progress. No results yet.

## Repository layout

```
lfm3_instant/
  parsing.py      split thinking vs answer, code / number extraction, leak detection
  benchmarks.py   HumanEval+ & MBPP+ (official evalplus scoring), GSM8K, IFEval (Google's checker)
  generate.py     backends: vLLM (GPU), transformers (fallback), mock (tests, no model)
  run_eval.py     CLI: runs both modes, caches generations, writes summary.md / summary.json
tests/            parsing unit tests + end-to-end runs with a mock model
notebooks/
  LFM3_Thinking_ON_vs_OFF.ipynb   Experiment 1: the full ON vs OFF study in one readable Kaggle notebook
  kaggle_baseline.ipynb           earlier vLLM-based version
```

## Quick start

Run the tests (CPU only, no model needed):

```bash
pip install -r requirements.txt
pytest -q tests
```

The end-to-end tests use evalplus's code sandbox, which hangs on Windows, so they are skipped there. Run them on Linux, WSL or Kaggle.

Run the baseline on a GPU:

```bash
pip install vllm
python -m lfm3_instant.run_eval --model LiquidAI/LFM3-1.7B --limit 20 --out results/smoke
python -m lfm3_instant.run_eval --model LiquidAI/LFM3-1.7B --limit 100 --out results/baseline
# fallback if vLLM fails: --backend hf
```

On Kaggle: import `notebooks/LFM3_Thinking_ON_vs_OFF.ipynb`, set *Accelerator* to **GPU T4 x2** and *Internet* to **On**, then *Run All*. It checks the scorers first (reference answers must score ~100%), then runs both modes and shows tables, charts, failure examples, a speed test and an interactive ON/OFF sandbox.

## Method notes

- Both modes use the same settings: same prompts, same `max_new_tokens` (8192), same seed, and Liquid's recommended sampling (code/general: temperature 0.1; math: temperature 1.0).
- If thinking never closes (runaway or truncated), the answer counts as wrong instead of being graded on half-finished reasoning.
- Benchmark subsets are seeded random samples, so runs can be reproduced.
- Scoring was checked against known references: evalplus reproduces exact pass rates, and the IFEval checker reproduces GPT-4's published score (77.0% strict vs 76.9% in the paper).
- Evaluation sets are never used for training.

## License

Code in this repository: [Apache-2.0](LICENSE).
The model itself is under the [LFM Open License v1.0](https://www.liquid.ai/lfm-license); this repo contains no model weights. Benchmarks: HumanEval (MIT), EvalPlus (Apache-2.0), GSM8K (MIT), IFEval (Apache-2.0).

This is an independent project and is not affiliated with Liquid AI.
