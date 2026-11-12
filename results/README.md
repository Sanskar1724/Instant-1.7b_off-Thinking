# Results

Every experiment in this project, in the order it was run. Each folder has its own README with the details, charts and raw files.

All model runs were done on Kaggle (2x T4, float16) with `LiquidAI/LFM3-1.7B` or our fine-tune of it. All runs used the same seed (0), Liquid's recommended sampling, and the same limit of 4,096 new tokens (8,192 for LiveCodeBench).

**All answers on Hugging Face:** [EquilStable/lfm3-1.7b-thinking-on-off-eval](https://huggingface.co/datasets/EquilStable/lfm3-1.7b-thinking-on-off-eval). It has every answer from thinking ON, thinking OFF and Instant round 1, each with its score, token count, cut-off flag and hidden-reasoning flag.

## Overview

| Folder | Step | What it is | Key result |
|---|---|---|---|
| [baseline_v1](baseline_v1/README.md) | 1 | First small ON vs OFF run, 30 questions per benchmark | First sign that "thinking OFF" still reasons in plain text |
| [baseline_v2](baseline_v2/README.md) | 1 | **Main baseline**: ON vs OFF on 600 questions (100 per benchmark; LiveCodeBench 30, BFCL 70) | ON 72.6%, OFF 69.4%; OFF still reasons in 24% of answers (HumanEval+ 70%) |
| [step2_data](step2_data/README.md) | 2 | Training data: the model's own verified answers, thinking removed | 2,069 examples (1,966 train, 103 val), 223 tokens on average |
| [step3_lora](step3_lora/README.md) | 3 | First LoRA fine-tune (r=16, 2 epochs, 31 min on one T4) | On held-out questions: hidden reasoning 11% → 0%, cut-offs 15% → 3% |
| [step4_eval](step4_eval/README.md) | 4 | Full benchmark of the fine-tune (Instant round 1), same 600 questions | 1.8x fewer tokens than OFF and 3.3x fewer than ON, but 3.8 points below OFF |

## Main numbers so far (600 questions, same settings for all three)

| | LFM3 thinking ON | LFM3 thinking OFF | Instant round 1 |
|---|---|---|---|
| Average accuracy (7 benchmarks) | **72.6%** | 69.4% | 65.6% |
| Mean tokens per answer | 2,226 | 1,206 | **674** |
| Answers cut off at the token limit | 18% | 6.5% | **2%** |
| Answers with hidden reasoning | – | 24% | 6% |

| Benchmark | ON | OFF | Instant r1 |
|---|---|---|---|
| HumanEval+ | 82 | 83 | 74 |
| MBPP+ | 80 | 69 | 65 |
| LiveCodeBench v6 | 60 | 67 | 60 |
| GSM8K | 85 | 88 | 88 |
| IFEval | 77 | 75 | 72 |
| IFBench | 34 | 27 | 23 |
| BFCL | 90 | 77 | 77 |

**In short:**
- Turning thinking off in LFM3 does not reliably stop it from thinking, and where it really does answer directly, it loses accuracy.
- The first fine-tune makes answers much shorter and more predictable, but it removed the hidden reasoning that was helping on code.
- Round 2 adds short visible plans and more hard examples to win that accuracy back.

> **Correction:** earlier READMEs list the thinking-ON average as 73.0%. The correct mean of the seven benchmarks is **72.6%** (82+80+60+85+77+34+90 = 508, ÷ 7), so the ON − OFF gap is 3.2 points, not 3.6. The per-benchmark numbers were always correct.

## How to read the numbers

- **Small samples.** One question is worth 1 point on most benchmarks, 3.3 on LiveCodeBench and 1.4 on BFCL. Differences under about 10 points on a single benchmark are not statistically clear. The paired tests in `step4_eval/paired_tests.csv` show which differences are clear.
- **Hidden reasoning** means thinking was OFF, but the model still reasoned in plain text before the answer.
- **Cut off** means the answer hit the token limit. A cut-off answer whose reasoning never closed is counted as wrong.
- **Our thinking-ON scores are lower than Liquid's published ones** because Liquid uses no output limit and the full benchmarks. With a realistic limit of 4,096 tokens, thinking ON often runs out before it answers (58% of IFBench answers).

## Next

Round 2, which uses three notebooks in `notebooks/`:
1. `Step2b_Training_Data_Round2` builds the new data.
2. `Step3b_LoRA_Round2` fine-tunes on it.
3. `Step4b_Instant_R2_vs_Base` benchmarks the result.

The goal is to at least match thinking OFF's accuracy while staying clearly shorter. Results will be added here as `step2b_data`, `step3b_lora` and `step4b_eval`.
