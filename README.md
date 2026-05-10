# Thinking First, Failing Later: Two-Stage Prompting Does Not Mitigate MCQ Positional Bias

This repository contains the full code and experiment pipeline for a benchmarking study on multiple-choice selection bias in large language models.

**Short version:** the hypothesis was wrong. Two-stage prompting (separating free-text answer generation from final option selection) does not reliably mitigate positional bias in MCQ evaluation, and in most settings makes accuracy and bias worse. Cyclic permutation remains the stronger robustness intervention. PriDe provides a modest gain on the one model that supports it.

---

## Research Question

Can multiple-choice selection bias be reduced at the prompting stage by separating answer generation from option selection?

The motivating intuition is straightforward: if option labels distort model behavior, then reasoning should happen *before* labels are introduced. This project tests that idea in a controlled benchmark across four models and two benchmarks, and finds it insufficient.

## Experiment Design

Four conditions are compared across 1,000 stratified questions per benchmark:

| Condition | Description |
|---|---|
| **Baseline** | Direct MCQ answering, single prompt |
| **Two-Stage** | Free-text answer generation, then option matching |
| **Cyclic** | Four cyclic rotations of answer options; majority vote |
| **PriDe** | Logprob-based positional prior debiasing (Zheng et al., ICLR 2024) |

PriDe requires first-token logprobabilities and runs on Together AI only (Qwen model).

**Benchmarks:** MMLU and ARC-Challenge (robustness split, 1,000 questions each)

**Models:**

| Model | Provider |
|---|---|
| GPT-4.1-mini | OpenAI |
| Gemini 2.5 Flash | Google |
| Llama 3.1 8B Instant | Groq |
| Qwen 2.5 7B Turbo | Together AI |

This is a robustness study, not a capability benchmark. The primary metric is end-to-end accuracy (correct / total), which includes failures from unscorable outputs and provider errors. Conditional accuracy (correct / scored) is reported as a secondary metric. Positional bias is quantified as mean absolute deviation from the ground-truth answer-position distribution.

## Main Findings

**Two-stage prompting hurts accuracy and increases bias across the board.** On MMLU, average E2E accuracy drops 2.4pp relative to baseline; on ARC-Challenge, 3.7pp. Mean absolute deviation from the ground-truth option distribution increases under two-stage across all models. Gemini 2.5 Flash shows the largest effect, partly driven by parse failures on the option-matching stage.

**Cyclic permutation is the strongest robustness method.** Average accuracy is at or above baseline on both benchmarks (+1.0pp MMLU, +0.6pp ARC), and positional bias is consistently lower than baseline (0.34pp MAD on ARC vs. 0.44pp baseline).

**PriDe provides a consistent but modest gain on Qwen.** +1.1pp on MMLU, +0.3pp on ARC, with lower bias than baseline on ARC. Given it requires logprob access unavailable on most closed-source APIs, the practical tradeoff is narrow.

**Unscorable burden is not a footnote.** Conditional accuracy (correct / scored) can remain high while end-to-end performance collapses. Any evaluation that reports only conditional accuracy will overstate the practical usefulness of two-stage methods.

## Repository Structure

```
two-prompt-research/
├── config/
│   └── default.yaml          # job matrix, model configs, rate limits
├── data/                     # benchmark data, normalized CSVs, stratified splits
├── runs/                     # raw per-job CSVs (gitignored)
├── scripts/
│   ├── run_experiment.py     # overnight runner
│   ├── evaluate_run.py       # accuracy, bias, overlap, per-subject stats
│   ├── aggregate_results.py  # paper-ready tables (text + LaTeX)
│   ├── generate_figures.py   # paper-ready matplotlib figures
│   └── prepare_data.py       # one-time data preprocessing
├── src/twoprompt/
│   ├── clients/              # async provider clients (OpenAI, Gemini, Groq, Together)
│   ├── runners/              # condition runners (baseline, two-stage, cyclic, PriDe)
│   ├── infra/                # disk cache, checkpointing
│   ├── benchmarks/           # benchmark loaders (MMLU, ARC-Challenge)
│   ├── parsing/              # answer parser
│   ├── scoring/              # scorer
│   └── pipeline/             # prompt builder
├── prompts/v1/               # prompt templates
└── tests/                    # test suite
```

`reports/` and `checkpoints/` are gitignored and generated locally by the pipeline.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env          # fill in API keys
```

Required keys in `.env`: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`

## Configuration

All run settings live in `config/default.yaml`. Edit that file to change models, methods, benchmarks, rate limits, or checkpoint frequency. The modules under `src/twoprompt/config/` are kept for backward compatibility and should not be edited for run configuration.

## Running the Pipeline

```bash
# Dry run: cost and time estimate, no API calls
python scripts/run_experiment.py --dry-run

# Full run (prompts for confirmation)
python scripts/run_experiment.py

# Skip confirmation
python scripts/run_experiment.py --yes

# Resume a previous run (skips completed jobs via checkpointing)
python scripts/run_experiment.py --run-id <RUN_ID> --yes
```

## Evaluation and Reporting

```bash
# Evaluate one benchmark — writes CSVs to reports/<RUN_ID>/<benchmark>/
python scripts/evaluate_run.py <RUN_ID> --benchmark mmlu --apply-fallback
python scripts/evaluate_run.py <RUN_ID> --benchmark arc --apply-fallback

# Aggregate into paper tables (text + LaTeX)
python scripts/aggregate_results.py <RUN_ID> --benchmark mmlu
python scripts/aggregate_results.py <RUN_ID> --benchmark arc_challenge

# Cross-benchmark comparison table
python scripts/aggregate_results.py <RUN_ID> --cross-benchmark

# Generate paper figures (PDF + PNG)
python scripts/generate_figures.py <RUN_ID> --benchmark mmlu
python scripts/generate_figures.py <RUN_ID> --benchmark arc_challenge
```

`--apply-fallback` substitutes baseline results for unscorable two-stage rows at eval time, without modifying the raw run CSVs.

Paper outputs are written to `reports/<RUN_ID>/<benchmark>/paper/`.

## Metrics

**End-to-end accuracy** (`correct / total`) is the headline metric. It includes provider failures and unscorable outputs in the denominator.

**Conditional accuracy** (`correct / scored`) is supplementary.

**Mean absolute deviation** from the ground-truth answer-position distribution is the primary bias metric, reported with 95% bootstrap confidence intervals (10,000 resamples).

Additional outputs: question-level overlap vs. baseline, choice shift analysis (broken/fixed per method), per-subject accuracy, and two-stage method metrics (free-text availability, latency).

## AI Usage Disclosure

The research question, hypothesis, experiment design, evaluation framing, and interpretation of results are my own. The core implementation was written primarily by me.

Infrastructure (checkpointing, response caching, retry/backoff, rate-limit settings, YAML config, overnight orchestrator) and evaluation/reporting scripts were written with substantial AI assistance under my direction and reviewed manually. The test suite used AI assistance for fixtures and scaffolding. Metric definitions and any logic that directly affects paper claims were reviewed and corrected manually.

AI tools were used for wording support during paper drafting. Design decisions and conclusions are my own.

## Why a Negative Result

Negative results in evaluation methodology are undersupplied. The intuition behind two-stage prompting is reasonable enough that it is worth documenting why it fails under a stricter robustness evaluation, particularly the unscorable burden problem, which is easy to miss if conditional accuracy is the only reported metric.

---

**Author:** Karl Hanna
