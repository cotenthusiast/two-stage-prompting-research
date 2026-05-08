# two-prompt-research

Research project measuring positional bias in LLMs on multiple-choice benchmarks, and testing whether prompting interventions reduce that bias. Based on Zheng et al. (ICLR 2024) "Large Language Models Are Not Robust Multiple Choice Selectors".

## Setup

```bash
pip install -e ".[dev]"   # installs twoprompt package + dev deps
cp .env.example .env      # fill in API keys
```

API keys required in `.env`: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`

## Running experiments

All job configuration lives in `config/default.yaml` — edit that file to add models, adjust rate limits, or change the job matrix. Do not edit `src/twoprompt/config/` for run configuration; those files exist for backward compatibility with evaluation scripts.

```bash
# Dry run (preflight cost/time estimate, no API calls)
python scripts/run_experiment.py --dry-run

# Full overnight run (prompts for confirmation)
python scripts/run_experiment.py

# Re-run specific run ID (skips already-complete jobs via checkpointing)
python scripts/run_experiment.py --run-id 20260505_200402 --yes

# Skip cache for a run
python scripts/run_experiment.py --no-cache
```

## Evaluating a run

```bash
python scripts/evaluate_run.py <run_id> --benchmark mmlu
python scripts/evaluate_run.py <run_id> --benchmark arc     # "arc" aliases "arc_challenge"
```

Reports are written to `reports/<run_id>/<benchmark>/`. The most recent run is `20260505_200402`.

## Aggregating into paper tables

```bash
python scripts/aggregate_results.py
```

## Project structure

```
config/
  default.yaml              # job matrix, model configs, rate limits — edit this to configure runs

scripts/
  run_experiment.py         # overnight runner; reads config/default.yaml, runs jobs concurrently per model
  evaluate_run.py           # computes accuracy, positional bias, overlap, per-subject stats for one run
  aggregate_results.py      # builds paper-ready tables from report CSVs
  prepare_data.py           # one-time data preprocessing

src/twoprompt/
  clients/                  # async provider clients
    base.py                 # BaseClient with retry/rate-limit logic
    types.py                # ModelRequest, ModelResponse, error types
    openai_client.py        # OpenAI
    gemini_client.py        # Google Gemini
    groq_client.py          # Groq
    together_client.py      # Together AI (OpenAI-compatible); handles non-standard logprob format
  runners/                  # one class per experiment method
    base.py                 # ExperimentRunner base class
    direct_mcq.py           # baseline: single prompt, parse letter
    permutation.py          # cyclic: 4 rotations, majority vote
    two_stage.py            # two_prompt: free-text stage 1, matching stage 2
    two_stage_permutation.py # two_prompt_cyclic: free-text + cyclic matching
    pride.py                # PriDe: logprob-based debiasing (Together AI only)
    pride_debias.py         # PriDe math: Eq.(1), Eq.(7), Eq.(8), calibration state
  infra/
    cache.py                # disk-backed response cache (keyed by model+prompt+params)
    checkpoint.py           # per-job checkpoint save/load/delete
  config/
    experiment.py           # method/split/subject constants (do not edit for run config)
    models.py               # API key loading, provider/model constants
    paths.py                # default directory paths
  benchmarks/               # benchmark-specific loaders (mmlu.py, arc.py)
  parsing/                  # answer parser (extracts A/B/C/D from raw text)
  scoring/                  # scorer (correct/incorrect/unscorable)
  io/                       # readers (load normalized CSVs) and writers (save run CSVs)
  pipeline/
    prompt_builder.py       # fills prompt templates

prompts/v1/
  direct_mcq.txt            # template for single-call MCQ prompts
  free_text.txt             # template for stage-1 free-text elicitation
  option_matching.txt       # template for stage-2 option matching

runs/                       # raw per-job CSVs from each run, named <run_id>_<method>_<model>_<benchmark>.csv
reports/                    # aggregated evaluation outputs per run/benchmark
checkpoints/                # in-progress checkpoints (auto-deleted on job completion)
.cache/responses/           # disk cache for API responses (SHA-256 keyed JSON files)
data/processed/             # normalized benchmark CSVs
data/splits/                # question ID splits per benchmark/track
```

## Experiment methods

| Key | Class | Description |
|-----|-------|-------------|
| `baseline` | `DirectMCQRunner` | Single prompt, parse first letter |
| `two_prompt` | `TwoStageRunner` | Stage 1: free-text answer; Stage 2: match to A/B/C/D |
| `cyclic` | `PermutationRunner` | 4 cyclic rotations of options; majority vote |
| `two_prompt_cyclic` | `TwoStagePermutationRunner` | Free-text stage 1 + cyclic matching stage 2; majority vote |
| `pride` | `PriDeRunner` | First-token logprobs on Together AI; estimates positional prior via Eq.(7); Eq.(8) debiasing at inference |

## Models and providers

| Model | Provider | Notes |
|-------|----------|-------|
| `gpt-4.1-mini` | OpenAI | |
| `gemini-2.5-flash` | Google Gemini | High API failure rate (~40% on ARC) |
| `llama-3.1-8b-instant` | Groq | |
| `Qwen/Qwen2.5-7B-Instruct-Turbo` | Together AI | Only model that supports PriDe (needs logprobs) |

## PriDe implementation notes

PriDe requires first-token logprobs, which only Together AI provides in this setup. Key implementation details:

- Together AI returns logprobs in a **non-standard format**: parallel arrays (`tokens`, `token_logprobs`, `top_logprobs` as list of dicts) rather than the OpenAI-standard `content` list. `together_client.py` handles both formats.
- An assistant prefill `"The answer is "` is injected when `request_logprobs=True` to guarantee the first generated token is a letter. This is a practical deviation from the paper (which has full logit access to open-source models).
- `top_logprobs=20` is requested. Letters missing from top 20 get `_LOGPROB_FLOOR = -30.0`.
- Calibration sidecars are saved to `runs/<run_id>/pride_calibration__<model>__<benchmark>.json`.
- If stale cache entries exist with empty logprobs (from a broken run), find and delete them with: `grep -rl '"logprobs": \[\]' .cache/responses/ | xargs rm`

## Evaluation outputs (per benchmark)

All written to `reports/<run_id>/<benchmark>/`:
- `accuracy.csv` — end-to-end and conditional accuracy with 95% bootstrap CIs
- `positional_bias.csv` — mean absolute deviation from ground-truth answer distribution
- `overlap.csv` — question-level overlap vs baseline (both correct, net effect, etc.)
- `choice_shifts.csv` — how often each method changes the answer vs baseline
- `subject_accuracy.csv` — per-subject breakdown
- `two_stage_metrics.csv` — free-text availability and latency for two_prompt methods

## Key gotchas

- `MODEL_ORDER` in both `evaluate_run.py` and `aggregate_results.py` must use the exact model name string as it appears in run CSVs (e.g. `Qwen/Qwen2.5-7B-Instruct-Turbo`, not `Qwen/Qwen2.5-7B-Instruct`). A mismatch silently drops all rows for that model.
- `evaluate_run.py --benchmark arc` is aliased to `arc_challenge` via `_BENCHMARK_ALIASES`.
- The cache key includes `request_logprobs`, so logprob and non-logprob requests for the same prompt are cached separately.
- PriDe jobs cannot be re-run from checkpoint if the calibration sidecar exists but the run CSV was deleted — delete both together.
- `src/twoprompt/config/models.py` is for constants and key loading only. All run configuration (models, jobs, rate limits) is in `config/default.yaml`.
