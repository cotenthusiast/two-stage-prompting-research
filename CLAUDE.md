# two-prompt-research

Research codebase for measuring positional bias in LLM multiple-choice answering and testing whether prompting/debiasing interventions reduce that bias. Based on Zheng et al. (ICLR 2024), "Large Language Models Are Not Robust Multiple Choice Selectors".

**Current paper framing:**

- Primary: robustness / positional-bias robustness
- Secondary: faithfulness of two-stage decomposition
- Accuracy: tertiary/supporting
- Main empirical claim: two-stage prompting does not reliably mitigate MCQ positional bias and reduces end-to-end accuracy across evaluated models/benchmarks.

**Methods:** Direct MCQ baseline, two-stage prompting, cyclic permutation, two-stage cyclic, PriDe logprob debiasing.

**Benchmarks:** MMLU, ARC-Challenge.

**Models/providers:**

- OpenAI: `gpt-4.1-mini`
- Google Gemini: `gemini-2.5-flash`
- Groq: `llama-3.1-8b-instant`
- Together AI: `Qwen/Qwen2.5-7B-Instruct-Turbo`

PriDe currently only runs on Together AI because it requires first-token logprobs.

---

## How Claude should work in this repository

Claude should behave conservatively. The current priority is paper correctness, reproducibility, and clean extension toward model-generalization experiments.

**Prefer:**

- Small, reviewable diffs
- Preserving existing file structure
- Explaining the intended change before editing
- Adding tests or sanity checks when modifying logic
- Avoiding unnecessary new dependencies and premature abstraction

**Do not:**

- Run expensive API experiments unless explicitly instructed
- Modify `.env` or expose API keys
- Delete cache/checkpoints unless explicitly instructed
- Change benchmark schemas casually or silently rename method/model keys
- Edit `src/twoprompt/config/` for normal run configuration
- Rewrite working modules for style or convert this into a generic framework before the paper is finished
- Prioritize framework refactoring until after the paper is reviewed/submitted

**For important changes, prefer this workflow:**

1. State what the file/change should do.
2. Identify inputs, outputs, and edge cases.
3. Make the smallest change.
4. Explain the change.
5. Run or suggest tests.
6. Avoid unrelated cleanup.

**Rule:** If AI vanished tomorrow, the user should still be able to explain, modify, and continue the project slowly.

---

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env
```

`.env` requires: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`

---

## Running experiments

All run configuration lives in `config/default.yaml` (model list, job matrix, rate limits, temperature/seed, cache settings, prompt version). Do not edit `src/twoprompt/config/` for ordinary experiment configuration.

```bash
# Dry run only, no API calls
python scripts/run_experiment.py --dry-run

# Full run, asks for confirmation
python scripts/run_experiment.py

# Re-run specific run ID using checkpoint skips
python scripts/run_experiment.py --run-id 20260505_200402 --yes

# Force no cache
python scripts/run_experiment.py --no-cache
```

Current important run: `20260505_200402`

---

## Evaluating runs

```bash
python scripts/evaluate_run.py <run_id> --benchmark mmlu
python scripts/evaluate_run.py <run_id> --benchmark arc   # arc aliases arc_challenge
```

Reports written to `reports/<run_id>/<benchmark>/`.

---

## Aggregating paper tables

```bash
python scripts/aggregate_results.py
```

---

## Project structure

```text
config/
  default.yaml

scripts/
  run_experiment.py
  evaluate_run.py
  aggregate_results.py
  prepare_data.py

src/twoprompt/
  clients/
    base.py           openai_client.py   gemini_client.py
    groq_client.py    together_client.py types.py
  runners/
    base.py           direct_mcq.py      permutation.py
    two_stage.py      two_stage_permutation.py
    pride.py          pride_debias.py
  infra/
    cache.py          checkpoint.py
  config/
    experiment.py     models.py          paths.py
  benchmarks/         parsing/           scoring/
  io/                 pipeline/

prompts/v1/
  direct_mcq.txt    free_text.txt    option_matching.txt

runs/   reports/   checkpoints/   .cache/responses/
data/processed/    data/splits/
```

---

## Experiment methods

| Key | Class | Description |
| --- | ----- | ----------- |
| `baseline` | `DirectMCQRunner` | Single prompt, parse first answer letter |
| `two_prompt` | `TwoStageRunner` | Stage 1 free-text answer, Stage 2 option matching |
| `cyclic` | `PermutationRunner` | Four cyclic option rotations with majority vote |
| `two_prompt_cyclic` | `TwoStagePermutationRunner` | Free-text answer plus cyclic option matching |
| `pride` | `PriDeRunner` | Logprob-based positional-prior debiasing |

---

## Evaluation outputs

Per benchmark, written to `reports/<run_id>/<benchmark>/`:

- `accuracy.csv` — end-to-end and conditional accuracy with 95% CIs
- `positional_bias.csv` — MAD from ground-truth answer distribution
- `overlap.csv` — question-level overlap vs baseline
- `choice_shifts.csv` — how often each method changes the answer vs baseline
- `subject_accuracy.csv` — per-subject breakdown
- `two_stage_metrics.csv` — free-text availability and latency

**Interpretation rules:**

- End-to-end accuracy counts unscorable outputs as incorrect; conditional uses only scorable outputs.
- Conditional accuracy alone can hide matching-stage failures.
- MAD lower means less positional bias.
- Do not overclaim statistical significance unless explicitly computed.

---

## Current main empirical findings

- On MMLU and ARC-Challenge, two-stage prompting reduces end-to-end accuracy for every model.
- On MMLU, two-stage prompting increases MAD point estimate for every model.
- On ARC-Challenge, MAD results are mixed; two-stage does not show consistent bias reduction.
- Gemini 2.5 Flash suffers substantial parse failures under two-stage prompting, especially on MMLU.
- GPT-4.1 mini shows two-stage can harm accuracy even without parse failures.
- Cyclic permutation is the strongest model-agnostic intervention overall.
- PriDe gives limited positive results for Qwen/Together but requires logprob access so is not broadly model-agnostic.

The accurate claim is: **two-stage prompting fails to reliably mitigate MCQ positional bias and reduces end-to-end accuracy across all evaluated model/benchmark settings.** Do not rewrite this as "two-stage always increases bias everywhere."

---

## PriDe implementation notes

- Together AI returns logprobs in a non-standard parallel-array format; `together_client.py` handles both formats.
- Assistant prefill `"The answer is "` is injected when `request_logprobs=True` (practical deviation from the paper).
- `top_logprobs=20` is requested; missing letters get `_LOGPROB_FLOOR = -30.0`.
- Calibration sidecars saved to `runs/<run_id>/pride_calibration__<model>__<benchmark>.json`.
- If stale cache entries with empty logprobs exist from broken runs: `grep -rl '"logprobs": \[\]' .cache/responses/ | xargs rm`
- Do not delete cache broadly unless explicitly asked.

---

## Gotchas

- **Exact model names matter.** `MODEL_ORDER` in `evaluate_run.py` and `aggregate_results.py` must match exact strings in run CSVs. `Qwen/Qwen2.5-7B-Instruct-Turbo` ≠ `Qwen/Qwen2.5-7B-Instruct` — a mismatch silently drops model rows.
- **ARC alias:** `--benchmark arc` aliases `arc_challenge`.
- **Cache keys include** `request_logprobs`, so logprob and non-logprob requests are cached separately.
- **PriDe reruns:** if the run CSV is deleted, delete the matching calibration sidecar too — they cannot be rerun independently.
- **Prompt snapshots** are versioned under `prompts/<version>/` and may be copied into run folders; preserve for reproducibility.

---

## Coding style

Use simple, explicit Python. **Prefer:** dataclasses/typed dicts, clear function boundaries, readable loops, explicit error handling, deterministic seeds, stable CSV schemas, small helpers, tests for parsing/scoring changes. **Avoid:** hidden global state, broad exception swallowing, changing CSV columns or method/model keys without migration, heavy dependencies, complex framework machinery.

---

## Testing and sanity checks

```bash
pytest
python scripts/run_experiment.py --dry-run
python scripts/evaluate_run.py 20260505_200402 --benchmark mmlu
python scripts/evaluate_run.py 20260505_200402 --benchmark arc
python scripts/aggregate_results.py
```

When modifying **parsing/scoring**: test A/B/C/D extraction, unscorable outputs, lowercase/verbose outputs, outputs with explanations, deterministic scoring.

When modifying **PriDe**: test logprob extraction, missing-letter floor, calibration state load/save, Eq.(7)/Eq.(8) separately.

---

## Future direction: model generalization

The next research direction is likely: do MCQ bias-mitigation methods validated in limited model settings preserve their effects across larger open-source models?

Do not implement this inside the current codebase until the design is explicitly chosen. Reversible prep is allowed (literature table, candidate-method table, Kelvin2/SLURM learning, small toy HPC jobs). Do not prematurely restructure this repository into a generic MCQ framework.

---

## Kelvin2 / HPC caution

- Do not run heavy jobs on the login node — use SLURM batch jobs.
- Start with tiny test jobs; log stdout/stderr clearly.
- Do not assume local laptop paths work on HPC.
- Keep environment setup documented.
- Mental model: login node = prepare/submit; compute node = run workload; SLURM = scheduler.

---

## Immediate priorities

- Preserve the working paper pipeline; avoid broad refactors.
- Support supervisor-review changes.
- Add/maintain benchmark table for the survey paper separately.
- Scope model-generalization direction before implementation.
- Learn Kelvin2/SLURM safely before running real experiments.
