# Two-Stage Prompting Does Not Mitigate MCQ Positional Bias in LLMs
 
This repository contains the full code, experiment outputs, evaluation reports, and paper draft for a benchmarking study on multiple-choice selection bias in large language models.
 
**Short version:** the hypothesis was wrong. Two-stage prompting — separating free-text answer generation from final option selection — does not reliably mitigate positional bias in MCQ evaluation, and in some settings makes things worse. Cyclic permutation remains the stronger robustness intervention.
 
---
 
## Research Question
 
Can multiple-choice selection bias be reduced at the prompting stage by separating answer generation from option selection?
 
The motivating intuition is straightforward: if option labels distort model behavior, then reasoning should happen *before* labels are introduced. This project tests that idea in a controlled benchmark and finds it insufficient.
 
## Experiment Design
 
Four conditions are compared across 1,000 stratified MMLU questions:
 
| Condition | Description |
|---|---|
| **Baseline** | Direct MCQ answering |
| **Two-Stage** | Free-text answer, then option matching |
| **Cyclic** | Cyclic permutation of answer options |
| **Two-Stage + Cyclic** | Both interventions combined |
 
Models evaluated: **GPT-4.1 mini**, **Gemini 2.5 Flash**, **Llama 3.1 8B** (via Groq).
 
This is a robustness study, not a capability benchmark. The primary metric is end-to-end accuracy (correct / total), which includes failures from unscorable outputs and provider errors. Conditional accuracy (correct / scored) is reported as a secondary metric.
 
## Main Findings
 
- Two-stage prompting does not consistently reduce positional bias. In some cases it shifts the bias distribution rather than flattening it.
- Gemini 2.5 Flash shows a large end-to-end accuracy drop under two-stage evaluation, driven by a high volume of outputs that pass semantic review but fail the strict scoring pipeline.
- Conditional accuracy can remain high even when end-to-end performance collapses. This means unscorable burden is not a footnote — it is part of the practical cost of a method.
- Cyclic permutation remains the strongest robustness baseline in this setup, despite its higher inference cost.
- The pilot study (smaller subset, Gemini 2.5 Flash only) suggested a possible benefit. The full benchmark did not replicate that signal.
 
The conclusion is intentionally narrow: **in this benchmark, separating generation from option selection is not sufficient to reliably mitigate MCQ positional bias.** This is not a claim that two-stage prompting is universally useless.
 
## Relation to Prior Work
 
This project is positioned relative to approaches that require logit access (PriDe, Myrzakhan et al.) or full answer permutation enumeration (Wang et al., ABCD-style methods). The two-stage approach is model-agnostic and works on closed-source APIs without internal access. The tradeoff is that it provides weaker robustness guarantees under strict evaluation.
 
The key distinction from chain-of-thought: CoT targets *reasoning quality*. The two-stage structure specifically targets *bias mitigation* by preventing option labels from influencing the generation step. These are different goals, and conflating them misframes what the method is trying to do.
 
## Repository Structure
 
```
two-prompt-research/
├── data/          benchmark data, normalized files, stratified splits
├── reports/       evaluation reports and paper-ready tables
├── runs/          raw outputs per experiment run
├── scripts/       orchestration, evaluation, aggregation, reporting
├── src/twoprompt/ core package
├── tests/         test suite
└── main.tex       paper draft
```
 
## Setup
 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
 
## Running the Pipeline
 
```bash
# Prepare benchmark data
python -m scripts.prepare_data
 
# Evaluate a completed run
python -m scripts.evaluate_run <RUN_ID>
 
# Aggregate into paper-ready tables
python -m scripts.aggregate_results <RUN_ID>
```
 
Example:
 
```bash
python -m scripts.evaluate_run 20260326_102510
python -m scripts.aggregate_results 20260326_102510
```
 
## Metrics
 
**End-to-end accuracy** (`correct / total`) is the headline metric. It accounts for provider failures and unscorable final outputs.
 
**Conditional accuracy** (`correct / scored`) is supplementary. It measures performance among outputs that completed the full pipeline.
 
Robustness is additionally assessed via mean absolute deviation from the ground-truth answer-position distribution, question-level overlap with the baseline condition, and raw unscorable and failure counts per model and condition.
 
## AI Usage Disclosure
 
The research question, hypothesis, experiment design, evaluation framing, and interpretation of results are my own. The core implementation in `src/` was written primarily by me with minimal AI assistance.
 
AI tools were used more substantially in the test suite (for repetitive fixtures and scaffolding) and in the scripts folder (for evaluation, aggregation, and report formatting utilities). Even there, metric definitions and any logic that directly affects paper claims were reviewed and corrected manually.
 
AI was also used for wording and revision support during paper drafting. The ideas, design decisions, and conclusions are my own.
 
## Why a Negative Result
 
Negative results in evaluation methodology are undersupplied. The intuition behind two-stage prompting is reasonable enough that it is worth documenting why it fails under a stricter robustness evaluation — particularly the unscorable burden problem, which is easy to miss if conditional accuracy is the only reported metric.
 
---
 
**Author:** Karl Hanna