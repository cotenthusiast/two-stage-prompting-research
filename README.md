<img width="2879" height="1799" alt="image" src="https://github.com/user-attachments/assets/df0c29b7-9b79-4d85-8ad5-bed20170d935" /># Two-Stage Prompting for MCQ Bias Mitigation

A pilot experiment testing whether a two-stage prompting approach can reduce selection bias in LLM multiple-choice question evaluation, inspired by the [PriDe paper (ICLR 2024)](https://arxiv.org/abs/2401.12485).

## Report

Full report available on [Hugging Face](https://huggingface.co/spaces/cotenthusiast/mcq-selection-bias-two-stage-prompting)
## Overview

Large language models exhibit systematic selection bias when answering MCQs — they tend to favour certain answer positions (e.g. option A) or tokens regardless of question content. This experiment tests a simple mitigation: decouple reasoning from option selection by splitting the prompt into two stages.

- **Stage 1** — Ask the model to reason through the question without seeing the answer choices
- **Stage 2** — Present the reasoning alongside the options and ask the model to select the best match

## Results

Evaluated on a 100-question sample from the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu) across 10 subjects using `gemini-2.5-flash`.

| Method | Overall Accuracy | Failure Rate |
|---|---|---|
| Baseline (standard MCQ) | 58% | — |
| Two-stage prompting | **76%** | 0% |

The two-stage approach yielded an **18 percentage point improvement** in overall accuracy, with gains observed across most subjects. The largest improvements were in college mathematics (+50pp) and high school physics (+50pp), suggesting that reasoning-heavy subjects benefit most from the decoupled approach.

![Comparison chart](results/comparison.png)

## Project Structure

```
two-prompt-research/
├── data/
│   └── questions.csv          # 100-question MMLU sample
├── results/
│   ├── baseline_results.csv
│   ├── two_stage_results.csv
│   ├── baseline_accuracy.png
│   ├── two_stage_accuracy.png
│   └── comparison.png
├── scripts/
│   ├── baseline.py            # Standard MCQ prompting
│   ├── two_prompt.py          # Two-stage prompting
│   ├── eval.py                # Per-script evaluation + bar chart
│   └── comparison_eval.py     # Side-by-side comparison chart
├── .env                       # GEMINI_API_KEY (not committed)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/yourusername/two-prompt-research
cd two-prompt-research
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

## Usage

**Run baseline experiment:**
```bash
python scripts/baseline.py
```

**Run two-stage experiment:**
```bash
python scripts/two_prompt.py
```

**Evaluate results:**
```bash
python scripts/eval.py results/baseline_results.csv
python scripts/eval.py results/two_stage_results.csv
```

**Generate comparison chart:**
```bash
python -m scripts.comparison_eval
```

## Dependencies

- `google-genai`
- `pandas`
- `matplotlib`
- `numpy`
- `python-dotenv`

## Motivation

Based on the PriDe paper's analysis of token and position bias in MCQ evaluation. The two-stage approach is proposed as a lightweight, prompt-only mitigation that requires no model fine-tuning or access to token logits.

## Limitations

- 100-question sample; results may not generalise to the full MMLU benchmark
- Single model (`gemini-2.5-flash`); behaviour may differ across model families
- Per-subject sample sizes are small (~10 questions each), so subject-level results should be interpreted cautiously
