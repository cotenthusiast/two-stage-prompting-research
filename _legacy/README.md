# Two-Stage Prompting for MCQ Bias Mitigation

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
| Baseline (standard MCQ) | 58% | 1% |
| Two-stage prompting | **76%** | 0% |

The two-stage approach yielded an **18 percentage point improvement** in overall accuracy, with gains observed across most subjects. The largest improvements were in college mathematics (+50pp) and high school physics (+50pp).

### Bias Reduction

Token distribution analysis confirmed that the improvement was driven at least in part by bias reduction. The baseline selected token A 60 times against a ground truth of 32 — nearly double. The two-stage approach reduced this to 48, cutting the excess bias by approximately 50%.

### Per-Question Agreement

On a per-question basis, the two-stage approach corrected 23 questions the baseline got wrong while only regressing on 5 questions. The remaining 72 questions were answered the same way by both methods.

![Comparison chart](results/plots/comparison.png)

## Project Structure

```
two-prompt-research/
├── data/
│   └── questions.csv                       # 100-question MMLU sample
├── scripts/
│   ├── data/
│   │   └── sample_data.py                  # Download & sample MMLU questions
│   ├── experiments/
│   │   ├── baseline.py                     # Standard MCQ prompting
│   │   └── two_prompt.py                   # Two-stage prompting
│   ├── analysis/
│   │   ├── eval.py                         # Per-method evaluation + bar chart
│   │   ├── comparison_chart.py             # Side-by-side accuracy comparison
│   │   ├── token_distribution.py           # Token bias distribution analysis
│   │   └── answer_agreement_analysis.py    # Per-question regression analysis
│   └── utils/
│       └── test_env.py                     # Environment / API check
├── results/
│   ├── baseline/
│   │   └── baseline_results.csv
│   ├── two_stage/
│   │   └── two_stage_results.csv
│   └── plots/
│       ├── baseline_accuracy.png
│       ├── two_stage_accuracy.png
│       ├── comparison.png
│       ├── token_distribution.png
│       ├── agreement_overall.png
│       └── agreement_per_subject.png
├── .env                                    # GEMINI_API_KEY (not committed)
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/cotenthusiast/two-prompt-research
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

All scripts should be run from the project root directory.

**Sample MMLU questions:**
```bash
python scripts/data/sample_data.py
```

**Run baseline experiment:**
```bash
python scripts/experiments/baseline.py
```

**Run two-stage experiment:**
```bash
python scripts/experiments/two_prompt.py
```

**Evaluate results:**
```bash
python scripts/analysis/eval.py results/baseline/baseline_results.csv
python scripts/analysis/eval.py results/two_stage/two_stage_results.csv
```

**Generate comparison chart:**
```bash
python scripts/analysis/comparison_chart.py
```

**Generate token distribution chart:**
```bash
python scripts/analysis/token_distribution.py
```

**Generate answer agreement charts:**
```bash
python scripts/analysis/answer_agreement_analysis.py
```

## Dependencies

- `google-genai`
- `pandas`
- `matplotlib`
- `numpy`
- `python-dotenv`

## Motivation

Based on the PriDe paper's analysis of token and position bias in MCQ evaluation. The two-stage approach is proposed as a lightweight, prompt-only mitigation that requires no model fine-tuning or access to token logits. Unlike chain-of-thought prompting which aims to improve reasoning quality, this method specifically targets the influence of option labels on the selection process.

## Limitations

- 100-question sample; results may not generalise to the full MMLU benchmark
- Single model (`gemini-2.5-flash`); behaviour may differ across model families
- Per-subject sample sizes are small (~10 questions each), so subject-level results should be interpreted cautiously
