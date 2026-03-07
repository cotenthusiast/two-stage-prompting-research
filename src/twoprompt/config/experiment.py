# src/twoprompt/config/experiment.py

# MMLU ------------------------------------------------
MCQ_ANSWER_MAP = "ABCDEFG"

SUBJECTS = [
    "high_school_physics",
    "college_mathematics",
    "anatomy",
    "college_chemistry",
    "computer_security",
    "medical_genetics",
    "college_biology",
    "clinical_knowledge",
    "high_school_psychology",
    "econometrics",
    "sociology",
    "philosophy",
    "high_school_world_history",
    "jurisprudence",
    "professional_law",
    "professional_medicine",
    "professional_accounting",
    "moral_scenarios",
    "nutrition",
    "global_facts",
]

NO_OF_SUBJECTS = len(SUBJECTS)

# Track A: robustness / accuracy ---------------------
ROBUSTNESS_TRACK_NAME = "robustness"
BENCHMARK_TOTAL_QUESTIONS = 1000
BENCHMARK_SUBJECTS = SUBJECTS

BASELINE_METHOD = "baseline"
TWOPROMPT_METHOD = "two_prompt"
PRIDE_METHOD = "pride"
TWOPROMPT_PRIDE_METHOD = "two_prompt_pride"

ROBUSTNESS_METHODS = [
    BASELINE_METHOD,
    TWOPROMPT_METHOD,
    PRIDE_METHOD,
    TWOPROMPT_PRIDE_METHOD,
]

# Track B: faithfulness ------------------------------
FAITHFULNESS_TRACK_NAME = "faithfulness"
FAITHFULNESS_SUBJECTS = SUBJECTS
FAITHFULNESS_QUESTIONS_PER_SUBJECT = 3
FAITHFULNESS_TOTAL_QUESTIONS = len(FAITHFULNESS_SUBJECTS) * FAITHFULNESS_QUESTIONS_PER_SUBJECT

ANSWER_MATCHING_METHOD = "answer_matching"

FAITHFULNESS_METHODS = [
    BASELINE_METHOD,
    TWOPROMPT_METHOD,
    ANSWER_MATCHING_METHOD,
]

# Human review labels --------------------------------
HUMAN_LABEL_CORRECT = "correct"
HUMAN_LABEL_INCORRECT = "incorrect"
HUMAN_LABEL_AMBIGUOUS = "ambiguous"

HUMAN_LABELS = [
    HUMAN_LABEL_CORRECT,
    HUMAN_LABEL_INCORRECT,
    HUMAN_LABEL_AMBIGUOUS,
]