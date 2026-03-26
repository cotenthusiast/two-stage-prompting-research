"""Download MMLU, normalize it, and generate experiment splits."""

from twoprompt.benchmarks.split import build_all_splits
from twoprompt.config.experiment import ROBUSTNESS_TRACK_NAME, REVIEW_TRACK_NAME
from twoprompt.config.paths import (
    NORMALIZED_QUESTIONS_FILENAME,
    NORMALIZED_QUESTIONS_PATH,
    PROCESSED_DIR,
    RAW_QUESTIONS_PATH,
    SPLITS_DIR,
)
from twoprompt.io.readers import read_normalized_questions
from twoprompt.io.writers import (
    write_group_splits,
    write_normalized_questions,
    write_raw_questions,
)


def main():
    """Download MMLU, produce a normalized CSV, and generate experiment splits."""
    if RAW_QUESTIONS_PATH.exists():
        print(f"[skip] Raw questions already exist at {RAW_QUESTIONS_PATH}")
    else:
        print("[1/4] Downloading MMLU test split from HuggingFace...")
        write_raw_questions(RAW_QUESTIONS_PATH)
        print(f"[done] Saved raw questions to {RAW_QUESTIONS_PATH}")

    if NORMALIZED_QUESTIONS_PATH.exists():
        print(f"[skip] Normalized questions already exist at {NORMALIZED_QUESTIONS_PATH}")
    else:
        print("[2/4] Normalizing raw questions...")
        write_normalized_questions(RAW_QUESTIONS_PATH, NORMALIZED_QUESTIONS_PATH)
        print(f"[done] Saved normalized questions to {NORMALIZED_QUESTIONS_PATH}")

    print("[3/4] Reading normalized questions and deduplicating...")
    df = read_normalized_questions(NORMALIZED_QUESTIONS_FILENAME, PROCESSED_DIR)
    original_count = len(df)
    df = df.drop_duplicates(subset="question_id")
    deduped_count = len(df)
    print(f"[done] {original_count} rows -> {deduped_count} unique questions ({original_count - deduped_count} duplicates removed)")

    print("[4/4] Building and writing experiment splits...")
    all_splits = build_all_splits(df)

    write_group_splits(
        {ROBUSTNESS_TRACK_NAME: all_splits[ROBUSTNESS_TRACK_NAME]},
        SPLITS_DIR,
        "benchmark",
    )
    write_group_splits(
        {REVIEW_TRACK_NAME: all_splits[REVIEW_TRACK_NAME]},
        SPLITS_DIR,
        "faithfulness",
    )
    write_group_splits(
        {REVIEW_TRACK_NAME: all_splits[REVIEW_TRACK_NAME]},
        SPLITS_DIR,
        "stronger_model",
    )

    robustness_count = len(all_splits[ROBUSTNESS_TRACK_NAME]["ids"])
    review_count = len(all_splits[REVIEW_TRACK_NAME]["ids"])
    print(f"[done] Robustness split: {robustness_count} questions")
    print(f"[done] Review split: {review_count} questions (written to faithfulness + stronger_model)")
    print("[complete] Data preparation finished.")


if __name__ == "__main__":
    main()