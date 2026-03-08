import pandas as pd
import twoprompt.benchmarks.split as split

def test_build_robustness_split_returns_expected_number_of_ids(
    split_test_dataframe,
    split_test_subjects,
) -> None:
    per_subject = 2

    split_ids = split.build_robustness_split(
        split_test_dataframe,
        subjects=split_test_subjects,
        per_subject=per_subject,
    )

    assert len(split_ids) == len(split_test_subjects) * per_subject

def test_build_robustness_split_raises_when_subject_has_insufficient_questions(
    insufficient_split_dataframe,
) -> None: ...


def test_build_review_split_returns_expected_number_of_ids(
    split_test_dataframe,
    split_test_subjects,
) -> None: ...


def test_validate_split_ids_raises_for_wrong_size(
    split_test_dataframe,
    split_test_valid_ids,
) -> None: ...


def test_validate_split_ids_raises_for_duplicate_ids(
    split_test_dataframe,
    split_test_duplicate_ids,
) -> None: ...


def test_validate_split_ids_raises_for_unknown_ids(
    split_test_dataframe,
    split_test_unknown_ids,
) -> None: ...


def test_assert_disjoint_raises_for_overlapping_splits(
    overlapping_split_map,
) -> None: ...


def test_build_split_metadata_returns_expected_fields(
    split_test_dataframe,
    split_test_valid_ids,
    split_test_subjects,
    split_test_exclude_ids,
) -> None: ...


def test_build_all_splits_returns_both_split_artifacts(
    full_default_split_dataframe,
) -> None: ...