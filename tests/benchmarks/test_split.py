import pytest

import twoprompt.benchmarks.split as split
from twoprompt.benchmarks.split import (
    DuplicateSplitIdsError,
    InsufficientQuestionsError,
    SplitSizeMismatchError,
)
from twoprompt.config.experiment import ROBUSTNESS_SPLIT_SEED


def _assert_split_ids_match_expected_subject_distribution(
    dataframe,
    split_ids,
    expected_subjects,
    per_subject,
) -> None:
    selected_rows = dataframe[dataframe["question_id"].isin(split_ids)]
    subject_counts = selected_rows["subject"].value_counts().to_dict()

    assert len(split_ids) == len(expected_subjects) * per_subject
    assert len(split_ids) == len(set(split_ids))
    assert set(subject_counts.keys()) == set(expected_subjects)
    assert all(count == per_subject for count in subject_counts.values())


def test_build_robustness_split_returns_balanced_unique_ids_for_requested_subjects(
    split_test_dataframe,
    split_test_subjects,
) -> None:
    per_subject = 2

    split_ids = split.build_robustness_split(
        split_test_dataframe,
        subjects=split_test_subjects,
        per_subject=per_subject,
    )

    _assert_split_ids_match_expected_subject_distribution(
        dataframe=split_test_dataframe,
        split_ids=split_ids,
        expected_subjects=split_test_subjects,
        per_subject=per_subject,
    )


def test_build_robustness_split_raises_when_subject_has_insufficient_questions(
    insufficient_split_dataframe,
) -> None:
    per_subject = 2
    subjects = ["computer_security", "high_school_physics"]

    with pytest.raises(InsufficientQuestionsError):
        split.build_robustness_split(
            insufficient_split_dataframe,
            subjects=subjects,
            per_subject=per_subject,
        )


def test_build_review_split_returns_balanced_unique_ids_for_requested_subjects(
    split_test_dataframe,
    split_test_subjects,
) -> None:
    per_subject = 2

    review_ids = split.build_review_split(
        split_test_dataframe,
        subjects=split_test_subjects,
        per_subject=per_subject,
    )

    _assert_split_ids_match_expected_subject_distribution(
        dataframe=split_test_dataframe,
        split_ids=review_ids,
        expected_subjects=split_test_subjects,
        per_subject=per_subject,
    )


def test_build_review_split_raises_when_subject_has_insufficient_questions(
    insufficient_split_dataframe,
) -> None:
    with pytest.raises(InsufficientQuestionsError):
        split.build_review_split(insufficient_split_dataframe)


def test_validate_split_ids_raises_for_wrong_size(
    split_test_dataframe,
    split_test_valid_ids,
) -> None:
    candidate_ids = split_test_valid_ids.copy()
    candidate_ids.pop()

    with pytest.raises(SplitSizeMismatchError):
        split.validate_split_ids(split_test_dataframe, candidate_ids, 4)


def test_validate_split_ids_raises_for_duplicate_ids(
    split_test_dataframe,
    split_test_duplicate_ids,
) -> None:
    with pytest.raises(DuplicateSplitIdsError):
        split.validate_split_ids(split_test_dataframe, split_test_duplicate_ids, 3)


def test_validate_split_ids_raises_for_unknown_ids(
    split_test_dataframe,
    split_test_unknown_ids,
) -> None:
    with pytest.raises(split.UnknownSplitIdsError):
        split.validate_split_ids(split_test_dataframe, split_test_unknown_ids, 2)


def test_validate_split_ids_accepts_valid_ids(
    split_test_dataframe,
    split_test_valid_ids,
) -> None:
    split.validate_split_ids(
        split_test_dataframe,
        split_test_valid_ids,
        expected_size=4,
    )


def test_assert_disjoint_raises_for_overlapping_splits(
    overlapping_split_map,
) -> None:
    with pytest.raises(split.OverlappingSplitIdsError):
        split.assert_disjoint(overlapping_split_map)


def test_build_split_metadata_returns_expected_fields(
    split_test_dataframe,
    split_test_valid_ids,
    split_test_subjects,
    split_test_exclude_ids,
) -> None:
    actual_metadata = split.build_split_metadata(
        split_test_dataframe,
        "robustness",
        split_test_valid_ids,
        split_test_subjects,
        per_subject=2,
        seed=ROBUSTNESS_SPLIT_SEED,
        strategy="robustness",
        exclude_ids=split_test_exclude_ids,
    )

    expected_metadata = {
        "split_name": "robustness",
        "split_ids": [
            "computer_security__01",
            "computer_security__02",
            "high_school_physics__01",
            "anatomy__01",
        ],
        "subjects": [
            "computer_security",
            "high_school_physics",
            "anatomy",
        ],
        "per_subject": 2,
        "seed": ROBUSTNESS_SPLIT_SEED,
        "strategy": "robustness",
        "actual_size": 4,
        "actual_subject_counts": {
            "computer_security": 2,
            "high_school_physics": 1,
            "anatomy": 1,
        },
        "eligible_pool_size": 13,
        "excluded_id_count": 2,
    }

    assert expected_metadata == actual_metadata


def test_build_all_splits_returns_both_split_artifacts(
    full_default_split_dataframe,
) -> None:
    artifacts = split.build_all_splits(full_default_split_dataframe)

    assert set(artifacts.keys()) == {"robustness", "review"}

    robustness_artifact = artifacts["robustness"]
    review_artifact = artifacts["review"]

    required_keys = {
        "split_name",
        "split_ids",
        "subjects",
        "per_subject",
        "seed",
        "strategy",
        "actual_size",
        "actual_subject_counts",
        "eligible_pool_size",
        "excluded_id_count",
    }

    assert required_keys <= set(robustness_artifact.keys())
    assert required_keys <= set(review_artifact.keys())

    robustness_ids = robustness_artifact["split_ids"]
    review_ids = review_artifact["split_ids"]

    robustness_expected_size = (
        len(robustness_artifact["subjects"]) * robustness_artifact["per_subject"]
    )
    review_expected_size = (
        len(review_artifact["subjects"]) * review_artifact["per_subject"]
    )

    split.validate_split_ids(
        full_default_split_dataframe,
        robustness_ids,
        robustness_expected_size,
    )
    split.validate_split_ids(
        full_default_split_dataframe,
        review_ids,
        review_expected_size,
    )

    split.assert_disjoint(
        {
            "robustness": robustness_ids,
            "review": review_ids,
        }
    )

    assert robustness_artifact["split_name"] == "robustness"
    assert review_artifact["split_name"] == "review"

    assert robustness_artifact["actual_size"] == len(robustness_ids)
    assert review_artifact["actual_size"] == len(review_ids)


def test_build_robustness_split_excludes_excluded_ids(
    split_test_dataframe,
    split_test_subjects,
    split_test_exclude_ids,
) -> None:
    actual_ids = split.build_robustness_split(
        split_test_dataframe,
        split_test_subjects,
        per_subject=2,
        exclude_ids=split_test_exclude_ids,
    )
    intersection = set(actual_ids).intersection(split_test_exclude_ids)

    assert len(intersection) == 0


def test_build_review_split_excludes_excluded_ids(
    split_test_dataframe,
    split_test_subjects,
    split_test_exclude_ids,
) -> None:
    actual_ids = split.build_review_split(
        split_test_dataframe,
        split_test_subjects,
        per_subject=2,
        exclude_ids=split_test_exclude_ids,
    )
    intersection = set(actual_ids).intersection(split_test_exclude_ids)

    assert len(intersection) == 0


def test_build_robustness_split_is_deterministic_for_same_seed(
    split_test_dataframe,
    split_test_subjects,
) -> None:
    run1 = split.build_robustness_split(
        split_test_dataframe,
        split_test_subjects,
        per_subject=2,
        seed=1,
    )
    run2 = split.build_robustness_split(
        split_test_dataframe,
        split_test_subjects,
        per_subject=2,
        seed=1,
    )

    assert run1 == run2


def test_build_review_split_is_deterministic_for_same_seed(
    split_test_dataframe,
    split_test_subjects,
) -> None:
    run1 = split.build_review_split(
        split_test_dataframe,
        split_test_subjects,
        per_subject=2,
        seed=1,
    )
    run2 = split.build_review_split(
        split_test_dataframe,
        split_test_subjects,
        per_subject=2,
        seed=1,
    )

    assert run1 == run2