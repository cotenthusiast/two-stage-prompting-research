import pytest

from twoprompt.parsing.parser import (
    extract_choice_letter,
    extract_choice_text_match,
    normalize_output_text,
    parse_model_answer,
)
from twoprompt.parsing.types import (
    PARSE_AMBIGUOUS,
    PARSE_INVALID,
    PARSE_MISSING,
    PARSE_OK,
    ParseResult
)


# ---------------------------------------------------------------------
# normalize_output_text
# ---------------------------------------------------------------------


def test_normalize_output_text_returns_empty_string_for_none() -> None:
    """Returns an empty string when raw_text is None."""
    assert normalize_output_text(None) == ""


def test_normalize_output_text_returns_empty_string_for_blank_input() -> None:
    """Returns an empty string when raw_text contains only whitespace."""
    assert normalize_output_text(" ") == ""


def test_normalize_output_text_strips_leading_and_trailing_whitespace() -> None:
    """Removes surrounding whitespace from the raw text."""
    assert normalize_output_text(" a b c ") == "a b c"


def test_normalize_output_text_collapses_repeated_internal_spaces() -> None:
    """Collapses repeated spaces into a single space."""
    assert normalize_output_text("a  b  c") == "a b c"


def test_normalize_output_text_collapses_tabs_and_newlines_into_single_spaces() -> None:
    """Normalizes tabs and newlines into single internal spaces."""
    assert normalize_output_text("a\n\t b\t c") == "a b c"


def test_normalize_output_text_preserves_meaningful_punctuation() -> None:
    """Preserves punctuation needed by downstream parsing."""
    assert normalize_output_text("(a)") == "(a)"


# ---------------------------------------------------------------------
# extract_choice_letter
# ---------------------------------------------------------------------


def test_extract_choice_letter_returns_missing_for_empty_normalized_text() -> None:
    """Returns PARSE_MISSING when normalized_text is empty."""
    assert extract_choice_letter("").status == PARSE_MISSING


def test_extract_choice_letter_parses_single_uppercase_letter() -> None:
    """Parses a single uppercase answer token such as 'B'."""
    result = extract_choice_letter("B")
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_letter_parses_single_lowercase_letter() -> None:
    """Parses a single lowercase answer token and canonicalizes it."""
    result = extract_choice_letter("b")
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_letter_parses_letter_wrapped_in_punctuation() -> None:
    """Parses a valid answer token surrounded by punctuation."""
    result = extract_choice_letter("B:")
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_letter_parses_answer_is_pattern() -> None:
    """Parses a cue-based pattern such as 'answer is C'."""
    result = extract_choice_letter("answer is B")
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_letter_parses_answer_direct_pattern() -> None:
    """Parses a cue-based pattern such as 'answer C'."""
    result = extract_choice_letter("answer B")
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_letter_parses_final_answer_pattern() -> None:
    """Parses a strong cue pattern such as 'final answer A'."""
    result = extract_choice_letter("final answer B")
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_letter_parses_final_answer_is_pattern() -> None:
    """Parses a strong cue pattern such as 'final answer is D'."""
    result = extract_choice_letter("final answer is B")
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_letter_returns_ambiguous_for_conflicting_strong_candidates() -> None:
    """Returns PARSE_AMBIGUOUS when multiple strong answer candidates conflict."""
    assert extract_choice_letter(
        "The final answer is B or maybe the answer is C"
    ).status == PARSE_AMBIGUOUS


def test_extract_choice_letter_returns_ambiguous_for_conflicting_weak_candidates() -> None:
    """Returns PARSE_AMBIGUOUS when multiple standalone valid choice tokens conflict."""
    assert extract_choice_letter("A or C or B").status == PARSE_AMBIGUOUS


def test_extract_choice_letter_prefers_strong_candidate_over_weak_mentions() -> None:
    """Uses the strong cue candidate even if weaker standalone mentions also appear."""
    result = extract_choice_letter("The final answer is B, not A or C")
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_letter_returns_missing_when_no_direct_letter_exists() -> None:
    """Returns PARSE_MISSING when no usable direct answer letter is found."""
    assert extract_choice_letter("Im not sure").status == PARSE_MISSING


# ---------------------------------------------------------------------
# extract_choice_text_match
# ---------------------------------------------------------------------


def test_extract_choice_text_match_returns_ok_for_exact_option_text_match(
    sample_options: dict[str, str],
) -> None:
    """Parses successfully when normalized_text exactly matches one option text."""
    result = extract_choice_text_match(sample_options["B"], sample_options)
    assert result.final_choice == "B"
    assert result.status == PARSE_OK


def test_extract_choice_text_match_returns_ok_when_option_text_appears_in_longer_response(
    sample_options: dict[str, str],
) -> None:
    """Parses successfully when one option text appears inside a longer response."""
    normalized_text = f"The answer is {sample_options['C']}"
    result = extract_choice_text_match(normalized_text, sample_options)
    assert result.final_choice == "C"
    assert result.status == PARSE_OK


def test_extract_choice_text_match_returns_ambiguous_when_multiple_option_texts_match(
    sample_options: dict[str, str],
) -> None:
    """Returns PARSE_AMBIGUOUS when more than one option text appears valid."""
    normalized_text = f"It could be {sample_options['A']} or {sample_options['D']}"
    result = extract_choice_text_match(normalized_text, sample_options)
    assert result.final_choice is None
    assert result.status == PARSE_AMBIGUOUS


def test_extract_choice_text_match_returns_missing_when_no_option_text_matches(
    sample_options: dict[str, str],
) -> None:
    """Returns PARSE_MISSING when no option text can be matched."""
    result = extract_choice_text_match("I am not sure what the answer is.", sample_options)
    assert result.final_choice is None
    assert result.status == PARSE_MISSING


# ---------------------------------------------------------------------
# parse_model_answer
# ---------------------------------------------------------------------


def test_parse_model_answer_parses_single_letter_answer(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Parses a clean single-letter answer such as 'B'."""
    case = sample_case_map["clean_b"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.final_choice == case["expected_choice"]
    assert result.status == case["expected_status"]


def test_parse_model_answer_parses_lowercase_letter(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Parses a lowercase answer letter and canonicalizes it."""
    case = sample_case_map["lowercase_with_spaces"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.final_choice == case["expected_choice"]
    assert result.status == case["expected_status"]


def test_parse_model_answer_ignores_punctuation_around_letter(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Parses an answer letter even when punctuation surrounds it."""
    case = sample_case_map["punctuation_around_letter"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.final_choice == case["expected_choice"]
    assert result.status == case["expected_status"]


def test_parse_model_answer_rejects_multiple_conflicting_letters(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Returns ambiguous when conflicting answer letters appear."""
    case = sample_case_map["conflicting_letters"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.final_choice == case["expected_choice"]
    assert result.status == case["expected_status"]


def test_parse_model_answer_rejects_empty_output(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Returns missing when the model output is empty."""
    case = sample_case_map["empty_string"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.final_choice == case["expected_choice"]
    assert result.status == case["expected_status"]


def test_parse_model_answer_fallback_matches_option_text(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Falls back to matching answer text when no letter is present."""
    case = sample_case_map["option_text_instead_of_letter"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.final_choice == case["expected_choice"]
    assert result.status == case["expected_status"]


def test_parse_model_answer_returns_ambiguous_on_multiple_valid_text_matches(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Returns ambiguous when multiple option texts plausibly match."""
    case = sample_case_map["ambiguous_text_match"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.final_choice == case["expected_choice"]
    assert result.status == case["expected_status"]


def test_parse_model_answer_handles_reasoning_plus_final_answer_format(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Parses outputs that include reasoning followed by a final answer."""
    case = sample_case_map["reasoning_final_a"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.final_choice == case["expected_choice"]
    assert result.status == case["expected_status"]


def test_parse_model_answer_preserves_original_raw_text_in_final_result(
    sample_case_map: dict[str, dict[str, object]],
    sample_options: dict[str, str],
) -> None:
    """Preserves the original raw_text in the final ParseResult."""
    case = sample_case_map["lowercase_with_spaces"]
    result = parse_model_answer(str(case["raw_text"]), sample_options)
    assert result.raw_text == case["raw_text"]


def test_parse_model_answer_uses_text_match_only_after_missing_direct_letter(
    sample_options: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uses fallback text matching only when direct letter parsing returns missing."""
    text_match_called = False

    def fake_extract_choice_letter(normalized_text: str) -> ParseResult:
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=None,
            normalized_text=normalized_text,
            reason="No direct answer letter found",
        )

    def fake_extract_choice_text_match(normalized_text: str, options: dict[str, str]) -> ParseResult:
        nonlocal text_match_called
        text_match_called = True
        return ParseResult(
            final_choice="B",
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed from option text match",
        )

    monkeypatch.setattr("twoprompt.parsing.parser.extract_choice_letter", fake_extract_choice_letter)
    monkeypatch.setattr("twoprompt.parsing.parser.extract_choice_text_match", fake_extract_choice_text_match)

    result = parse_model_answer("Type 2 diabetes mellitus", sample_options)

    assert text_match_called is True
    assert result.final_choice == "B"
    assert result.status == PARSE_OK
