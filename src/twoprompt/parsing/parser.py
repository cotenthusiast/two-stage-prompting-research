from __future__ import annotations

from collections.abc import Collection, Mapping

from twoprompt.parsing.types import (
    PARSE_AMBIGUOUS,
    PARSE_INVALID,
    PARSE_MISSING,
    PARSE_OK,
    ParseResult,
)


DEFAULT_VALID_CHOICES = ("A", "B", "C", "D")


def normalize_output_text(raw_text: str | None) -> str:
    """
    Normalize raw model output before parsing.

    Intended later behavior:
    - Handle None safely.
    - Strip leading and trailing whitespace.
    - Collapse repeated internal whitespace.
    - Preserve enough content for both letter extraction and text matching.

    Args:
        raw_text: Raw model output text from a provider response.

    Returns:
        Normalized text string suitable for downstream parsing.
    """
    if raw_text is None:
        return ""
    raw_text = raw_text.strip()
    if raw_text == "":
        return ""
    words = raw_text.split()
    normalized_string = " ".join(words)
    return normalized_string



def extract_choice_letter(
    normalized_text: str,
    valid_choices: Collection[str] = DEFAULT_VALID_CHOICES,
) -> ParseResult:
    """
    Attempt to extract a direct answer letter from normalized output.

    Intended later behavior:
    - Detect isolated answer letters such as A/B/C/D.
    - Handle lowercase output by canonicalizing to uppercase.
    - Prefer explicit final-answer style patterns when present.
    - Return ambiguous if multiple conflicting letters are found.
    - Return missing when no usable letter is present.
    - Never raise because of malformed output.

    Args:
        normalized_text: Pre-normalized model output text.
        valid_choices: Allowed answer letters.

    Returns:
        ParseResult describing the letter-extraction attempt.
    """
    if normalized_text == "":
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=None,
            normalized_text=normalized_text,
            reason="No output to parse"
        )
    words = normalized_text.split()
    for i in range (len(words)):
        words[i] = words[i].strip('()[]{}<>".,:;!?'+"'")
    if len(words) == 1 and words[0].upper() in valid_choices: # high confidenceS
        return ParseResult(
            final_choice=words[0].upper(),
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed"
        )
    cue_words = {"answer", "choice", "option"}
    strong_candidates = []
    weak_candidates = []
    for i in range (len(words)):
        if i + 3 < len(words) and words[i].lower() == "final" and words[i+1].lower() == "answer" and words[i+2].lower() == "is" and words[i+3].upper() in valid_choices:
            strong_candidates.append(words[i + 3].upper())
        if (i + 2 < len(words) and words[i].lower() in cue_words and words[i+1].lower() == "is" and words[i+2].upper() in valid_choices) or (i + 2 < len(words) and words[i].lower() == "final" and words[i+1].lower() == "answer" and words[i+2].upper() in valid_choices):
            strong_candidates.append(words[i + 2].upper())
        if i + 1 < len(words) and words[i].lower() in cue_words and words[i+1].upper() in valid_choices:
            strong_candidates.append(words[i + 1].upper())
        if words[i] in valid_choices:
            weak_candidates.append(words[i])

    unique_strong_candidates = set(strong_candidates)
    unique_weak_candidates = set(weak_candidates)

    if len(unique_strong_candidates) == 1:
        return ParseResult(
            final_choice=next(iter(unique_strong_candidates)),
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed from strong cue pattern"
        )
    elif len(unique_strong_candidates) > 1:
        return ParseResult(
            final_choice=None,
            status=PARSE_AMBIGUOUS,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Multiple conflicting strong answer candidates found"
        )
    else:
        if len(unique_weak_candidates) == 1:
            return ParseResult(
                final_choice=next(iter(unique_weak_candidates)),
                status=PARSE_OK,
                raw_text=None,
                normalized_text=normalized_text,
                reason="Answer successfully parsed from standalone candidate"
            )
        elif len(unique_weak_candidates) > 1:
            return ParseResult(
                final_choice=None,
                status=PARSE_AMBIGUOUS,
                raw_text=None,
                normalized_text=normalized_text,
                reason="Multiple conflicting standalone answer candidates found"
            )
        else:
            return ParseResult(
                final_choice=None,
                status=PARSE_MISSING,
                raw_text=None,
                normalized_text=normalized_text,
                reason="No direct answer letter found"
            )


def extract_choice_text_match(
    normalized_text: str,
    options: Mapping[str, str],
) -> ParseResult:
    """
    Attempt fallback matching against the option texts themselves.

    Intended later behavior:
    - Compare normalized output against normalized option text.
    - Support outputs where the model gives the answer text instead of a letter.
    - Return ambiguous if more than one option text appears valid.
    - Return missing when no option text can be matched.
    - Never raise because of malformed output.

    Args:
        normalized_text: Pre-normalized model output text.
        options: Mapping from answer letter to answer text.

    Returns:
        ParseResult describing the text-matching attempt.
    """
    candidates = []
    for letter, option_text in options.items():
        normalized_option = normalize_output_text(option_text)
        if normalized_option.lower() == normalized_text.lower():
            candidates.append(letter)
        elif normalized_option.lower() in normalized_text.lower():
            candidates.append(letter)

    candidates = set(candidates)

    if len(candidates) == 1:
        return ParseResult(
            final_choice=next(iter(candidates)),
            status=PARSE_OK,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Answer successfully parsed from option text match"
        )
    elif len(candidates) > 1:
        return ParseResult(
            final_choice=None,
            status=PARSE_AMBIGUOUS,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Multiple conflicting option text matches found"
        )
    else:
        return ParseResult(
            final_choice=None,
            status=PARSE_MISSING,
            raw_text=None,
            normalized_text=normalized_text,
            reason="Text present but no option text match found"
        )


def parse_model_answer(
    raw_text: str | None,
    options: Mapping[str, str],
) -> ParseResult:
    """
    Parse a model's MCQ answer into a structured result.

    Intended later flow:
    1. Normalize raw text.
    2. Try direct letter extraction first.
    3. If that fails cleanly, try fallback option-text matching.
    4. Return a structured ParseResult.
    5. Never crash on junk, empty, or unexpected output.

    Args:
        raw_text: Raw text returned by the model.
        options: Mapping from answer letter to answer text.

    Returns:
        Final ParseResult for downstream scoring.
    """
    normalized_text = normalize_output_text(raw_text)
    temp_result = extract_choice_letter(normalized_text)
    if temp_result.status == PARSE_MISSING:
        temp_result = extract_choice_text_match(normalized_text, options)
    return ParseResult(
        final_choice=temp_result.final_choice,
        status=temp_result.status,
        raw_text=raw_text,
        normalized_text=temp_result.normalized_text,
        reason=temp_result.reason
    )