from twoprompt.config.prompts import PROMPTS

def build_direct_mcq_prompt(
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
) -> str:
    """
    Build the direct multiple-choice baseline prompt.

    Args:
        question: Question stem to present to the model.
        option_a: Text of answer option A.
        option_b: Text of answer option B.
        option_c: Text of answer option C.
        option_d: Text of answer option D.

    Returns:
        Fully formatted prompt string instructing the model to answer
        the multiple-choice question by selecting one option letter.
    """
    return PROMPTS["direct_mcq"].format(
        question=question,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )

def build_free_text_prompt(
    question: str,
) -> str:
    """
    Build the free-text prompt for stage one of the two-stage method.

    Args:
        question: Question stem to present to the model without answer options.

    Returns:
        Fully formatted prompt string instructing the model to provide
        a short direct free-text answer.
    """
    return PROMPTS["free_text"].format(question=question)

def build_option_matching_prompt(
    question: str,
    free_text: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
) -> str:
    """
    Build the option-matching prompt for stage two of the two-stage method.

    Args:
        question: Original question stem.
        free_text: Free-text answer produced in stage one.
        option_a: Text of answer option A.
        option_b: Text of answer option B.
        option_c: Text of answer option C.
        option_d: Text of answer option D.

    Returns:
        Fully formatted prompt string instructing the model to select
        the option letter that best matches the free-text answer in the
        context of the original question.
    """
    return PROMPTS["option_matching"].format(
        question=question,
        free_text=free_text,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
    )