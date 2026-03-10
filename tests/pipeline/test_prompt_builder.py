from twoprompt.pipeline.prompt_builder import build_free_text_prompt, build_direct_mcq_prompt, build_option_matching_prompt

def test_build_direct_mcq_prompt_includes_question_options_and_letter_instruction():
    question = "Which number has one factor?"
    option_a = "one"
    option_b = "two"
    option_c = "three"
    option_d = "four"

    prompt = build_direct_mcq_prompt(question, option_a, option_b, option_c, option_d)

    assert question in prompt
    assert "Respond with only the letter." in prompt
    assert prompt.index("A. one") < prompt.index("B. two")
    assert prompt.index("B. two") < prompt.index("C. three")
    assert prompt.index("C. three") < prompt.index("D. four")

def test_build_free_text_prompt_includes_question_and_excludes_options():
    question = "Which number has one factor?"
    option_a = "one"
    option_b = "two"
    option_c = "three"
    option_d = "four"

    actual = build_free_text_prompt(question)

    assert question in actual
    assert "Options:" not in actual
    assert "A." not in actual
    assert "B." not in actual
    assert "C." not in actual
    assert "D." not in actual

def test_build_option_matching_prompt_includes_question_free_text_options_and_letter_instruction():
    question = "Which number has one factor?"
    option_a = "one"
    option_b = "two"
    option_c = "three"
    option_d = "four"
    free_response = "one"

    prompt = build_option_matching_prompt(question,free_response, option_a, option_b, option_c, option_d)

    assert "Select the option that best matches the reference answer in the context of the question.".lower() in prompt.lower()
    assert question in prompt
    assert "Respond with only the letter." in prompt
    assert prompt.index("A. one") < prompt.index("B. two")
    assert prompt.index("B. two") < prompt.index("C. three")
    assert prompt.index("C. three") < prompt.index("D. four")
    assert free_response in prompt