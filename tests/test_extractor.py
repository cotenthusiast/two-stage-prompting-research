# test/test_extractor.py

from core.extractor import extract_answer

def test_extract_answer_logic(mock_extractor_data):
    """
    Tests the answer extraction logic across various 
    simple and chatty LLM response formats.
    """
    for case in mock_extractor_data:
        # Arrange
        response_text = case["response"]
        expected_letter = case["expected"]
        
        # Act
        actual_result = extract_answer(response_text)
        
        # Assert
        assert actual_result == expected_letter, f"Failed on input: {response_text}"