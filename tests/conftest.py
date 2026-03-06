# tests/conftest.py

import pytest

@pytest.fixture
def mock_extractor_data():
    return [
        {"response": "A", "expected": "A"},
        {"response": "A.", "expected": "A"},
        {"response": "A ", "expected": "A"},
        {"response": "A. ", "expected": "A"},
        {"response": "A .", "expected": "A"},
        {"response": " A", "expected": "A"},
        {"response": "A)", "expected": "A"},
        {"response": "The answer is A.", "expected": "A"},
        {"response": "At first, I thought the answer was B, however, I ended up picking A", "expected": "A"},
        {"response": "I am not sure", "expected": None},
        {"response": "ABCD", "expected": None},
    ]