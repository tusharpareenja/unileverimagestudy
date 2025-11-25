"""Test script to verify classification question schema conversion"""

from uuid import uuid4
from app.schemas.study_schema import StudyClassificationQuestionOut

def test_is_required_conversion():
    print("Testing is_required Y/N to boolean conversion...")

    # Mock a database question object with 'Y'/'N' strings
    class MockQuestion:
        def __init__(self, is_required_value):
            self.id = uuid4()
            self.question_id = "Q1"
            self.question_text = "Test question?"
            self.question_type = "multiple_choice"
            self.is_required = is_required_value  # 'Y' or 'N' from database
            self.order = 1
            self.answer_options = [{"id": "A", "text": "Option A", "order": 1}]
            self.config = None

    # Test 1: 'Y' should convert to True
    mock_q1 = MockQuestion('Y')
    try:
        q1_out = StudyClassificationQuestionOut.model_validate(mock_q1)
        result = q1_out.model_dump()
        if result['is_required'] is True:
            print(f"[PASS] 'Y' converted to True")
        else:
            print(f"[FAIL] 'Y' converted to {result['is_required']} instead of True")
    except Exception as e:
        print(f"[ERROR] Failed to convert 'Y': {e}")

    # Test 2: 'N' should convert to False
    mock_q2 = MockQuestion('N')
    try:
        q2_out = StudyClassificationQuestionOut.model_validate(mock_q2)
        result = q2_out.model_dump()
        if result['is_required'] is False:
            print(f"[PASS] 'N' converted to False")
        else:
            print(f"[FAIL] 'N' converted to {result['is_required']} instead of False")
    except Exception as e:
        print(f"[ERROR] Failed to convert 'N': {e}")

    # Test 3: lowercase 'y' should convert to True
    mock_q3 = MockQuestion('y')
    try:
        q3_out = StudyClassificationQuestionOut.model_validate(mock_q3)
        result = q3_out.model_dump()
        if result['is_required'] is True:
            print(f"[PASS] lowercase 'y' converted to True")
        else:
            print(f"[FAIL] lowercase 'y' converted to {result['is_required']} instead of True")
    except Exception as e:
        print(f"[ERROR] Failed to convert 'y': {e}")

    # Test 4: Already boolean True should stay True
    mock_q4 = MockQuestion(True)
    try:
        q4_out = StudyClassificationQuestionOut.model_validate(mock_q4)
        result = q4_out.model_dump()
        if result['is_required'] is True:
            print(f"[PASS] Boolean True stays True")
        else:
            print(f"[FAIL] Boolean True became {result['is_required']}")
    except Exception as e:
        print(f"[ERROR] Failed with boolean True: {e}")

if __name__ == "__main__":
    test_is_required_conversion()
