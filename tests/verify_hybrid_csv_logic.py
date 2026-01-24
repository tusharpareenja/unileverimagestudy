import sys
import os
import unittest
from unittest.mock import MagicMock, ANY, patch
from uuid import uuid4

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.services.response import StudyResponseService
# Use MagicMock for models
Study = MagicMock()
StudyResponse = MagicMock()
StudyClassificationQuestion = MagicMock()
CompletedTask = MagicMock()
ClassificationAnswer = MagicMock()
StudyCategory = MagicMock()
StudyElement = MagicMock()

# Mock proper behaviors for models
Study.__name__ = 'Study'
StudyResponse.__name__ = 'StudyResponse'
StudyClassificationQuestion.__name__ = 'StudyClassificationQuestion'
CompletedTask.__name__ = 'CompletedTask'
ClassificationAnswer.__name__ = 'ClassificationAnswer'
StudyCategory.__name__ = 'StudyCategory'
StudyElement.__name__ = 'StudyElement'

def test_generate_csv_rows_hybrid(mock_select):
    # Setup mock_select to return a recognizable object based on input
    def select_side_effect(*args):
        mock_query = MagicMock()
        str_repr = "SELECT Unknown"
        if args[0] == StudyCategory:
            str_repr = "SELECT StudyCategory"
        elif args[0] == StudyElement:
            str_repr = "SELECT StudyElement"
        elif args[0] == StudyResponse:
            str_repr = "SELECT StudyResponse"
        elif args[0] == StudyClassificationQuestion:
            str_repr = "SELECT StudyClassificationQuestion"
        elif args[0] == CompletedTask:
            str_repr = "SELECT CompletedTask"
            
        mock_query.__str__ = lambda *a: str_repr
        # Support chaining for .where().order_by()
        mock_query.where.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        return mock_query

    mock_select.side_effect = select_side_effect

    # Mock DB
    mock_db = MagicMock()
    service = StudyResponseService(mock_db)

    # Setup mock study as 'hybrid'
    mock_study = MagicMock()
    mock_study.study_type = 'hybrid'
    mock_study.id = uuid4()
    
    # Setup mock retrieval for study
    mock_db.get.return_value = mock_study

    # Setup mock responses
    mock_response = MagicMock()
    mock_response.id = uuid4()
    mock_response.session_id = "sess_001"
    mock_row = MagicMock()
    mock_row.__getitem__ = lambda s, x: {0: mock_response.id, 1: "sess_001", 2: {}}.get(x)

    # Setup categories and elements
    cat1 = MagicMock()
    cat1.name = "MyCategory"
    cat1.id = uuid4()
    
    elem1 = MagicMock()
    elem1.name = "MyElement"
    elem1.element_id = uuid4()

    def execute_side_effect(*args, **kwargs):
        query = args[0]
        str_query = str(query)
        print(f"DEBUG: execute called with query: {str_query}")
        result = MagicMock()
        
        if "SELECT StudyResponse" in str_query:
            result.all.return_value = [mock_row]
            result.scalars.return_value.all.return_value = [mock_response]
        elif "SELECT StudyClassificationQuestion" in str_query:
            result.scalars.return_value.all.return_value = []
        elif "SELECT CompletedTask" in str_query:
            result.scalars.return_value.all.return_value = []
        elif "SELECT StudyCategory" in str_query:
            # categories - THIS IS WHAT WE WANT TO CHECK
            result.scalars.return_value.all.return_value = [cat1]
        elif "SELECT StudyElement" in str_query:
            # elements - THIS IS WHAT WE WANT TO CHECK
            result.scalars.return_value.all.return_value = [elem1]
        elif "StudyLayer" in str_query:
            result.scalars.return_value.all.return_value = []
        else:
                result.all.return_value = []
                result.scalars.return_value.all.return_value = []
        return result

    mock_db.execute.side_effect = execute_side_effect
    
    # Run function
    print("running service.generate_csv_rows_for_study")
    gen = service.generate_csv_rows_for_study(mock_study.id)
    
    header = next(gen)
    print(f"Generated Header: {header}")
    
    expected = "MyCategory-MyElement"
    if expected in header:
        print("SUCCESS: Found expected hybrid header column.")
    else:
        raise Exception(f"FAILURE: Did not find {expected} in header.")

if __name__ == '__main__':
    try:
        # Patch manually
        with patch('app.services.response.select') as mock_select:
             test_generate_csv_rows_hybrid(mock_select)
        print("\nTEST PASSED\n")
    except Exception as e:
        print(f"\nTEST FAILED: {e}\n")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
