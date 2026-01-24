import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from uuid import uuid4

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.services.response import StudyResponseService
    from app.models.response_model import StudyResponse, CompletedTask
    from app.models.study_model import Study, StudyElement
    from app.schemas.response_schema import SubmitTaskRequest
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_hybrid_submission_enrichment():
    print("Testing hybrid submission enrichment...")
    
    # Setup mocks
    db = MagicMock()
    service = StudyResponseService(db)
    
    study_id = uuid4()
    session_id = "test_session"
    respondent_id = 1
    
    # Mock Study
    study = Study()
    study.id = study_id
    study.study_type = 'hybrid'
    study.tasks = {
        "1": [
            {
                "task_id": "1_0",
                "task_index": 0,
                "phase_type": "grid",
                "elements_shown_content": {
                    "CAT1_IMG1": {
                        "name": "IMG1",
                        "content": "http://img1.url",
                        "element_id": str(uuid4())
                    }
                }
            }
        ]
    }
    
    # Mock StudyResponse
    response = StudyResponse()
    response.id = uuid4()
    response.study_id = study_id
    response.session_id = session_id
    response.respondent_id = respondent_id
    response.current_task_index = 0
    response.total_tasks_assigned = 1
    response.completed_tasks_count = 0
    
    # Configure DB mocks
    db.get.side_effect = lambda model, id: study if model == Study else None
    
    # Mock get_response_by_session to return our response
    service.get_response_by_session = MagicMock(return_value=response)
    
    # Mock StudyElement query
    mock_element = MagicMock()
    mock_element.element_id = "E1"
    mock_element.name = "IMG1"
    db.execute.return_value.scalars.return_value.all.return_value = [mock_element]
    
    # Create request
    request = SubmitTaskRequest(
        task_id="1_0",
        rating_given=5,
        task_duration_seconds=10.0,
        element_interactions=[]
    )
    
    # Run submit_task
    with patch('app.services.response.datetime') as mock_datetime:
        mock_datetime.utcnow.return_value = datetime(2023, 1, 1)
        mock_datetime.fromtimestamp = datetime.fromtimestamp
        mock_datetime.now = datetime.now
        
        service.submit_task(session_id, request)
    
    # Verify that db.add was called with a CompletedTask
    added_objects = [call.args[0] for call in db.add.call_args_list]
    completed_task = None
    for obj in added_objects:
        if obj.__class__.__name__ == 'CompletedTask':
            completed_task = obj
            break
            
    if not completed_task:
        print("Error: CompletedTask not added to DB")
        return False
    
    print(f"Recorded Task Type: {completed_task.task_type}")
    print(f"Enriched Content: {completed_task.elements_shown_content}")
    
    assert completed_task.task_type == "grid", f"Expected task_type 'grid', got '{completed_task.task_type}'"
    assert completed_task.elements_shown_content is not None, "elements_shown_content should not be None"
    assert "CAT1_IMG1" in completed_task.elements_shown_content, "CAT1_IMG1 should be in elements_shown_content"
    
    print("Test passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_hybrid_submission_enrichment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
