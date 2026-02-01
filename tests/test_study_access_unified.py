
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from fastapi import HTTPException

# Add app to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.study import _load_owned_study_minimal, _load_owned_study_launch_only, regenerate_tasks
from app.api.v1.study import _ensure_study_exists, GenerateTasksRequest
from app.models.study_model import Study, StudyMember
from app.models.user_model import User

class TestUnifiedStudyAccess(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.study_id = uuid4()
        self.creator_id = uuid4()
        self.member_id = uuid4()
        
        self.study = Study(id=self.study_id, creator_id=self.creator_id, title="Test Study", status="draft")
        self.current_user = User(id=self.member_id, is_active=True)
        self.creator_user = User(id=self.creator_id, is_active=True)

    def test_load_minimal_creator_success(self):
        # Mock finding by creator_id
        self.db.scalars.return_value.first.return_value = self.study
        
        result = _load_owned_study_minimal(self.db, self.study_id, self.creator_id)
        self.assertEqual(result.id, self.study_id)
        self.assertEqual(result.creator_id, self.creator_id)

    def test_load_minimal_member_success(self):
        # 1. First call (check creator) returns None
        # 2. Second call (check member) returns study
        self.db.scalars.return_value.first.side_effect = [None, self.study]
        
        result = _load_owned_study_minimal(self.db, self.study_id, self.member_id)
        self.assertEqual(result.id, self.study_id)
        
    def test_load_minimal_no_access_fail(self):
        # Both calls return None
        self.db.scalars.return_value.first.side_effect = [None, None]
        
        with self.assertRaises(HTTPException) as cm:
            _load_owned_study_minimal(self.db, self.study_id, self.member_id)
        self.assertEqual(cm.exception.status_code, 404)

    def test_load_launch_only_member_success(self):
        # Mock result set for execute()
        mock_result = MagicMock()
        mock_result.id = self.study_id
        mock_result.title = "Test Study"
        mock_result.status = "draft"
        mock_result.launched_at = None
        mock_result.creator_id = self.creator_id
        
        # 1. First execute check creator returns None
        # 2. Second execute check member returns result
        self.db.execute.return_value.first.side_effect = [None, mock_result]
        
        result = _load_owned_study_launch_only(self.db, self.study_id, self.member_id)
        self.assertEqual(result.id, self.study_id)

    def test_ensure_study_exists_member_success(self):
        payload = GenerateTasksRequest(study_id=self.study_id, study_type="grid", audience_segmentation={"number_of_respondents": 10})
        
        # scalars().first() sequence: None (creator check), study (member check)
        self.db.scalars.return_value.first.side_effect = [None, self.study]
        
        result = _ensure_study_exists(payload, self.db, self.current_user)
        self.assertEqual(result.id, self.study_id)

    @patch('app.services.study.generate_grid_tasks')
    def test_regenerate_tasks_member_success(self, mock_gen):
        # Mock study load with member check
        self.db.scalars.return_value.first.side_effect = [None, self.study]
        self.db.scalars.return_value.all.return_value = [MagicMock(element_id=1)] # elements
        
        mock_gen.return_value = {"tasks": {"r1": []}, "metadata": {"tasks_per_consumer": 5}}
        
        # We need to mock the study_type and audience
        self.study.study_type = 'grid'
        self.study.audience_segmentation = {"number_of_respondents": 10}
        
        from app.schemas.study_schema import RegenerateTasksResponse
        result = regenerate_tasks(self.db, self.study_id, self.member_id, generator={"grid": mock_gen, "layer": MagicMock()})
        self.assertIsInstance(result, RegenerateTasksResponse)
        self.db.commit.assert_called()

if __name__ == '__main__':
    unittest.main()
