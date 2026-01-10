
import sys
import os
import unittest
from unittest.mock import MagicMock
from uuid import uuid4

# Add app to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.study_member_service import StudyMemberService
from app.schemas.study_schema import StudyMemberInvite
from fastapi import HTTPException
from unittest.mock import patch

class TestStudyMemberPermissions(unittest.TestCase):
    def setUp(self):
        # Patch email_service
        self.email_patcher = patch('app.services.study_member_service.email_service')
        self.mock_email_service = self.email_patcher.start()
        
        self.service = StudyMemberService()
        self.db = MagicMock()
        
        # Identifiers
        self.study_id = uuid4()
        self.creator_id = uuid4()
        self.editor_member_id = uuid4()
        
        # Mocks
        self.study = MagicMock()
        self.study.id = self.study_id
        self.study.creator_id = self.creator_id
        self.study.title = "Test Study"
        
        self.editor_user = MagicMock()
        self.editor_user.id = self.editor_member_id
        self.editor_user.name = "Editor"
        
        # Member Mock objects
        self.editor_member_obj = MagicMock()
        self.editor_member_obj.role = 'editor'
        self.editor_member_obj.study_id = self.study_id
        self.editor_member_obj.user_id = self.editor_member_id

        self.viewer_member_obj = MagicMock()
        self.viewer_member_obj.role = 'viewer'

        # User to be invited
        self.invited_user_obj = MagicMock()
        self.invited_user_obj.id = uuid4()
        self.invited_user_obj.name = "Invited User"

        # Mock DB get
        def get_side_effect(model, id):
            if id == self.study_id:
                return self.study
            return None
        self.db.get.side_effect = get_side_effect

    def tearDown(self):
        self.email_patcher.stop()

    def test_editor_invite_viewer_should_succeed(self):
        # Setup: Current user is editor
        # Call sequence of scalars().first():
        # 1. inviter permissions check -> returns editor_member_obj
        # 2. existing member check -> returns None (not already member)
        # 3. user exists check -> returns invited_user_obj
        
        self.db.scalars.return_value.first.side_effect = [
            self.editor_member_obj,
            None,
            self.invited_user_obj
        ]
        
        # Invite payload
        payload = StudyMemberInvite(email="new@example.com", role="viewer")
        
        print("\n[TEST] Testing Editor inviting Viewer (Should SUCCEED)...")
        try:
            result = self.service.invite_member(self.db, self.study_id, self.editor_user, payload)
            print("SUCCESS")
            # Verify it didn't crash
            self.assertTrue(result)
        except HTTPException as e:
            print(f"FAILED with {e.status_code} {e.detail}")
            self.fail(f"Should not have raised HTTPException: {e.detail}")
        except Exception as e:
            print(f"FAILED with unexpected exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Unexpected exception: {e}")

    def test_editor_invite_admin_should_fail(self):
        # Setup: Current user is editor
        # Call sequence:
        # 1. inviter permissions check -> returns editor_member_obj
        
        self.db.scalars.return_value.first.side_effect = [
            self.editor_member_obj,
            None, 
            self.invited_user_obj
        ]
        
        print("\n[TEST] Testing Editor inviting Admin (Should FAIL)...")
        payload = StudyMemberInvite(email="admin@example.com", role="admin")
        
        try:
            self.service.invite_member(self.db, self.study_id, self.editor_user, payload)
            print("SUCCESS (Should have failed!)")
            self.fail("Should have raised 403")
        except HTTPException as e:
            print(f"CAUGHT EXPECTED ERROR: {e.status_code} {e.detail}")
            self.assertEqual(e.status_code, 403)
            self.assertEqual(e.detail, "Editors cannot invite admins")

if __name__ == '__main__':
    unittest.main()
