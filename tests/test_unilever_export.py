"""Tests for Unilever-only flattened-csv raw data (domain check and dataframe columns)."""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from uuid import uuid4

from app.core.domain import is_unilever_domain


class TestIsUnileverDomain(unittest.TestCase):
    def test_unilever_com_domain_returns_true(self):
        self.assertTrue(is_unilever_domain("tusharpareenja@unilever.com"))
        self.assertTrue(is_unilever_domain("user@UNILEVER.COM"))
        self.assertTrue(is_unilever_domain(" a@unilever.com "))

    def test_other_domain_returns_false(self):
        self.assertFalse(is_unilever_domain("user@gmail.com"))
        self.assertFalse(is_unilever_domain("user@unilever.co.uk"))
        self.assertFalse(is_unilever_domain("user@company.com"))

    def test_empty_or_invalid_returns_false(self):
        self.assertFalse(is_unilever_domain(""))
        self.assertFalse(is_unilever_domain(None))
        self.assertFalse(is_unilever_domain("no-at-sign"))
        self.assertFalse(is_unilever_domain(" @unilever.com"))


class TestGetStudyDataframeUnileverColumns(unittest.TestCase):
    """Assert that get_study_dataframe(..., unilever_format=True) adds Unilever columns."""

    @patch("app.services.response.pd.read_sql")
    def test_unilever_format_adds_expected_columns(self, mock_read_sql):
        from app.services.response import StudyResponseService

        db = MagicMock()
        study = MagicMock()
        study.study_type = "grid"
        study.product_id = "PROD-123"
        study.product_keys = [
            {"name": "Gourmand", "percentage": 50},
            {"name": "Fresh", "percentage": 50},
        ]
        db.get.return_value = study

        rid1, rid2 = uuid4(), uuid4()
        responses_data = {
            "id": [rid1, rid2],
            "session_id": ["sess1", "sess2"],
            "personal_info": [{"gender": "Male", "age": 25}, {"gender": "Female", "age": 30}],
            "panelist_id": ["P001", None],
        }
        responses_df = pd.DataFrame(responses_data)

        tasks_data = {
            "study_response_id": [rid1, rid2],
            "task_index": [0, 0],
            "rating_given": [5, 4],
            "task_duration_seconds": [10.0, 12.0],
            "elements_shown_in_task": [{"Cat1_1": 1}, {"Cat1_1": 0}],
            "elements_shown_content": [{}, {}],
            "layers_shown_in_task": [None, None],
            "task_type": ["grid", "grid"],
        }
        tasks_df = pd.DataFrame(tasks_data)

        answers_df = pd.DataFrame({
            "study_response_id": [rid1, rid2],
            "question_id": ["q1", "q1"],
            "answer": ["A", "B"],
        })

        q = MagicMock()
        q.question_id = "q1"
        q.question_text = "Q1"
        q.answer_options = []
        q.order = 1

        cat = MagicMock()
        cat.id = uuid4()
        cat.name = "Cat1"
        cat.order = 1
        el = MagicMock()
        el.element_id = uuid4()
        el.name = "Elem1"
        el.category_id = cat.id

        mock_read_sql.side_effect = [responses_df, tasks_df, answers_df]
        db.execute.return_value.scalars.return_value.all.side_effect = [
            [q],
            [cat],
            [el],
        ]

        service = StudyResponseService(db)
        df = service.get_study_dataframe(uuid4(), unilever_format=True)

        self.assertIn("Panelist", df.columns)
        self.assertIn("Product ID", df.columns)
        self.assertIn("Gourmand", df.columns)
        self.assertIn("Fresh", df.columns)
        self.assertIn("Task type", df.columns)
        self.assertEqual(list(df["Panelist"].dropna().values), ["P001", "sess2"])
        self.assertEqual(df["Product ID"].iloc[0], "PROD-123")
        self.assertEqual(df["Task type"].iloc[0], "Image")
