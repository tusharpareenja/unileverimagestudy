"""Tests for Unilever-only flattened-csv raw data (domain check and dataframe columns)."""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from uuid import uuid4

# Import all models so SQLAlchemy relationship names resolve when get_study_dataframe builds ORM queries.
import app.models.user_model  # noqa: F401
import app.models.project_model  # noqa: F401
import app.models.study_model  # noqa: F401
import app.models.response_model  # noqa: F401

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
        self.assertEqual(sorted(df["Panelist"].tolist()), sorted(["P001", "sess2"]))
        self.assertEqual(df["Product ID"].iloc[0], "PROD-123")
        self.assertEqual(df["Task type"].iloc[0], "Image")


class TestYesNoClassificationExportDomain(unittest.TestCase):
    """Yes/No answers export as 0/1 only when unilever_format=True (unilever.com downloads)."""

    @patch("app.services.response.pd.read_sql")
    def test_yes_no_strings_for_non_unilever_numeric_for_unilever(self, mock_read_sql):
        from app.services.response import StudyResponseService

        rid1, rid2 = uuid4(), uuid4()
        cat_id = uuid4()

        q_yesno = MagicMock()
        q_yesno.question_id = "qy"
        q_yesno.question_text = "Like fragrance?"
        q_yesno.answer_options = [
            {"id": "Y", "text": "Yes"},
            {"id": "N", "text": "No"},
        ]
        q_yesno.order = 1

        cat = MagicMock()
        cat.id = cat_id
        cat.name = "Cat1"
        cat.order = 1
        el = MagicMock()
        el.element_id = uuid4()
        el.name = "Elem1"
        el.category_id = cat.id

        tasks_df = pd.DataFrame(
            {
                "study_response_id": [rid1, rid2],
                "task_index": [0, 0],
                "rating_given": [5, 4],
                "task_duration_seconds": [10.0, 12.0],
                "elements_shown_in_task": [{"Cat1_1": 1}, {"Cat1_1": 0}],
                "elements_shown_content": [{}, {}],
                "layers_shown_in_task": [None, None],
                "task_type": ["grid", "grid"],
            }
        )
        answers_df = pd.DataFrame(
            {
                "study_response_id": [rid1, rid2],
                "question_id": ["qy", "qy"],
                "answer": ["Y", "N"],
            }
        )

        for unilever_format, expected in [(False, ["Yes", "No"]), (True, [1, 0])]:
            study = MagicMock()
            study.study_type = "grid"
            study.product_id = "P1"
            study.product_keys = []

            db = MagicMock()
            db.get.return_value = study

            if unilever_format:
                responses_df = pd.DataFrame(
                    {
                        "id": [rid1, rid2],
                        "session_id": ["s1", "s2"],
                        "personal_info": [{"gender": "M", "age": 20}, {"gender": "F", "age": 21}],
                        "panelist_id": [None, None],
                        "session_start_time": [
                            pd.Timestamp("2024-01-01"),
                            pd.Timestamp("2024-01-02"),
                        ],
                        "respondent_id": [1, 2],
                    }
                )
            else:
                responses_df = pd.DataFrame(
                    {
                        "id": [rid1, rid2],
                        "session_id": ["s1", "s2"],
                        "personal_info": [{"gender": "M", "age": 20}, {"gender": "F", "age": 21}],
                        "respondent_id": [1, 2],
                    }
                )

            mock_read_sql.side_effect = [responses_df, tasks_df.copy(), answers_df.copy()]
            db.execute.return_value.scalars.return_value.all.side_effect = [
                [q_yesno],
                [cat],
                [el],
            ]

            service = StudyResponseService(db)
            df = service.get_study_dataframe(uuid4(), unilever_format=unilever_format)

            col = "Like fragrance?"
            self.assertIn(col, df.columns)
            # Row order follows merge/sort, not input row order — assert per panelist (rid1 -> s1 -> Y, rid2 -> s2 -> N)
            by_panelist = df.set_index("Panelist")[col].to_dict()
            self.assertEqual(by_panelist["s1"], expected[0])
            self.assertEqual(by_panelist["s2"], expected[1])
