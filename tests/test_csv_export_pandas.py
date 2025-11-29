
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from uuid import uuid4
import sys
import os

# Add app to path
sys.path.append(os.getcwd())

from app.services.response import StudyResponseService
from app.models.study_model import Study, StudyClassificationQuestion, StudyCategory, StudyElement, StudyLayer, LayerImage

class TestPandasExport(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.service = StudyResponseService(self.db)
        self.study_id = uuid4()

    @patch('app.services.response.pd.read_sql')
    def test_generate_csv_rows_pandas_grid(self, mock_read_sql):
        # Setup Mock Data
        
        # 1. Study
        study = MagicMock(spec=Study)
        study.study_type = 'grid'
        self.db.get.return_value = study

        # 2. Responses DataFrame
        responses_data = {
            'id': [uuid4(), uuid4()],
            'session_id': ['sess1', 'sess2'],
            'personal_info': [{'gender': 'Male', 'age': 25}, {'gender': 'Female', 'dob': '1990-01-01'}]
        }
        responses_df = pd.DataFrame(responses_data)
        
        # 3. Tasks DataFrame
        tasks_data = {
            'study_response_id': [responses_data['id'][0], responses_data['id'][1]],
            'task_index': [0, 0],
            'rating_given': [5.0, 4.0],
            'task_duration_seconds': [10.5, 12.0],
            'elements_shown_in_task': [{'Cat1_1': 1}, {'Cat1_1': 0}],
            'elements_shown_content': [{}, {}],
            'layers_shown_in_task': [{}, {}]
        }
        tasks_df = pd.DataFrame(tasks_data)

        # 4. Answers DataFrame
        answers_data = {
            'study_response_id': [responses_data['id'][0]],
            'question_id': ['q1'],
            'answer': ['opt1 - Option 1']
        }
        answers_df = pd.DataFrame(answers_data)

        # Configure mock_read_sql to return these DFs in order
        # Order in code: responses, tasks, answers
        mock_read_sql.side_effect = [responses_df, tasks_df, answers_df]

        # 5. Questions Config
        q1 = MagicMock(spec=StudyClassificationQuestion)
        q1.question_id = 'q1'
        q1.question_text = 'Question 1'
        q1.answer_options = [{'code': 'opt1', 'text': 'Option 1'}]
        q1.order = 1
        
        # 6. Grid Config
        cat1 = MagicMock(spec=StudyCategory)
        cat1.id = uuid4()
        cat1.name = 'Cat1'
        cat1.order = 1
        
        el1 = MagicMock(spec=StudyElement)
        el1.element_id = uuid4()
        el1.name = 'Elem1'
        el1.category_id = cat1.id
        
        # Mock db.execute for questions and grid
        # Calls:
        # 1. questions
        # 2. categories
        # 3. elements (for cat1)
        
        mock_result_questions = MagicMock()
        mock_result_questions.scalars.return_value.all.return_value = [q1]
        
        mock_result_cats = MagicMock()
        mock_result_cats.scalars.return_value.all.return_value = [cat1]
        
        mock_result_elems = MagicMock()
        mock_result_elems.scalars.return_value.all.return_value = [el1]
        
        self.db.execute.side_effect = [mock_result_questions, mock_result_cats, mock_result_elems]

        # Run
        chunks = list(self.service.generate_csv_rows_for_study_pandas(self.study_id))
        full_csv = "".join(chunks)
        
        print("\nGenerated CSV:")
        print(full_csv)
        
        # Verify
        self.assertIn("Panelist", full_csv)
        self.assertIn("Question 1", full_csv)
        self.assertIn("Cat1-Elem1", full_csv)
        self.assertIn("sess1", full_csv)
        self.assertIn("Option 1", full_csv) # Formatted answer
        self.assertIn("Male", full_csv)

if __name__ == '__main__':
    unittest.main()
