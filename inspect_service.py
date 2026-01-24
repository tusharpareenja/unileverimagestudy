import sys
import os
import inspect

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.services.response import StudyResponseService

try:
    method = getattr(StudyResponseService, 'generate_csv_rows_for_study_pandas')
    print(f"Found method: {method}")
    print(f"Source file: {inspect.getsourcefile(method)}")
    lines, start_line = inspect.getsourcelines(method)
    print(f"Line number: {start_line}")
except AttributeError:
    print("Method generate_csv_rows_for_study_pandas NOT found in StudyResponseService")
    # List all methods
    print("Available methods:")
    for name, member in inspect.getmembers(StudyResponseService):
        print(f"- {name}")

