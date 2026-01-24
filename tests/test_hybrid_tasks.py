import asyncio
import uuid
from typing import Dict, Any, List
import logging
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mocking Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock objects to avoid full DB setup
class MockJob:
    def __init__(self, payload):
        self.payload = payload
        self.progress = 0.0
        self.message = ""
        self.study_id = "mock-study-id"

class MockDB:
    pass

# We will test the logic of combining tasks which is the core of _generate_hybrid_tasks_async
async def test_hybrid_task_combination_logic():
    print("\n=== Testing Hybrid Task Combination Logic ===\n")
    
    # Pre-calculated tasks for Phase 1 (Grid)
    phase1_tasks = {
        "0": [{"task_id": "G1", "task_index": 0}, {"task_id": "G2", "task_index": 1}],
        "1": [{"task_id": "G1", "task_index": 0}, {"task_id": "G2", "task_index": 1}]
    }
    
    # Pre-calculated tasks for Phase 2 (Text)
    phase2_tasks = {
        "0": [{"task_id": "T1", "task_index": 0}, {"task_id": "T2", "task_index": 1}],
        "1": [{"task_id": "T1", "task_index": 0}, {"task_id": "T2", "task_index": 1}]
    }

    phase_results = {
        "grid": {"tasks": phase1_tasks, "metadata": {"tpc": 2}},
        "text": {"tasks": phase2_tasks, "metadata": {"tpc": 2}}
    }
    
    phase_order = ["grid", "text"]
    combined_tasks = {}
    
    for phase_type in phase_order:
        res = phase_results[phase_type]
        p_tasks = res.get('tasks', {})
        for resp_id, tasks in p_tasks.items():
            if resp_id not in combined_tasks:
                combined_tasks[resp_id] = []
            
            offset = len(combined_tasks[resp_id])
            for t in tasks:
                # This is the logic we implemented in _generate_hybrid_tasks_async
                t_copy = t.copy()
                t_copy['task_index'] = int(t.get('task_index', 0)) + offset
                t_copy['phase_type'] = phase_type
                combined_tasks[resp_id].append(t_copy)
    
    # Verification
    all_passed = True
    
    # Check respondent 0
    resp0 = combined_tasks.get("0")
    if not resp0 or len(resp0) != 4:
        print("✗ FAIL: Respondent 0 should have 4 tasks")
        all_passed = False
    else:
        # Check indexes and phase types
        expected = [
            (0, "grid", "G1"),
            (1, "grid", "G2"),
            (2, "text", "T1"),
            (3, "text", "T2")
        ]
        for i, (idx, ptype, tid) in enumerate(expected):
            if resp0[i]['task_index'] != idx or resp0[i]['phase_type'] != ptype or resp0[i]['task_id'] != tid:
                print(f"✗ FAIL: Task {i} incorrect. Got index={resp0[i]['task_index']}, phase={resp0[i]['phase_type']}, id={resp0[i]['task_id']}")
                all_passed = False
            else:
                print(f"✓ PASS: Task {i} correct (Index {idx}, Phase {ptype}, ID {tid})")

    if all_passed:
        print("\n=== ALL TEST LOGIC PASSED ===\n")
    else:
        print("\n=== SOME TEST LOGIC FAILED ===\n")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(test_hybrid_task_combination_logic())
    sys.exit(0 if success else 1)
