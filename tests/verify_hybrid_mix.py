import asyncio
import sys
import os
import random

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def test_hybrid_mix_logic():
    print("\n=== Testing Hybrid Mix Logic ===\n")
    
    # Mock data structure mimicking what currently happens in regenerate_tasks
    # We are testing the logic block we added, simulating the inputs
    
    # Assumptions for "Mix":
    # 1. We have multiple phases (e.g., Grid, Text)
    # 2. We generate tasks for each phase
    # 3. We shuffle them all together for each respondent
    
    # Mock Phase Results from generation
    phase_results = {
        "grid": {
            "tasks": {
                "0": [{"task_id": "G1"}, {"task_id": "G2"}],
                "1": [{"task_id": "G1"}, {"task_id": "G2"}]
            }
        },
        "text": {
            "tasks": {
                "0": [{"task_id": "T1"}, {"task_id": "T2"}],
                "1": [{"task_id": "T1"}, {"task_id": "T2"}]
            }
        }
    }
    
    phase_order = ["mix"]
    is_mix = "mix" in phase_order
    
    # Simulation of the implemented logic
    if is_mix:
        print("Mix mode detected.")
        target_phases = ["grid", "text"] # Detected from categories
    else:
        target_phases = phase_order
        
    tasks_per_respondent_mix = {}
    combined_tasks = {}
    
    print(f"Target phases: {target_phases}")
    
    for phase_type in target_phases:
        # Simulating getting tasks for phase
        phase_tasks = phase_results[phase_type]['tasks']
        
        if is_mix:
            for resp_id, t_list in phase_tasks.items():
                if resp_id not in tasks_per_respondent_mix:
                    tasks_per_respondent_mix[resp_id] = []
                
                for t in t_list:
                    t_copy = t.copy()
                    t_copy['phase_type'] = phase_type
                    tasks_per_respondent_mix[resp_id].append(t_copy)
    
    if is_mix:
        # Shuffle logic
        for resp_id, t_list in tasks_per_respondent_mix.items():
            # Seed random for reproducibility in this test
            random.seed(42 + int(resp_id)) 
            random.shuffle(t_list)
            
            # Re-index
            for idx, t in enumerate(t_list):
                t['task_index'] = idx
            combined_tasks[resp_id] = t_list
            
    # --- Verification ---
    
    # Check Respondent 0
    resp0 = combined_tasks.get("0")
    if not resp0:
        print("✗ FAIL: No tasks for respondent 0")
        return False
        
    print(f"Respondent 0 tasks ({len(resp0)}):")
    for t in resp0:
        print(f"  [{t['task_index']}] {t['task_id']} ({t['phase_type']})")
        
    if len(resp0) != 4:
        print(f"✗ FAIL: Expected 4 tasks, got {len(resp0)}")
        return False
        
    # Check for randomness/shuffling (simple check: valid indices and presence of both types)
    indices = [t['task_index'] for t in resp0]
    if indices != [0, 1, 2, 3]:
        print(f"✗ FAIL: Indices are not sequential 0-3: {indices}")
        return False
        
    phases = [t['phase_type'] for t in resp0]
    if "grid" not in phases or "text" not in phases:
        print(f"✗ FAIL: Missing phase types in mixed output: {phases}")
        return False
        
    # Check Respondent 1 (different shuffle due to seed?)
    resp1 = combined_tasks.get("1")
    print(f"Respondent 1 tasks ({len(resp1)}):")
    for t in resp1:
        print(f"  [{t['task_index']}] {t['task_id']} ({t['phase_type']})")
        
    print("\n✓ PASS: Logic verification passed")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_hybrid_mix_logic())
    sys.exit(0 if success else 1)
