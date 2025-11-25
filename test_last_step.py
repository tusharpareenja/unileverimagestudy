"""Test script to verify last_step update logic"""

# Test the forward-only logic
def test_forward_only():
    # Simulate current step = 7
    current_step = 7

    # Test 1: Try to update to 4 (should not update)
    payload_step = 4
    if payload_step > current_step:
        print(f"[FAIL] Would update from {current_step} to {payload_step} - WRONG!")
    else:
        print(f"[PASS] Step stays at {current_step} (tried to set {payload_step})")

    # Test 2: Try to update to 8 (should update)
    payload_step = 8
    if payload_step > current_step:
        print(f"[PASS] Would update from {current_step} to {payload_step} - CORRECT!")
    else:
        print(f"[FAIL] Step stays at {current_step} - WRONG!")

    # Test 3: Try to update to 7 (should not update, equal)
    payload_step = 7
    if payload_step > current_step:
        print(f"[FAIL] Would update from {current_step} to {payload_step} - WRONG!")
    else:
        print(f"[PASS] Step stays at {current_step} (tried to set {payload_step})")

if __name__ == "__main__":
    test_forward_only()
