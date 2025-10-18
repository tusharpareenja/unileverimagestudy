#!/usr/bin/env python3
"""
Debug script to test PUT /api/v1/studies endpoint
"""
import requests
import json

# Test data that might cause validation errors
test_cases = [
    {
        "name": "Invalid rating scale max_value",
        "data": {
            "rating_scale": {
                "min_value": 1,
                "max_value": 10,  # Invalid - should be 5, 7, or 9
                "min_label": "Not at all",
                "max_label": "Very much"
            }
        }
    },
    {
        "name": "Invalid rating scale min_value",
        "data": {
            "rating_scale": {
                "min_value": 0,  # Invalid - should be 1-9
                "max_value": 5,
                "min_label": "Not at all",
                "max_label": "Very much"
            }
        }
    },
    {
        "name": "min_value exceeds max_value",
        "data": {
            "rating_scale": {
                "min_value": 5,
                "max_value": 3,  # Invalid - min > max
                "min_label": "Not at all",
                "max_label": "Very much"
            }
        }
    },
    {
        "name": "Missing rating scale fields",
        "data": {
            "rating_scale": {
                "min_value": 1,
                "max_value": 5
                # Missing min_label and max_label
            }
        }
    },
    {
        "name": "Invalid status",
        "data": {
            "status": "invalid_status"  # Should be draft, active, paused, or completed
        }
    },
    {
        "name": "Title too long",
        "data": {
            "title": "A" * 256  # Exceeds 255 character limit
        }
    }
]

def test_put_endpoint():
    base_url = "http://127.0.0.1:8000"
    study_id = "9070740a-d3e5-4972-b3a1-2a9e94142b69"  # From your error
    
    print("🔍 Testing PUT /api/v1/studies endpoint validation...")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        print(f"   Data: {json.dumps(test_case['data'], indent=2)}")
        
        try:
            response = requests.put(
                f"{base_url}/api/v1/studies/{study_id}",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 400:
                print(f"   Error: {response.text}")
            else:
                print(f"   Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    test_put_endpoint()
