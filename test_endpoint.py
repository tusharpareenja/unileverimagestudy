#!/usr/bin/env python3
"""
Test the /generate-tasks endpoint to see if background jobs work
"""
import requests
import json
import time

# Test data for a study with 10 respondents (should trigger background processing with threshold=5)
test_payload = {
    "study_type": "grid",
    "audience_segmentation": {
        "number_of_respondents": 10
    },
    "categories": [
        {
            "category_id": "cat1",
            "name": "Test Category"
        }
    ],
    "elements": [
        {
            "element_id": "el1",
            "name": "Test Element",
            "content": "Test content",
            "category_id": "cat1",
            "element_type": "image"
        }
    ]
}

def test_generate_tasks():
    """Test the generate-tasks endpoint"""
    try:
        print("🧪 Testing /generate-tasks endpoint...")
        
        # Make request to your local server
        response = requests.post(
            "http://localhost:8000/api/v1/studies/generate-tasks",
            json=test_payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            data = response.json()
            if "metadata" in data and "job_id" in data["metadata"]:
                job_id = data["metadata"]["job_id"]
                print(f"✅ Background job created: {job_id}")
                
                # Test status endpoint
                print(f"\n🔍 Testing status endpoint...")
                status_response = requests.get(f"http://localhost:8000/api/v1/studies/generate-tasks/status/{job_id}")
                print(f"Status Response: {json.dumps(status_response.json(), indent=2)}")
                
                return True
            else:
                print("❌ No job_id in response - using synchronous processing")
                return False
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing background job endpoint...")
    print("=" * 50)
    
    success = test_generate_tasks()
    
    if success:
        print("\n✅ Background job test completed!")
    else:
        print("\n❌ Background job test failed!")
