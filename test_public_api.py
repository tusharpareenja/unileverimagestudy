#!/usr/bin/env python3
"""
Test script for the public studies API endpoint
"""

import requests
import json

def test_public_study_api():
    """Test the public study API endpoint"""
    
    # Test with the study ID from the URL you provided
    study_id = "1ef5cb74-9072-402a-8902-efe6270cb6d9"
    base_url = "https://api.mindsurvey.mindgenome.org"
    
    # Test the public endpoint
    url = f"{base_url}/api/v1/studies/public/{study_id}"
    
    print(f"Testing: {url}")
    print("="*60)
    
    try:
        response = requests.get(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS - Response Data:")
            print(json.dumps(data, indent=2))
            
            # Check if the new fields are present
            if "orientation_text" in data:
                print("\n✅ orientation_text field found!")
            else:
                print("\n❌ orientation_text field missing!")
                
            if "language" in data:
                print("✅ language field found!")
            else:
                print("❌ language field missing!")
                
        else:
            print(f"❌ ERROR - Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_public_study_api()
