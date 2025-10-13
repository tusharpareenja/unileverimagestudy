#!/usr/bin/env python3
"""
Test script for the respondent study info API endpoint
"""

import requests
import json

def test_respondent_study_info_api():
    """Test the respondent study info API endpoint"""
    
    # Test with the study ID and respondent ID from the URL you provided
    study_id = "5dadaab8-6739-4304-807b-42b5f7f1e057"
    respondent_id = 6
    base_url = "https://api.mindsurvey.mindgenome.org"
    
    # Test the respondent study info endpoint
    url = f"{base_url}/api/v1/responses/respondent/{respondent_id}/study/{study_id}/info"
    
    print(f"Testing: {url}")
    print("="*80)
    
    try:
        response = requests.get(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print()
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS - Response Data:")
            print(json.dumps(data, indent=2))
            
            # Check if the language field is present in study_info
            if "study_info" in data and "language" in data["study_info"]:
                print(f"\n✅ language field found in study_info: {data['study_info']['language']}")
            else:
                print("\n❌ language field missing from study_info!")
                
        else:
            print(f"❌ ERROR - Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_respondent_study_info_api()

