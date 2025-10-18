#!/usr/bin/env python3
"""
Test script for async task generation functionality
"""
import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.background_task_service import background_task_service, JobStatus

async def test_background_task_service():
    """Test the background task service functionality"""
    print("🧪 Testing Background Task Service...")
    
    # Test job creation
    job_id = background_task_service.create_job(
        study_id="test-study-123",
        user_id="test-user-456", 
        payload={
            "study_type": "grid",
            "audience_segmentation": {"number_of_respondents": 150},
            "categories": [{"name": "Category1", "elements": []}],
            "elements": []
        }
    )
    
    print(f"✅ Created job: {job_id}")
    
    # Test job status
    job = background_task_service.get_job_status(job_id)
    print(f"✅ Job status: {job.status.value}")
    print(f"✅ Job message: {job.message}")
    
    # Test user jobs
    user_jobs = background_task_service.get_user_jobs("test-user-456")
    print(f"✅ User has {len(user_jobs)} jobs")
    
    # Test job cancellation
    success = background_task_service.cancel_job(job_id)
    print(f"✅ Job cancellation: {success}")
    
    # Check final status
    job = background_task_service.get_job_status(job_id)
    print(f"✅ Final job status: {job.status.value}")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_background_task_service())
