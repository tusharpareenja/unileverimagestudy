# Frontend Integration Guide for Async Task Generation

## Problem Solved
Your frontend was losing the timer and preview when calling `/generate-tasks` for large studies because the endpoint now returns immediately with a `job_id` instead of full tasks.

## Solution Implemented

### 1. **Immediate Preview Tasks**
When you call `/generate-tasks` for large studies (>50 respondents), you now get:

```json
{
  "tasks": {
    "respondent_1": [...],  // Preview tasks for 1-3 respondents
    "respondent_2": [...],
    "respondent_3": [...]
  },
  "metadata": {
    "job_id": "abc123-def456",
    "status": "started",
    "is_preview": true,
    "total_respondents": 500,
    "preview_respondents": 3,
    "message": "Task generation started in background for 500 respondents"
  }
}
```

### 2. **Frontend Integration**

#### **Step 1: Handle the Response**
```javascript
// Your existing code
const response = await fetch('/api/v1/studies/generate-tasks', {
  method: 'POST',
  body: JSON.stringify(payload)
});

const result = await response.json();

// NEW: Check if it's a preview
if (result.metadata.is_preview) {
  // Store preview tasks in localStorage (your existing logic)
  localStorage.setItem('study_tasks', JSON.stringify(result.tasks));
  
  // Start polling for completion
  startPollingForCompletion(result.metadata.job_id);
  
  // Show preview with timer (your existing logic)
  showPreviewWithTimer();
} else {
  // Handle normal response (existing logic)
  localStorage.setItem('study_tasks', JSON.stringify(result.tasks));
  showPreviewWithTimer();
}
```

#### **Step 2: Poll for Completion**
```javascript
function startPollingForCompletion(jobId) {
  const pollInterval = setInterval(async () => {
    try {
      const statusResponse = await fetch(`/api/v1/studies/generate-tasks/status/${jobId}`);
      const status = await statusResponse.json();
      
      if (status.status === 'completed') {
        // Get full tasks
        const fullTasksResponse = await fetch(`/api/v1/studies/generate-tasks/result/${jobId}`);
        const fullTasks = await fullTasksResponse.json();
        
        // Update localStorage with full tasks
        localStorage.setItem('study_tasks', JSON.stringify(fullTasks.tasks));
        
        // Update UI to show full tasks
        updatePreviewWithFullTasks(fullTasks.tasks);
        
        clearInterval(pollInterval);
      } else if (status.status === 'failed') {
        // Handle error
        showError(status.error);
        clearInterval(pollInterval);
      } else {
        // Update progress
        updateProgress(status.progress, status.message);
      }
    } catch (error) {
      console.error('Polling error:', error);
    }
  }, 2000); // Poll every 2 seconds
}
```

#### **Step 3: Update Progress Display**
```javascript
function updateProgress(progress, message) {
  // Update your progress indicator
  document.getElementById('progress-bar').style.width = `${progress}%`;
  document.getElementById('progress-message').textContent = message;
}

function updatePreviewWithFullTasks(fullTasks) {
  // Replace preview tasks with full tasks
  localStorage.setItem('study_tasks', JSON.stringify(fullTasks));
  
  // Update your timer/preview logic
  showPreviewWithTimer();
  
  // Hide progress indicator
  document.getElementById('progress-indicator').style.display = 'none';
}
```

### 3. **New API Endpoints**

#### **Check Job Status**
```javascript
GET /api/v1/studies/generate-tasks/status/{job_id}
```
Returns:
```json
{
  "job_id": "abc123",
  "study_id": "study-uuid",
  "status": "processing",  // pending, started, processing, completed, failed
  "progress": 45.0,
  "message": "Generating tasks...",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": null,
  "error": null
}
```

#### **Get Full Tasks (when completed)**
```javascript
GET /api/v1/studies/generate-tasks/result/{job_id}
```
Returns:
```json
{
  "job_id": "abc123",
  "study_id": "study-uuid",
  "status": "completed",
  "tasks": {
    "respondent_1": [...],  // Full tasks for all respondents
    "respondent_2": [...],
    // ... all 500 respondents
  },
  "metadata": {
    "total_respondents": 500,
    "completed_at": "2024-01-15T10:45:00Z",
    "message": "Task generation completed successfully"
  }
}
```

#### **Cancel Job (if needed)**
```javascript
POST /api/v1/studies/generate-tasks/cancel/{job_id}
```

#### **List All Jobs**
```javascript
GET /api/v1/studies/generate-tasks/jobs
```

### 4. **Benefits**

✅ **Timer Never Disappears**: Preview tasks are always available immediately
✅ **Progress Tracking**: Real-time progress updates
✅ **No Timeouts**: Large studies work perfectly in Azure
✅ **Better UX**: Users see immediate feedback and progress
✅ **Cancellable**: Users can cancel long-running jobs

### 5. **Migration Steps**

1. **Update your `/generate-tasks` response handling** to check for `is_preview`
2. **Add polling logic** for background jobs
3. **Update progress display** to show job status
4. **Test with large studies** (>50 respondents)

The solution ensures your frontend works exactly the same for small studies, but now also works perfectly for large studies with progress tracking!
