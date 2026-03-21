# Panelist participation check – frontend integration

Use this **only in the main participate flow** (not in preview participate). When the study requires panelist selection (special creator / special domain, e.g. Unilever), run the validation on the **panelist page** when the user **chooses a panelist ID**. If the panelist has already responded, show the message and block; otherwise allow continuing.

---

## 1. When to show the panelist step and run validation

- **Main participate only:** Use the study payload from the **main** participate entry (e.g. `GET /api/v1/studies/public/{study_id}`). Do **not** use the preview endpoint for this.
- **Panelist required:** If the response includes `require_panelist_selection === true`, show the panelist selection step and run the validation below.
- **Preview:** For preview participate, either do not show the panelist step or do not call the check API. The preview study endpoint returns `require_panelist_selection: false`.

Example: after loading the study for main participate:

```js
// Main participate: GET /api/v1/studies/public/{studyId}
const study = await fetch(`/api/v1/studies/public/${studyId}`).then(r => r.json());

if (study.error) {
  // Handle not found / paused / completed
  return;
}

const showPanelistStep = study.require_panelist_selection === true;
```

---

## 2. On the panelist page – when the user selects a panelist ID

When the user **selects or changes** the panelist ID (and you are in **main** participate with `require_panelist_selection === true`):

1. Call the check API **before** allowing “Next” / submit.
2. If the API says the panelist has already participated, show the message and **block** going to the next page.
3. If the API says they have not participated, allow the user to proceed (and then submit panelist + continue as you already do).

**API**

- **Method/URL:** `GET /api/v1/responses/check-panelist-participation`
- **Query params:**
  - `study_id` – study UUID (same as in your participate URL)
  - `panelist_id` – selected panelist ID string

**Response**

- `{ "ok": true, "participated": false }` → panelist has **not** responded → allow next page.
- `{ "ok": true, "participated": true, "message": "This panelist has already responded to this study." }` → show `message` and block.

**Example (fetch)**

```js
async function checkPanelistParticipation(studyId, panelistId) {
  const params = new URLSearchParams({
    study_id: studyId,
    panelist_id: panelistId.trim(),
  });
  const res = await fetch(
    `/api/v1/responses/check-panelist-participation?${params}`,
    { method: 'GET' }
  );
  if (!res.ok) throw new Error('Check failed');
  return res.json();
}
```

**Example (usage on panelist page – main participate only)**

```js
// Only when require_panelist_selection is true (main participate, special domain)
async function onPanelistSelected(studyId, panelistId) {
  if (!panelistId?.trim()) {
    setError('Please select a panelist.');
    setCanProceed(false);
    return;
  }
  setChecking(true);
  setError(null);
  setCanProceed(false);
  try {
    const data = await checkPanelistParticipation(studyId, panelistId);
    if (data.participated) {
      setError(data.message || 'This panelist has already responded to this study.');
      setCanProceed(false);
    } else {
      setError(null);
      setCanProceed(true);
    }
  } catch (e) {
    setError('Could not verify panelist. Please try again.');
    setCanProceed(false);
  } finally {
    setChecking(false);
  }
}
```

- Disable “Next” until `canProceed` is true (and optionally show a loading state while `checking`).
- Call `onPanelistSelected(studyId, panelistId)` when the user picks a panelist (e.g. on select change or on blur).
- **Do not** call this in preview participate; only when you loaded the study from the main public endpoint and `require_panelist_selection === true`.

---

## 3. Summary

| Context              | Show panelist step? | Call check API on panelist select? |
|----------------------|---------------------|-------------------------------------|
| Main participate     | Yes if `require_panelist_selection === true` | Yes (and block next if already participated) |
| Preview participate  | No (or show without validation) | No |

- **Backend:** Main participate uses `GET /api/v1/studies/public/{study_id}` and returns `require_panelist_selection: true` for special domain (e.g. Unilever). Preview uses the preview endpoint and returns `require_panelist_selection: false`.
- **Frontend:** In main participate, when `require_panelist_selection` is true, on the panelist page run the validation when the user chooses a panelist ID; if `participated === true`, show the message and do not allow next page.
