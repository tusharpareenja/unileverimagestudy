from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
import asyncio
from typing import List
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_user
from app.db.session import get_db
from app.models.user_model import User
from app.services.cloudinary_service import upload_file, upload_base64, delete_public_id

router = APIRouter()


@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    try:
        secure_url, public_id = upload_file(file)
        return {"secure_url": secure_url, "public_id": public_id}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Upload failed")


@router.post("/image-base64")
async def upload_image_base64(
    payload: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    data_url = payload.get("data_url")
    if not data_url:
        raise HTTPException(status_code=400, detail="data_url is required")
    try:
        secure_url, public_id = upload_base64(data_url)
        return {"secure_url": secure_url, "public_id": public_id}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Upload failed")


@router.post("/images")
async def upload_images(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    # Run uploads concurrently to reduce total wall time
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, upload_file, f) for f in (files or [])]

    results = []
    errors = []
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    for idx, outcome in enumerate(completed):
        if isinstance(outcome, Exception):
            msg = str(outcome) if isinstance(outcome, ValueError) else "Upload failed"
            errors.append({"index": idx, "error": msg})
        else:
            secure_url, public_id = outcome
            results.append({"secure_url": secure_url, "public_id": public_id, "index": idx})

    return {"results": results, "errors": errors}


@router.delete("")
async def delete_asset(
    public_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    ok = delete_public_id(public_id)
    return {"success": ok}



