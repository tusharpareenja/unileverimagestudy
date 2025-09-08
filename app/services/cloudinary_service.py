from __future__ import annotations

import base64
import re
from typing import Tuple

import cloudinary.uploader

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB


def _validate_filename(filename: str) -> None:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type")


def upload_file(file, folder: str = "studies") -> Tuple[str, str]:
    """Upload a file-like object to Cloudinary. Returns (secure_url, public_id)."""
    # file: starlette UploadFile.file or any file-like object
    # Basic validation when the file object has a filename attribute
    filename = getattr(file, "filename", None)
    if filename:
        _validate_filename(filename)
    # Cloudinary handles file-like objects; for FastAPI UploadFile use underlying .file
    stream = getattr(file, "file", None) or file
    result = cloudinary.uploader.upload(stream, folder=folder, resource_type="image")
    return result.get("secure_url"), result.get("public_id")


_data_url_re = re.compile(r"^data:(?P<mime>image\/(?:png|jpeg|jpg|webp));base64,(?P<data>.+)$")


def upload_base64(data_url: str, folder: str = "studies") -> Tuple[str, str]:
    """Upload a base64 data URL to Cloudinary. Returns (secure_url, public_id)."""
    m = _data_url_re.match(data_url or "")
    if not m:
        raise ValueError("Invalid data URL")
    b64 = m.group("data")
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        raise ValueError("Invalid base64 payload")
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("File too large")
    # Cloudinary can accept the full data URL directly
    result = cloudinary.uploader.upload(data_url, folder=folder, resource_type="image")
    return result.get("secure_url"), result.get("public_id")


def delete_public_id(public_id: str) -> bool:
    if not public_id:
        return True
    try:
        cloudinary.uploader.destroy(public_id, invalidate=True, resource_type="image")
        return True
    except Exception:
        return False



