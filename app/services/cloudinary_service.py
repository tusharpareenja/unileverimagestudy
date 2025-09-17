from __future__ import annotations

import base64
import re
from typing import Tuple
from uuid import uuid4
from azure.storage.blob import BlobServiceClient, ContentSettings
from app.core.config import settings

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB


def _validate_filename(filename: str) -> None:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type")


def _get_blob_client(container: str | None = None) -> BlobServiceClient:
    conn = settings.AZURE_STORAGE_CONNECTION_STRING
    if not conn:
        # Fallback to account name/key if provided
        if settings.AZURE_STORAGE_ACCOUNT_NAME and settings.AZURE_STORAGE_ACCOUNT_KEY:
            endpoint = f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
            return BlobServiceClient(account_url=endpoint, credential=settings.AZURE_STORAGE_ACCOUNT_KEY)
        raise ValueError("Azure storage credentials not configured")
    return BlobServiceClient.from_connection_string(conn)


def _account_name_from_env() -> str | None:
    if settings.AZURE_STORAGE_ACCOUNT_NAME:
        return settings.AZURE_STORAGE_ACCOUNT_NAME
    conn = settings.AZURE_STORAGE_CONNECTION_STRING or ""
    try:
        parts = {p.split('=', 1)[0].strip(): p.split('=', 1)[1].strip() for p in conn.split(';') if '=' in p}
        return parts.get('AccountName')
    except Exception:
        return None


def _public_url(container: str, blob_name: str) -> str:
    base = settings.AZURE_STORAGE_PUBLIC_BASE_URL
    if base:
        url = f"{base.rstrip('/')}/{container}/{blob_name}"
        if settings.AZURE_STORAGE_SAS_TOKEN:
            token = settings.AZURE_STORAGE_SAS_TOKEN.lstrip('?')
            url = f"{url}?{token}"
        return url
    account = _account_name_from_env()
    if account:
        url = f"https://{account}.blob.core.windows.net/{container}/{blob_name}"
        if settings.AZURE_STORAGE_SAS_TOKEN:
            token = settings.AZURE_STORAGE_SAS_TOKEN.lstrip('?')
            url = f"{url}?{token}"
        return url
    # As a last resort, rely on default endpoint of connection string account
    url = f"https://{container}.blob.core.windows.net/{blob_name}"
    if settings.AZURE_STORAGE_SAS_TOKEN:
        token = settings.AZURE_STORAGE_SAS_TOKEN.lstrip('?')
        url = f"{url}?{token}"
    return url


def upload_file(file, folder: str = "studies") -> Tuple[str, str]:
    """Upload a file-like object to Azure Blob. Returns (secure_url, public_id)."""
    filename = getattr(file, "filename", None)
    if filename:
        _validate_filename(filename)
    stream = getattr(file, "file", None) or file
    container = settings.AZURE_STORAGE_CONTAINER or folder
    blob_service = _get_blob_client()
    blob_name = f"{folder}/{uuid4().hex}_{filename or 'upload'}"
    content_settings = ContentSettings(content_type="image/png")
    try:
        blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
        blob_client.upload_blob(stream, overwrite=True, content_settings=content_settings)
    except Exception as e:
        raise ValueError("Upload failed") from e
    url = _public_url(container, blob_name)
    # Use blob_name as public_id equivalent
    return url, blob_name


_data_url_re = re.compile(r"^data:(?P<mime>image\/(?:png|jpeg|jpg|webp));base64,(?P<data>.+)$")


def upload_base64(data_url: str, folder: str = "studies") -> Tuple[str, str]:
    """Upload a base64 data URL to Azure Blob. Returns (secure_url, public_id)."""
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
    container = settings.AZURE_STORAGE_CONTAINER or folder
    blob_service = _get_blob_client()
    blob_name = f"{folder}/{uuid4().hex}.png"
    content_settings = ContentSettings(content_type=m.group("mime"))
    try:
        blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
        blob_client.upload_blob(raw, overwrite=True, content_settings=content_settings)
    except Exception as e:
        raise ValueError("Upload failed") from e
    url = _public_url(container, blob_name)
    return url, blob_name


def delete_public_id(public_id: str) -> bool:
    if not public_id:
        return True
    try:
        container = settings.AZURE_STORAGE_CONTAINER or "studies"
        blob_service = _get_blob_client()
        blob_client = blob_service.get_blob_client(container=container, blob=public_id)
        blob_client.delete_blob()
        return True
    except Exception:
        return False



