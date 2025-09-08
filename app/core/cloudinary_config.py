from __future__ import annotations

import os
import cloudinary
from typing import Optional
from app.core.config import settings


def init_cloudinary() -> None:
    """Initialize Cloudinary once at startup.

    Reads either CLOUDINARY_URL or separate CLOUDINARY_CLOUD_NAME/API_KEY/API_SECRET
    and sets secure delivery.
    """
    # Prefer explicit CLOUDINARY_URL from env if present
    url = os.getenv("CLOUDINARY_URL")
    if url:
        cloudinary.config(cloudinary_url=url, secure=True)
        return

    # Fall back to values from Settings (loaded from .env) or env vars
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME") or (settings.CLOUDINARY_CLOUD_NAME or None)
    api_key = os.getenv("CLOUDINARY_API_KEY") or (settings.CLOUDINARY_API_KEY or None)
    api_secret = os.getenv("CLOUDINARY_API_SECRET") or (settings.CLOUDINARY_API_SECRET or None)

    if cloud_name and api_key and api_secret:
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
        )
    # else: leave unconfigured; uploads will fail explicitly



