"""Shared helpers for API route handlers."""

import logging
from typing import NoReturn

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


def handle_hf_error(e: Exception, default_status: int = status.HTTP_400_BAD_REQUEST) -> NoReturn:
    """Raise the appropriate HTTPException for HuggingFace errors.

    Gated/auth errors → 403.  Other HF fetch errors → default_status (400 for
    capacity-planner, 500 for gpu-recommender).
    """
    msg = str(e).lower()
    if "gated" in msg or "403" in msg or "unauthorized" in msg:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Model is gated. Set HF_TOKEN on the backend: {e}",
        )
    raise HTTPException(
        status_code=default_status,
        detail=f"Could not fetch model from HuggingFace: {e}",
    )
