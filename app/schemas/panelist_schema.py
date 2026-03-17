from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import re

class PanelistBase(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None

class PanelistCreate(PanelistBase):
    id: str = Field(..., max_length=50, description="Up to 50-character alphanumeric panelist ID")
    creator_email: str
    
    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Validate that ID is up to 50 alphanumeric characters"""
        if not v:
            raise ValueError("Panelist ID is required")
        if len(v) > 50:
            raise ValueError("Panelist ID must be at most 50 characters")
        if not re.match(r'^[A-Za-z0-9]{1,50}$', v):
            raise ValueError("Panelist ID must contain only letters and numbers (1-50 characters)")
        # Convert to uppercase for consistency
        return v.upper()

class PanelistResponse(PanelistBase):
    id: str
    creator_email: str
    creator_domain: Optional[str] = None

    class Config:
        from_attributes = True

class PanelistList(BaseModel):
    total: int
    items: List[PanelistResponse]
