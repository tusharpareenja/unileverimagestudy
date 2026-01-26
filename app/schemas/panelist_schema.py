from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import re

class PanelistBase(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None

class PanelistCreate(PanelistBase):
    id: str = Field(..., min_length=8, max_length=8, description="8-character alphanumeric panelist ID")
    creator_email: str
    
    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Validate that ID is exactly 8 alphanumeric characters"""
        if not v:
            raise ValueError("Panelist ID is required")
        if len(v) != 8:
            raise ValueError("Panelist ID must be exactly 8 characters")
        if not re.match(r'^[A-Za-z0-9]{8}$', v):
            raise ValueError("Panelist ID must contain only letters and numbers")
        # Convert to uppercase for consistency
        return v.upper()

class PanelistResponse(PanelistBase):
    id: str
    creator_email: str

    class Config:
        from_attributes = True

class PanelistList(BaseModel):
    total: int
    items: List[PanelistResponse]
