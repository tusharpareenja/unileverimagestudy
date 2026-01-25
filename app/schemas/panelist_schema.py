from pydantic import BaseModel
from typing import Optional, List

class PanelistBase(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None

class PanelistCreate(PanelistBase):
    creator_email: str

class PanelistResponse(PanelistBase):
    id: str
    creator_email: str

    class Config:
        from_attributes = True

class PanelistList(BaseModel):
    total: int
    items: List[PanelistResponse]
