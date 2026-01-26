from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.schemas.panelist_schema import PanelistResponse, PanelistCreate
from app.services.panelist_service import panelist_service

router = APIRouter()

@router.get("/", response_model=List[PanelistResponse])
def get_panelists(
    creator_email: str = Query(..., description="Creator email to filter panelists"),
    number: int = Query(10, description="Number of panelists to fetch"),
    db: Session = Depends(get_db)
):
    """
    Get all panelists for a specific creator.
    """
    try:
        return panelist_service.get_by_creator(db=db, creator_email=creator_email, number=number)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=PanelistResponse)
def create_panelist(
    panelist_in: PanelistCreate,
    db: Session = Depends(get_db)
):
    """
    Add a new panelist.
    """
    try:
        return panelist_service.create(db=db, panelist_in=panelist_in)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search", response_model=List[PanelistResponse])
def search_panelists(
    creator_email: str = Query(..., description="Creator email to search within"),
    query: str = Query(..., description="Search by ID"),
    number: int = Query(10, description="Limit results"),
    db: Session = Depends(get_db)
):
    """
    Search panelists by ID.
    """
    try:
        return panelist_service.search(db=db, query=query, creator_email=creator_email, number=number)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
