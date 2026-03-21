from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core.cache import RedisCache
from app.db.session import get_db
from app.schemas.panelist_schema import PanelistResponse, PanelistCreate
from app.services.panelist_service import panelist_service

router = APIRouter()

def _panelist_list_cache_key(creator_email: str, number: int) -> str:
    """Normalize email for cache key so we can invalidate by creator."""
    normalized = creator_email.strip().lower()
    return f"panelist_list:{normalized}:{number}"

def _invalidate_panelist_list_cache(creator_email: str) -> None:
    """Clear cached panelist lists and search for this creator (e.g. after create)."""
    normalized = creator_email.strip().lower()
    RedisCache.delete_pattern(f"panelist_list:{normalized}:*")
    RedisCache.delete_pattern(f"panelist_search:{normalized}:*")

@router.get("/", response_model=List[PanelistResponse])
def get_panelists(
    creator_email: str = Query(..., description="Creator email to filter panelists"),
    number: int = Query(10, description="Number of panelists to fetch"),
    db: Session = Depends(get_db)
):
    """
    Get all panelists for a specific creator.
    """
    cache_key = _panelist_list_cache_key(creator_email, number)
    cached_data = RedisCache.get(cache_key)
    if cached_data:
        return [PanelistResponse.model_validate(d) for d in cached_data]

    try:
        result = panelist_service.get_by_creator(db=db, creator_email=creator_email, number=number)
        responses = [PanelistResponse.model_validate(p) for p in result]
        RedisCache.set(
            cache_key,
            [r.model_dump(mode="json") for r in responses],
            ttl_seconds=60,
        )
        return responses
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
        result = panelist_service.create(db=db, panelist_in=panelist_in)
        _invalidate_panelist_list_cache(panelist_in.creator_email)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _panelist_search_cache_key(creator_email: str, query: str, number: int) -> str:
    """Cache key for panelist search results."""
    normalized = creator_email.strip().lower()
    return f"panelist_search:{normalized}:{query.strip()}:{number}"

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
    cache_key = _panelist_search_cache_key(creator_email, query, number)
    cached_data = RedisCache.get(cache_key)
    if cached_data:
        return [PanelistResponse.model_validate(d) for d in cached_data]

    try:
        result = panelist_service.search(db=db, query=query, creator_email=creator_email, number=number)
        responses = [PanelistResponse.model_validate(p) for p in result]
        RedisCache.set(
            cache_key,
            [r.model_dump(mode="json") for r in responses],
            ttl_seconds=60,
        )
        return responses
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
