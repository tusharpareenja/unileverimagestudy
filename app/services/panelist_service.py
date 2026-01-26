from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from fastapi import HTTPException
from app.models.panelist_model import Panelist
from app.schemas.panelist_schema import PanelistCreate

class PanelistService:
    def get_by_creator(self, db: Session, creator_email: str, number: int = 10) -> List[Panelist]:
        return db.query(Panelist).filter(Panelist.creator_email == creator_email).limit(number).all()
    
    def check_duplicate(self, db: Session, panelist_id: str, creator_email: str) -> bool:
        """Check if panelist ID already exists for this creator"""
        existing = db.query(Panelist).filter(
            Panelist.id == panelist_id,
            Panelist.creator_email == creator_email
        ).first()
        return existing is not None

    def create(self, db: Session, panelist_in: PanelistCreate) -> Panelist:
        # Check for duplicate ID within the same creator's scope
        if self.check_duplicate(db, panelist_in.id, panelist_in.creator_email):
            raise HTTPException(
                status_code=400,
                detail=f"Panelist ID '{panelist_in.id}' already exists for this creator. Please use a different ID."
            )
        
        db_obj = Panelist(
            id=panelist_in.id,
            age=panelist_in.age,
            gender=panelist_in.gender,
            creator_email=panelist_in.creator_email
        )
        
        try:
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            db.rollback()
            # Handle race condition where duplicate was created between check and insert
            if "uq_panelist_creator_id" in str(e.orig):
                raise HTTPException(
                    status_code=400,
                    detail=f"Panelist ID '{panelist_in.id}' already exists for this creator. Please use a different ID."
                )
            raise HTTPException(status_code=500, detail="Failed to create panelist")

    def search(self, db: Session, query: str, creator_email: str, number: int = 10) -> List[Panelist]:
        # Search by id only (case insensitive)
        return db.query(Panelist).filter(
            Panelist.creator_email == creator_email,
            Panelist.id.ilike(f"%{query}%")
        ).limit(number).all()

panelist_service = PanelistService()
