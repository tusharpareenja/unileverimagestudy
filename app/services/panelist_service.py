from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional
from app.models.panelist_model import Panelist
from app.schemas.panelist_schema import PanelistCreate

class PanelistService:
    def get_by_creator(self, db: Session, creator_email: str, number: int = 10) -> List[Panelist]:
        return db.query(Panelist).filter(Panelist.creator_email == creator_email).limit(number).all()

    def create(self, db: Session, panelist_in: PanelistCreate) -> Panelist:
        db_obj = Panelist(
            name=panelist_in.name,
            age=panelist_in.age,
            gender=panelist_in.gender,
            creator_email=panelist_in.creator_email
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def search(self, db: Session, query: str, creator_email: str, number: int = 10) -> List[Panelist]:
        # Search by id or name (case insensitive)
        return db.query(Panelist).filter(
            Panelist.creator_email == creator_email,
            or_(
                Panelist.id.ilike(f"%{query}%"),
                Panelist.name.ilike(f"%{query}%")
            )
        ).limit(number).all()

panelist_service = PanelistService()
