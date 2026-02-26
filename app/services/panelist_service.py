from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from fastapi import HTTPException
from app.models.panelist_model import Panelist
from app.schemas.panelist_schema import PanelistCreate


def _extract_domain(email: str) -> str:
    """Extract company domain from email (e.g. user@unilever.com -> unilever.com)."""
    if not email or "@" not in email:
        return ""
    return email.strip().split("@")[-1].lower()


class PanelistService:
    def get_by_creator(self, db: Session, creator_email: str, number: int = 10) -> List[Panelist]:
        """Get panelists for the same company domain as the given email."""
        domain = _extract_domain(creator_email)
        return db.query(Panelist).filter(Panelist.creator_domain == domain).limit(number).all()

    def check_duplicate(self, db: Session, panelist_id: str, creator_email: str) -> bool:
        """Check if panelist ID already exists in this creator's domain."""
        domain = _extract_domain(creator_email)
        existing = db.query(Panelist).filter(
            Panelist.id == panelist_id,
            Panelist.creator_domain == domain
        ).first()
        return existing is not None

    def create(self, db: Session, panelist_in: PanelistCreate) -> Panelist:
        domain = _extract_domain(panelist_in.creator_email)
        if self.check_duplicate(db, panelist_in.id, panelist_in.creator_email):
            raise HTTPException(
                status_code=400,
                detail=f"Panelist ID '{panelist_in.id}' already exists for this domain. Please use a different ID."
            )

        db_obj = Panelist(
            id=panelist_in.id,
            age=panelist_in.age,
            gender=panelist_in.gender,
            creator_email=panelist_in.creator_email,
            creator_domain=domain,
        )

        try:
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
        except IntegrityError as e:
            db.rollback()
            if "uq_panelist_domain_id" in str(e.orig):
                raise HTTPException(
                    status_code=400,
                    detail=f"Panelist ID '{panelist_in.id}' already exists for this domain. Please use a different ID."
                )
            raise HTTPException(status_code=500, detail="Failed to create panelist")

    def search(self, db: Session, query: str, creator_email: str, number: int = 10) -> List[Panelist]:
        """Search panelists by ID within the same company domain."""
        domain = _extract_domain(creator_email)
        return db.query(Panelist).filter(
            Panelist.creator_domain == domain,
            Panelist.id.ilike(f"%{query}%")
        ).limit(number).all()


panelist_service = PanelistService()
