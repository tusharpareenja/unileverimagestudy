from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Index, UniqueConstraint
from sqlalchemy.sql import func
from app.db.base import Base

class Panelist(Base):
    __tablename__ = "panelists"

    internal_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String(8), nullable=False, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(50), nullable=True)
    # Creator email to link with the user who created this panelist
    creator_email = Column(String(255), ForeignKey("users.email"), nullable=False, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_panelist_creator", "creator_email"),
        # Ensure ID is unique per creator (allows same ID across different creators)
        UniqueConstraint("creator_email", "id", name="uq_panelist_creator_id"),
    )

    def __repr__(self):
        return f"<Panelist(id={self.id}, creator='{self.creator_email}')>"
