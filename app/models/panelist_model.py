from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Index, UniqueConstraint
from sqlalchemy.sql import func
from app.db.base import Base

class Panelist(Base):
    __tablename__ = "panelists"

    internal_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String(50), nullable=False, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(50), nullable=True)
    # Creator email (audit; who created this panelist)
    creator_email = Column(String(255), ForeignKey("users.email"), nullable=False, index=True)
    # Company domain from creator email (e.g. unilever.com); panelists are shared by all users with same domain
    creator_domain = Column(String(255), nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_panelist_creator", "creator_email"),
        Index("idx_panelist_creator_domain", "creator_domain"),
        # ID is unique per domain (shared pool for all users in that domain)
        UniqueConstraint("creator_domain", "id", name="uq_panelist_domain_id"),
    )

    def __repr__(self):
        return f"<Panelist(id={self.id}, domain='{self.creator_domain}')>"
