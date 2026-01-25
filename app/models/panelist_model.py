from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Index
from sqlalchemy.sql import func
from app.db.base import Base
import random
import string

def generate_panelist_id():
    """Generate a unique 10-character alphanumeric ID"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

class Panelist(Base):
    __tablename__ = "panelists"

    id = Column(String(10), primary_key=True, default=generate_panelist_id)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(50), nullable=True)
    # Creator email to link with the user who created this panelist
    creator_email = Column(String(255), ForeignKey("users.email"), nullable=False, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_panelist_creator", "creator_email"),
        Index("idx_panelist_search", "id", "name"),
    )

    def __repr__(self):
        return f"<Panelist(id={self.id}, name='{self.name}', creator='{self.creator_email}')>"
