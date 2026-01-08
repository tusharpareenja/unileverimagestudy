from __future__ import annotations
from typing import List, Optional, Tuple
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_
from fastapi import HTTPException, status
import logging

from app.models.study_model import Study, StudyMember
from app.models.user_model import User
from app.schemas.study_schema import StudyMemberInvite, StudyMemberUpdate, StudyRole
from app.services.email_service import email_service
from app.core.config import settings

logger = logging.getLogger(__name__)

class StudyMemberService:
    def invite_member(self, db: Session, study_id: UUID, inviter: User, payload: StudyMemberInvite) -> StudyMember:
        """
        Invite a member to a study by email.
        """
        # 1. Check if study exists and inviter has permission (only Admin/Creator for now)
        study = db.get(Study, study_id)
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")
        
        if study.creator_id != inviter.id:
            # Check if inviter is an admin member
            admin_member = db.scalars(
                select(StudyMember).where(
                    StudyMember.study_id == study_id,
                    StudyMember.user_id == inviter.id,
                    StudyMember.role == 'admin'
                )
            ).first()
            if not admin_member:
                raise HTTPException(status_code=403, detail="Only admins can invite members")

        # 2. Check if already a member
        existing_member = db.scalars(
            select(StudyMember).where(
                StudyMember.study_id == study_id,
                StudyMember.invited_email == payload.email
            )
        ).first()
        if existing_member:
            raise HTTPException(status_code=400, detail="User is already a member of this study")

        # 3. Check if user exists in the system
        user = db.scalars(
            select(User).where(User.email == payload.email)
        ).first()

        # 4. Create StudyMember record
        new_member = StudyMember(
            id=uuid4(),
            study_id=study_id,
            user_id=user.id if user else None,
            role=payload.role,
            invited_email=payload.email
        )
        db.add(new_member)
        db.commit()
        db.refresh(new_member)

        # 5. Send invitation email
        try:
            if user:
                # Existing user template
                email_service.send_study_invitation(
                    to_email=payload.email,
                    user_name=user.name,
                    study_title=study.title,
                    inviter_name=inviter.name,
                    role=payload.role,
                    is_new_user=False
                )
            else:
                # New user template
                email_service.send_study_invitation(
                    to_email=payload.email,
                    user_name="there",
                    study_title=study.title,
                    inviter_name=inviter.name,
                    role=payload.role,
                    is_new_user=True
                )
        except Exception as e:
            logger.error(f"Failed to send invitation email: {e}")
            # we don't fail the request if email fails, but maybe we should?
            # for now, just log it.

        return new_member

    def list_members(self, db: Session, study_id: UUID) -> List[StudyMember]:
        stmt = select(StudyMember).where(StudyMember.study_id == study_id).order_by(StudyMember.created_at)
        return list(db.scalars(stmt).all())

    def update_member_role(self, db: Session, study_id: UUID, member_id: UUID, current_user: User, payload: StudyMemberUpdate) -> StudyMember:
        member = db.get(StudyMember, member_id)
        if not member or member.study_id != study_id:
            raise HTTPException(status_code=404, detail="Member not found")
        
        # Check permission: Only study creator or an admin member can change roles
        study = db.get(Study, study_id)
        if study.creator_id != current_user.id:
            admin_check = db.scalars(
                select(StudyMember).where(
                    StudyMember.study_id == study_id,
                    StudyMember.user_id == current_user.id,
                    StudyMember.role == 'admin'
                )
            ).first()
            if not admin_check:
                raise HTTPException(status_code=403, detail="Permission denied")

        member.role = payload.role
        db.commit()
        db.refresh(member)
        return member

    def remove_member(self, db: Session, study_id: UUID, member_id: UUID, current_user: User):
        member = db.get(StudyMember, member_id)
        if not member or member.study_id != study_id:
            raise HTTPException(status_code=404, detail="Member not found")
        
        study = db.get(Study, study_id)
        # Permission: Creator can remove anyone. Admin can remove non-creators. Users can remove themselves.
        is_self = member.user_id == current_user.id
        is_creator = study.creator_id == current_user.id
        
        if not (is_self or is_creator):
            # Check if admin
            admin_check = db.scalars(
                select(StudyMember).where(
                    StudyMember.study_id == study_id,
                    StudyMember.user_id == current_user.id,
                    StudyMember.role == 'admin'
                )
            ).first()
            if not admin_check:
                raise HTTPException(status_code=403, detail="Permission denied")

        db.delete(member)
        db.commit()

study_member_service = StudyMemberService()
