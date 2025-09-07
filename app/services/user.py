from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import Optional, List
from datetime import datetime
import uuid

from app.models.user_model import User
from app.schemas.user_schema import UserRegister, UserUpdate, PasswordChange, UserResponse
from app.core.security import get_password_hash, verify_password, create_token_pair, refresh_access_token


class UserService:
    def __init__(self, db: Session):
        self.db = db

    def authenticate_user(self, username_or_email: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password"""
        # Check if input is email or username
        if '@' in username_or_email:
            user = self.db.query(User).filter(
                User.email == username_or_email.lower(),
                User.is_active == True
            ).first()
        else:
            user = self.db.query(User).filter(
                User.username == username_or_email.lower(),
                User.is_active == True
            ).first()
        
        if not user:
            return None
        
        if not verify_password(password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        return user

    def authenticate_user_with_tokens(self, username_or_email: str, password: str) -> Optional[dict]:
        """Authenticate user and return user data with JWT tokens"""
        user = self.authenticate_user(username_or_email, password)
        if not user:
            return None
        
        # Create JWT tokens
        tokens = create_token_pair(user.id, user.username)
        
        # Convert user to response schema
        user_response = UserResponse.from_orm(user)
        
        return {
            "user": user_response,
            "tokens": tokens
        }

    def create_user(self, user_data: UserRegister) -> User:
        """Create a new user"""
        # Check if username or email already exists
        existing_user = self.db.query(User).filter(
            or_(User.username == user_data.username.lower(), 
                User.email == user_data.email.lower())
        ).first()
        
        if existing_user:
            if existing_user.username == user_data.username.lower():
                raise ValueError("Username already exists")
            else:
                raise ValueError("Email already exists")
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        
        db_user = User(
            username=user_data.username.lower(),
            email=user_data.email.lower(),
            name=user_data.name,
            password_hash=hashed_password,
            phone=user_data.phone,
            date_of_birth=user_data.date_of_birth,
            is_active=True,
            is_verified=False
        )
        
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        
        return db_user

    def create_user_with_tokens(self, user_data: UserRegister) -> dict:
        """Create a new user and return user data with JWT tokens"""
        user = self.create_user(user_data)
        
        # Create JWT tokens
        tokens = create_token_pair(user.id, user.username)
        
        # Convert user to response schema
        user_response = UserResponse.from_orm(user)
        
        return {
            "user": user_response,
            "tokens": tokens
        }

    def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id, User.is_active == True).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.db.query(User).filter(
            User.username == username.lower(), 
            User.is_active == True
        ).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(
            User.email == email.lower(), 
            User.is_active == True
        ).first()

    def get_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get list of users with pagination"""
        return self.db.query(User).filter(User.is_active == True).offset(skip).limit(limit).all()

    def update_user(self, user_id: uuid.UUID, user_data: UserUpdate) -> Optional[User]:
        """Update user information"""
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        # Update only provided fields
        update_data = user_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        
        return user

    def change_password(self, user_id: uuid.UUID, password_data: PasswordChange) -> bool:
        """Change user password"""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        # Verify current password
        if not verify_password(password_data.current_password, user.password_hash):
            return False
        
        # Update password
        user.password_hash = get_password_hash(password_data.new_password)
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        return True

    def deactivate_user(self, user_id: uuid.UUID) -> bool:
        """Deactivate user (soft delete)"""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_active = False
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        return True

    def verify_user(self, user_id: uuid.UUID) -> bool:
        """Mark user as verified"""
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        user.is_verified = True
        user.updated_at = datetime.utcnow()
        self.db.commit()
        
        return True

    def check_username_available(self, username: str) -> bool:
        """Check if username is available"""
        user = self.db.query(User).filter(User.username == username.lower()).first()
        return user is None

    def check_email_available(self, email: str) -> bool:
        """Check if email is available"""
        user = self.db.query(User).filter(User.email == email.lower()).first()
        return user is None

    def search_users(self, query: str, skip: int = 0, limit: int = 50) -> List[User]:
        """Search users by name, username, or email"""
        search_filter = f"%{query.lower()}%"
        return self.db.query(User).filter(
            User.is_active == True,
            or_(
                User.name.ilike(search_filter),
                User.username.ilike(search_filter),
                User.email.ilike(search_filter)
            )
        ).offset(skip).limit(limit).all()

    def refresh_user_tokens(self, refresh_token: str) -> Optional[dict]:
        """Refresh access token using refresh token"""
        new_tokens = refresh_access_token(refresh_token)
        if not new_tokens:
            return None
        
        return new_tokens


# Convenience functions for direct use
def authenticate_user(db: Session, username_or_email: str, password: str) -> Optional[User]:
    """Authenticate user - convenience function"""
    service = UserService(db)
    return service.authenticate_user(username_or_email, password)


def create_user(db: Session, user_data: UserRegister) -> User:
    """Create user - convenience function"""
    service = UserService(db)
    return service.create_user(user_data)


def get_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    """Get user by ID - convenience function"""
    service = UserService(db)
    return service.get_user_by_id(user_id)


def authenticate_user_with_tokens(db: Session, username_or_email: str, password: str) -> Optional[dict]:
    """Authenticate user with tokens - convenience function"""
    service = UserService(db)
    return service.authenticate_user_with_tokens(username_or_email, password)


def create_user_with_tokens(db: Session, user_data: UserRegister) -> dict:
    """Create user with tokens - convenience function"""
    service = UserService(db)
    return service.create_user_with_tokens(user_data)


def refresh_user_tokens(db: Session, refresh_token: str) -> Optional[dict]:
    """Refresh user tokens - convenience function"""
    service = UserService(db)
    return service.refresh_user_tokens(refresh_token)
