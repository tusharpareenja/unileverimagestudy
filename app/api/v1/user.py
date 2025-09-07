from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid

from app.db.session import get_db
from app.schemas.user_schema import (
    UserLogin, UserRegister, UserResponse, UserUpdate, 
    PasswordChange, Token, TokenRefresh, TokenRefreshResponse,
    UserLoginResponse
)
from app.services.user import (
    authenticate_user_with_tokens, create_user_with_tokens,
    refresh_user_tokens, UserService
)
from app.core.dependencies import get_current_user, get_current_active_user
from app.models.user_model import User

router = APIRouter()


@router.post("/register", response_model=UserLoginResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user and return user data with JWT tokens
    """
    try:
        result = create_user_with_tokens(db, user_data)
        return UserLoginResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/login", response_model=UserLoginResponse)
async def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    """
    Login user with username/email and password, return JWT tokens
    """
    result = authenticate_user_with_tokens(
        db, 
        login_data.username_or_email, 
        login_data.password
    )
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password"
        )
    
    return UserLoginResponse(**result)


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_tokens(token_data: TokenRefresh, db: Session = Depends(get_db)):
    """
    Refresh access token using refresh token
    """
    result = refresh_user_tokens(db, token_data.refresh_token)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    return TokenRefreshResponse(**result)


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: User = Depends(get_current_active_user)):
    """
    Get current user profile information
    """
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update current user profile information
    """
    service = UserService(db)
    updated_user = service.update_user(current_user.id, user_data)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(updated_user)


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change user password
    """
    service = UserService(db)
    success = service.change_password(current_user.id, password_data)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    return {"message": "Password changed successfully"}


@router.get("/users", response_model=List[UserResponse])
async def get_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get list of users (admin function)
    """
    service = UserService(db)
    users = service.get_users(skip=skip, limit=limit)
    return [UserResponse.from_orm(user) for user in users]


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user by ID
    """
    service = UserService(db)
    user = service.get_user_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(user)


@router.get("/search", response_model=List[UserResponse])
async def search_users(
    q: str,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Search users by name, username, or email
    """
    service = UserService(db)
    users = service.search_users(q, skip=skip, limit=limit)
    return [UserResponse.from_orm(user) for user in users]


@router.post("/verify/{user_id}")
async def verify_user(
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Verify a user (admin function)
    """
    service = UserService(db)
    success = service.verify_user(user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User verified successfully"}


@router.delete("/me")
async def deactivate_user(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Deactivate current user account (soft delete)
    """
    service = UserService(db)
    success = service.deactivate_user(current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "Account deactivated successfully"}


@router.get("/check-username/{username}")
async def check_username_available(
    username: str,
    db: Session = Depends(get_db)
):
    """
    Check if username is available
    """
    service = UserService(db)
    available = service.check_username_available(username)
    
    return {
        "username": username,
        "available": available
    }


@router.get("/check-email/{email}")
async def check_email_available(
    email: str,
    db: Session = Depends(get_db)
):
    """
    Check if email is available
    """
    service = UserService(db)
    available = service.check_email_available(email)
    
    return {
        "email": email,
        "available": available
    }
