from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime
import uuid


class UserLogin(BaseModel):
    """Schema for user login - accepts username or email"""
    username_or_email: str = Field(..., min_length=3, description="Username or email address")
    password: str = Field(..., min_length=6, description="User password")

    @validator('username_or_email')
    def validate_username_or_email(cls, v):
        if '@' in v:
            # If contains @, validate as email
            if len(v) < 5:
                raise ValueError('Email must be at least 5 characters long')
        else:
            # If no @, validate as username
            if len(v) < 3:
                raise ValueError('Username must be at least 3 characters long')
        return v.lower().strip()


class UserRegister(BaseModel):
    """Schema for user registration"""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    name: str = Field(..., min_length=2, max_length=100, description="Full name")
    password: str = Field(..., min_length=6, description="User password")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")
    date_of_birth: Optional[datetime] = Field(None, description="Date of birth")

    @validator('username')
    def validate_username(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.lower().strip()

    @validator('name')
    def validate_name(cls, v):
        return v.strip().title()


class UserResponse(BaseModel):
    """Schema for user response (without sensitive data)"""
    id: uuid.UUID
    username: str
    email: str
    name: str
    phone: Optional[str]
    date_of_birth: Optional[datetime]
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for updating user information"""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[datetime] = None

    @validator('name')
    def validate_name(cls, v):
        if v is not None:
            return v.strip().title()
        return v


class PasswordChange(BaseModel):
    """Schema for password change"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=6, description="New password")

    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Schema for token refresh request"""
    refresh_token: str


class TokenRefreshResponse(BaseModel):
    """Schema for token refresh response"""
    access_token: str
    token_type: str = "bearer"


class UserLoginResponse(BaseModel):
    """Schema for successful login response"""
    user: UserResponse
    tokens: Token


# Rebuild models to resolve forward references
def rebuild_models():
    """Rebuild all models to resolve forward references"""
    # Get all model classes defined in this module
    import sys
    current_module = sys.modules[__name__]
    
    for name in dir(current_module):
        obj = getattr(current_module, name)
        if (isinstance(obj, type) and 
            issubclass(obj, BaseModel) and 
            obj != BaseModel and
            obj.__module__ == __name__):
            try:
                obj.model_rebuild()
            except Exception as e:
                print(f"Warning: Could not rebuild {name}: {e}")

# Call rebuild to resolve forward references
rebuild_models()
