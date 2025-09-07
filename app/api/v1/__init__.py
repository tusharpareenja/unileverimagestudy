from fastapi import APIRouter
from app.api.v1.user import router as user_router

api_router = APIRouter()

# Include user routes
api_router.include_router(user_router, prefix="/auth", tags=["authentication"])
