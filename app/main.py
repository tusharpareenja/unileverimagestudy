from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.api.v1.study import router as study_router
from app.api.v1.response import router as response_router
from app.api.v1.uploads import router as uploads_router
from app.core.config import settings
from app.core.cloudinary_config import init_cloudinary

app = FastAPI(
    title="Unilever Image Study API",
    description="API for user authentication and management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")
app.include_router(study_router, prefix="/api/v1/studies", tags=["studies"])
app.include_router(response_router, prefix="/api/v1/responses", tags=["responses"])
app.include_router(uploads_router, prefix="/api/v1/uploads", tags=["uploads"])


@app.get("/")
async def root():
    return {"message": "Unilever Image Study API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.on_event("startup")
async def on_startup():
    # Initialize Cloudinary once
    init_cloudinary()
