import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging

from app.api.v1 import api_router
from app.api.v1.study import router as study_router
from app.api.v1.response import router as response_router
from app.api.v1.uploads import router as uploads_router
from app.api.v1.panelist import router as panelist_router
from app.api.v1.project import router as project_router
from app.core.config import settings
from app.core.cloudinary_config import init_cloudinary

app = FastAPI(
    title="mindsurve API",
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
app.include_router(panelist_router, prefix="/api/v1/panelist", tags=["panelist"])
app.include_router(project_router, prefix="/api/v1/projects", tags=["projects"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger = logging.getLogger("uvicorn.error")
    errors = exc.errors()
    logger.error(f"422 Validation Error: {errors}")
    return JSONResponse(
        status_code=422,
        content={"detail": errors},
    )


@app.get("/")
async def root():
    return {"message": "mindsurve API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.on_event("startup")
async def on_startup():
    # Ensure progress and info logs (e.g. synthetic respondent job) appear in terminal
    # logging.basicConfig(level=logging.INFO)
    # app_logger = logging.getLogger("app")
    # app_logger.setLevel(logging.INFO)
    # if not app_logger.handlers:
    #     h = logging.StreamHandler()
    #     h.setLevel(logging.INFO)
    #     app_logger.addHandler(h)
    print("Starting application startup...")
    # Initialize Cloudinary once
    init_cloudinary()
    print("Cloudinary initialized")
    
    # Start background tasks
    from app.services.background_tasks import background_task_service
    background_task_service.task = asyncio.create_task(
        background_task_service.start_abandonment_checker(interval_minutes=15)
    )
    print("done background tadsk")
    # Start task generation cleanup service
    from app.services.background_task_service import background_task_service as task_service
    task_service._cleanup_task = asyncio.create_task(task_service._cleanup_old_jobs())
    print("done background cleanup")
    print("Startup function completed successfully!")

@app.on_event("shutdown")
async def on_shutdown():
    # Stop background tasks
    from app.services.background_tasks import background_task_service
    background_task_service.stop_abandonment_checker()  