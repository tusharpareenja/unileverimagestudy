from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    
    # JWT Settings
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days in minutes
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Cloudinary Settings
    CLOUDINARY_CLOUD_NAME: str | None = None
    CLOUDINARY_API_KEY: str | None = None
    CLOUDINARY_API_SECRET: str | None = None
    CLOUDINARY_FOLDER: str = "unilever_image_study"
    
    # Application Settings
    BASE_URL: str = "http://localhost:3000"  # Base URL for share links in frontend

    

    class Config:
        env_file = ".env"

settings = Settings()

