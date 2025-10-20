from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices

class Settings(BaseSettings):
    # Pydantic Settings config (v2)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
    
    # Azure Blob Storage Settings (support multiple env var names via validation aliases)
    AZURE_STORAGE_CONNECTION_STRING: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_STORAGE_CONNECTION_STRING",
            "azure_storage_connection_string",
        ),
    )
    AZURE_STORAGE_ACCOUNT_NAME: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_STORAGE_ACCOUNT_NAME",
            "azure_storage_account_name",
        ),
    )
    AZURE_STORAGE_ACCOUNT_KEY: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_STORAGE_ACCOUNT_KEY",
            "azure_storage_account_key",
        ),
    )
    AZURE_STORAGE_CONTAINER: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_STORAGE_CONTAINER",
            "AZURE_STORAGE_CONTAINER_NAME",
            "azure_storage_container_name",
            "azure_storage_container",
        ),
    )
    AZURE_STORAGE_PUBLIC_BASE_URL: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_STORAGE_PUBLIC_BASE_URL",
            "azure_storage_public_base_url",
            "AZURE_STORAGE_BASE_URL",
            "azure_storage_base_url",
        ),
    )  # Optional override for CDN/frontdoor
    AZURE_STORAGE_SAS_TOKEN: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AZURE_STORAGE_SAS_TOKEN",
            "azure_storage_sas_token",
            "AZURE_BLOB_SAS_TOKEN",
            "azure_blob_sas_token",
        ),
    )  # Optional, appended to URLs for private containers
    
    # Application Settings
    BASE_URL: str = "https://mindgenome.vercel.app"  # Base URL for share links in frontend
    
    # Task Generation Settings
    TASK_GENERATION_TIMEOUT: int = 3600  # 60 minutes timeout for task generation (Azure max is ~4-5 min for sync requests)
    MAX_RESPONDENTS_FOR_SYNC: int = 0  # Use async processing for studies with more than 4 respondents
    TASK_GENERATION_CHUNK_SIZE: int = 50  # Process respondents in smaller chunks for memory efficiency
    MAX_MEMORY_USAGE_MB: int = 1024  # 1GB memory limit for task generation (matches Azure P2 plan)
    
    # SMTP Email Settings
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str | None = None
    SMTP_PASSWORD: str | None = None
    FROM_EMAIL: str 
    APP_NAME: str = "Unilever Image Study"
    FRONTEND_URL: str = "https://mindgenome.vercel.app"

settings = Settings()

