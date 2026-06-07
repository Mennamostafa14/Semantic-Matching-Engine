from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str

    OPENAI_API_KEY:str
    OPENAI_API_URL:str
    # GEMINI_API_KEY:str
    # GENERATION_MODEL_ID:str
    
    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    # MONGODB_URL: str
    # MONGODB_DATABASE: str

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str



    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    INPUT_DAFAULT_MAX_CHARACTERS: int = None
    GENERATION_DAFAULT_MAX_TOKENS: int = None
    GENERATION_DAFAULT_TEMPERATURE: float = None

    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str = None
    VECTOR_DB_DISTANCE_METHOD: str = None

    VECTOR_DB_HOST : str = None
    VECTOR_DB_PORT : int = None


    class Config:
        env_file = ".env"

def get_settings():
    return Settings()