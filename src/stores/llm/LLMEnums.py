from enum import Enum


class LLMEnums(Enum):
    GEMINI = "GEMINI"


class GeminiRoleEnum(Enum):
    USER = "user"
    MODEL = "model"


class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"