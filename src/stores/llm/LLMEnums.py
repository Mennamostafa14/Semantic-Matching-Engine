from enum import Enum


class LLMEnums(Enum):
    GEMINI = "GEMINI"
    OPENAI = "OPENAI"
    SENTENCE_TRANSFORMER = "SENTENCE_TRANSFORMER"
    OLLAMA="OLLAMA"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class OllamaEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class GeminiRoleEnum(Enum):
    USER = "user"
    MODEL = "model"


class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"