from ..LLMInterface import LLMInterface
import logging
import requests
from ..LLMEnums import OllamaEnums

class OllamaProvider(LLMInterface):
    """
    Local LLM provider using Ollama — generation only.
    Calls /api/chat directly (no OpenAI compatibility layer).
    """

    def __init__(
        self,
        base_url: str = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = OllamaEnums
        self.logger = logging.getLogger(__name__)

    # -------------------------
    # Model configuration
    # -------------------------
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int = None):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    # -------------------------
    # Text preprocessing
    # -------------------------
    def process_text(self, text: str) -> str:
        if not text:
            return ""
        return text[: self.default_input_max_characters].strip()

    # -------------------------
    # Generation
    # -------------------------
    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        if not self.generation_model_id:
            self.logger.error("Generation model is not set")
            return None

        temperature = temperature or self.default_generation_temperature
        num_predict = max_output_tokens or self.default_generation_max_output_tokens

        # build messages safely
        messages = list(chat_history) if chat_history else []

        messages.append({
            "role": self.enums.USER.value,  # "user"
            "content": self.process_text(prompt),
        })

        try:
            url = f"{self.base_url}/api/chat"
            print("=" * 50)
            print("BASE URL:", repr(self.base_url))
            print("FULL URL:", repr(url))
            print("MODEL:", repr(self.generation_model_id))
            print("=" * 50)

            response = requests.post(
                url,
                json={
                    "model": self.generation_model_id,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict,
                    },
                },
                timeout=120,
            )
            print("STATUS:", response.status_code)
            print("RESPONSE:", response.text[:1000])
            response.raise_for_status()
            data = response.json()

            # Ollama chat response format
            content = data.get("message", {}).get("content")

            if not content:
                self.logger.error(f"Empty response from Ollama: {data}")
                return None

            return content

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama request error: {e}")
            return None

        except Exception as e:
            self.logger.error(f"Ollama generation error: {e}")
            return None
    # -------------------------
    # Embedding — not supported
    # -------------------------
    def embed_text(self, text: str, document_type: str = None):
        self.logger.warning(
            "OllamaProvider does not handle embeddings. "
            "Use SentenceTransformerProvider instead."
        )
        return None

    # -------------------------
    # Prompt helper
    # -------------------------
    def construct_prompt(self, prompt: str, role: str = None):
        return {
            "role": role if role else self.enums.USER.value,
            "content": self.process_text(prompt),
        }

