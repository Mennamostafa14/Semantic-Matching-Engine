from ..LLMInterface import LLMInterface
import logging
from google import genai
import time

class GeminiProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1
    ):

        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = genai.Client(api_key=self.api_key)

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
    def process_text(self, text: str):
        if not text:
            return ""
        return text[:self.default_input_max_characters].strip()

    # -------------------------
    # Generation (LLM)
    # -------------------------
    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None
    ):

        if not self.client:
            self.logger.error("Gemini client is not initialized")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model is not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        try:
            response = self.client.models.generate_content(
                model=self.generation_model_id,
                    contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "You are a strict JSON generator. "
                                "Return ONLY valid JSON. No extra text.\n\n"
                                + self.process_text(prompt)
                            )
                        }
                    ]
                }
            ],
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                    "response_mime_type": "application/json" 
                }
            )

            if not response or not response.text:
                self.logger.error("Empty response from Gemini generation")
                return None

            text = None

            # Case 1: normal SDK response
            if hasattr(response, "text") and response.text:
                text = response.text

            # Case 2: structured response (new Gemini SDK behavior)
            elif hasattr(response, "candidates"):
                try:
                    text = response.candidates[0].content.parts[0].text
                except Exception:
                    text = None

            if not text:
                self.logger.error("Empty extracted text from Gemini response")
                return None

            return text


        except Exception as e:
            self.logger.error(f"Gemini generation error: {e}")
            return None

    # -------------------------
    # Embeddings
    # -------------------------
    def embed_text(self, text: str, document_type: str = None):

        if not self.client:
            self.logger.error("Gemini client is not initialized")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model is not set")
            return None

        try:
            response = self.client.models.embed_content(
                model=self.embedding_model_id,
                contents=self.process_text(text)
            )

            if not response or not response.embeddings:
                self.logger.error("Empty embedding response from Gemini")
                return None

            return response.embeddings[0].values

        except Exception as e:
            self.logger.error(f"Gemini embedding error: {e}")
            return None

    # -------------------------
    # Prompt helper (optional)
    # -------------------------
    def construct_prompt(self, prompt: str, role: str = None):
        return self.process_text(prompt)