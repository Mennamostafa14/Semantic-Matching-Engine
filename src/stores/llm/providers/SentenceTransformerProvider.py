from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class SentenceTransformerProvider:

    def __init__(
        self,
        model_id: str,
        default_input_max_characters: int = 4000,
        
    ):

        self.model_id = model_id
        self.default_input_max_characters = default_input_max_characters
        

        logger.info(f"Loading embedding model: {model_id}")

        self.model = SentenceTransformer(model_id)
        self.embedding_size = self.model.get_sentence_embedding_dimension()

    def set_embedding_model(
        self,
        model_id: str,
        embedding_size: int = None,
    ):

        if model_id != self.model_id:

            logger.info(f"Reloading embedding model: {model_id}")

            self.model_id = model_id
        self.model = SentenceTransformer(model_id)
    # =========================================================
    # Embed Text
    # =========================================================
    def embed_text(
        self,
        text: Union[str, List[str]],
        document_type: str = None,
    ) -> Union[List[float], List[List[float]]]:

        try:
            single = isinstance(text, str)   # ← track if input was single string

            if isinstance(text, str):
                text = [text]

            cleaned_texts = [
                t[:self.default_input_max_characters]
                for t in text
            ]

            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            result = embeddings.tolist()
            return result[0] if single else result   # ← رجّعي vector واحد لو input واحد

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [] if not isinstance(text, str) else []
        

        