from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import nlp
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from utils.metrics import setup_metrics
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    try:
        logger.info("🚀 Starting up...")
        settings = get_settings()

        llm_provider_factory = LLMProviderFactory(settings)
        vectordb_provider_factory = VectorDBProviderFactory(settings)

        logger.info("Loading generation client...")
        app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
        app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)
        logger.info("✅ Generation client ready.")

        logger.info("Loading embedding client...")
        app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
        app.embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=settings.EMBEDDING_MODEL_SIZE
        )
        logger.info("✅ Embedding client ready.")

        logger.info("Connecting to VectorDB...")
        app.vectordb_client = vectordb_provider_factory.create(
            provider=settings.VECTOR_DB_BACKEND
        )
        app.vectordb_client.connect()
        logger.info("✅ VectorDB connected.")

        logger.info("✅ Startup complete.")

    except Exception as e:
        logger.error(f"❌ STARTUP FAILED: {e}", exc_info=True)
        raise  # مهم — خلّي Railway يشوف الـ error

    yield  # ── التطبيق شغّال ──

    # ── Shutdown ──
    logger.info("Shutting down...")
    app.vectordb_client.disconnect()

app = FastAPI(lifespan=lifespan)
setup_metrics(app)
app.include_router(nlp.nlp_router)