# src/routes/nlp.py
from fastapi import FastAPI, APIRouter, status, Request, UploadFile, Form, File
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest, ProposalRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController, ProcessController
from models import ResponseSignal
from helpers.scoring import build_proposals, extract_keywords   # ← new import
import os
import logging
from collections import defaultdict
import numpy as np
import hashlib
from stores.llm.LLMEnums import DocumentTypeEnum
from helpers.proposal_analysis import generate_similarity_analysis
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)


# ── push-proposal (unchanged from last reviewed version) ──────────────────────

@nlp_router.post("/index/push-proposal")
async def push_proposal(
    request: Request,
    proposal_id: str = Form(...),
    do_reset: bool = Form(False),
    file: UploadFile = File(...)
):
    process_controller = ProcessController(project_id="temp")
    file_bytes = await file.read()

    chunks = process_controller.process_file_bytes(
        file_bytes=file_bytes,
        file_name=file.filename,
        proposal_id=proposal_id,
        chunk_size=500,
        overlap_size=50,
    )

    if not chunks:
        return JSONResponse(
            status_code=422,
            content={"signal": ResponseSignal.PROCESSING_FAILED.value}
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    collection_name = "proposals"

    texts = [c.page_content for c in chunks]
    metadata = [
        {**(c.metadata or {}), "proposal_id": proposal_id}
        for c in chunks
    ]

    # 1. Create/reset collection FIRST
    if do_reset:
        request.app.vectordb_client.delete_collection(collection_name)

    if not request.app.vectordb_client.is_collection_existed(collection_name):
        request.app.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=nlp_controller.embedding_client.embedding_size,
            do_reset=False,
        )

    # 2. Delete only this proposal's existing records (safe re-push)
    if not do_reset and request.app.vectordb_client.is_collection_existed(collection_name):
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        request.app.vectordb_client.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="proposal_id", match=MatchValue(value=proposal_id))]
            )
        )

# 3. Embed — batch واحد بدل loop
    all_vectors = nlp_controller.embedding_client.embed_text(
        text=texts,
        document_type="document"
    )

    embedded = [
        (text, meta, vec)
        for text, meta, vec in zip(texts, metadata, all_vectors)
        if vec
    ]

    if not embedded:
        return JSONResponse(status_code=422, content={"signal": "embedding_failed_all_chunks"})

    texts_out, metadata_out, vectors_out = zip(*embedded)

    def _make_record_id(proposal_id: str, chunk_index: int) -> int:
        raw = f"{proposal_id}:{chunk_index}".encode()
        return int(hashlib.md5(raw).hexdigest()[:16], 16) % (2 ** 53)

    record_ids = [_make_record_id(proposal_id, i) for i in range(len(texts_out))]

    request.app.vectordb_client.insert_many(
        collection_name=collection_name,
        texts=list(texts_out),
        metadata=list(metadata_out),
        vectors=list(vectors_out),
        record_ids=record_ids,
    )

    return {
        "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
        "proposal_id": proposal_id,
        "chunks": len(chunks),
    }


# ── compare (upgraded) ────────────────────────────────────────────────────────

# ── additions to the top of nlp.py ───────────────────────────────────────────
#
# Add these two lines alongside your existing imports:
#
#   from routes.proposal_analysis import generate_similarity_analysis
#   from concurrent.futures import ThreadPoolExecutor, as_completed
#
# Add this to your FastAPI lifespan / startup event (once, not per-request):
#
#   from routes.proposal_analysis import configure_gemini
#   configure_gemini(api_key=settings.GEMINI_API_KEY)
#
# ─────────────────────────────────────────────────────────────────────────────


@nlp_router.post("/index/compare")
async def compare_index(
    request: Request,
    file: UploadFile = File(...)
):
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    result = await nlp_controller.compare_documents_scores_only(file=file)

    if not result:
        return JSONResponse(
            status_code=400,
            content={"signal": "compare_failed"}
        )

    return JSONResponse(content=result)

