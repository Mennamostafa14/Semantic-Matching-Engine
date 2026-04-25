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

    # 3. Embed with None-filtering
    embedded = []
    for text, meta in zip(texts, metadata):
        vec = nlp_controller.embedding_client.embed_text(text=text, document_type="document")
        if vec:
            embedded.append((text, meta, vec))
        else:
            logger.warning(f"Embedding failed for chunk, skipping: {text[:60]!r}")

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
async def compare_proposal(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Form(5),
    chunk_size: int = Form(300),
    overlap_size: int = Form(50),
    search_limit: int = Form(50),
    min_chunks: int = Form(2),
    explain: bool = Form(False),        # NEW: opt-in to LLM analysis
    explain_top_n: int = Form(3),       # NEW: how many proposals to explain (max)
):
    """
    Semantic comparison pipeline — unchanged through step 7.
    Step 8 (optional): generate LLM explanation for top-N proposals.
    """

    # ── helpers (unchanged) ──────────────────────────────────────────────────

    def _mean_pool(vecs: list[list[float]]) -> list[float]:
        arr  = np.array(vecs, dtype=np.float32)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm).tolist() if norm > 0 else mean.tolist()

    def _parse_hit(r) -> dict | None:
        meta = r.metadata or {}
        pid  = meta.get("proposal_id")
        if not pid:
            logger.warning(f"Hit with missing proposal_id skipped: {r.text[:60]!r}")
            return None
        return {
            "project_id": pid,
            "section":    (meta.get("section") or "unknown").strip(),
            "text":       r.text,
            "score":      r.score,
        }

    # ── 1. read file ──────────────────────────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        return JSONResponse(status_code=400, content={"signal": "empty_file"})

    # ── 2. chunk ──────────────────────────────────────────────────────────────
    process_controller = ProcessController()
    chunks = process_controller.process_file_bytes(
        file_bytes=file_bytes,
        file_name=file.filename,
        proposal_id="compare_temp",
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )
    if not chunks:
        return JSONResponse(
            status_code=400,
            content={"signal": "extraction_failed", "detail": "No text extracted or chunking failed"},
        )

    # ── 3. embed chunks ───────────────────────────────────────────────────────
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    chunk_vectors: list[list[float]] = []
    for chunk in chunks:
        vec = nlp_controller.embedding_client.embed_text(
            text=chunk.page_content,
            document_type=DocumentTypeEnum.QUERY.value,
        )
        if vec:
            chunk_vectors.append(vec)

    if not chunk_vectors:
        return JSONResponse(status_code=400, content={"signal": "embedding_error"})

    # ── 4. hybrid retrieval ───────────────────────────────────────────────────
    collection_name = "proposals"
    seen: set[tuple[str, str]] = set()
    raw_hits: list[dict] = []

    def _collect(results) -> None:
        if not results:
            return
        for r in results:
            hit = _parse_hit(r)
            if hit is None:
                continue
            key = (hit["project_id"], hit["text"])
            if key not in seen:
                seen.add(key)
                raw_hits.append(hit)

    global_vector = _mean_pool(chunk_vectors)
    _collect(
        request.app.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=global_vector,
            limit=search_limit,
        )
    )

    high_value_chunks = [
        c for c in chunks
        if (c.metadata or {}).get("section") in ("objectives", "problem_definition")
    ]
    for chunk in high_value_chunks[:3]:
        vec = nlp_controller.embedding_client.embed_text(
            text=chunk.page_content,
            document_type=DocumentTypeEnum.QUERY.value,
        )
        if not vec:
            continue
        _collect(
            request.app.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=vec,
                limit=max(search_limit // 3, 10),
            )
        )

    if not raw_hits:
        return JSONResponse(status_code=400, content={"signal": "vectordb_search_error"})

    # ── 5. extract query keywords ─────────────────────────────────────────────
    query_full_text = " ".join(c.page_content for c in chunks)
    query_keywords  = extract_keywords(query_full_text, top_n=20)

#-----------------------------------------------------------------------------
    # ── Build query proposal context (NEW) ─────────────────────────────

    query_proposal = {
        "project_id": "query",
        "overall_score": 100,  # مش مهم هنا
        "sections": {},        # اختياري
        "keywords": {
            "overlap": query_keywords
        },
        "top_passages": [
            {
                "score": 100,
                "text": query_full_text[:500]  # جزء من النص
            }
        ]
    }
    # ── 6 + 7. group, score, re-rank ─────────────────────────────────────────
    proposals = build_proposals(
        raw_hits=raw_hits,
        query_keywords=query_keywords,
        min_chunks=min_chunks,
        limit=limit,
    )

    if not proposals:
        return JSONResponse(
            status_code=400,
            content={"signal": "no_proposals_above_threshold"},
        )

    # ── 8. LLM analysis (opt-in, non-blocking) ────────────────────────────────
    #
    # Why ThreadPoolExecutor and not asyncio.gather?
    #   google-generativeai's generate_content() is a synchronous blocking
    #   call.  Running it directly in an async endpoint would block the entire
    #   event loop.  We offload it to a thread pool so FastAPI stays responsive
    #   while Gemini calls run in parallel across the top-N proposals.
    #
    # Why opt-in (explain=False by default)?
    #   LLM calls add ~1–3 s of latency per proposal.  Clients that only need
    #   scores should not pay that cost.


 
    if explain:
        top_n = min(explain_top_n, len(proposals))

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    generate_similarity_analysis,
                    query_proposal,
                    proposals[i],
                    request.app.generation_client
                )
                for i in range(top_n)
            ]

            analyses = [f.result() for f in futures]

        # attach explanation to proposals
        for i in range(top_n):
            proposals[i]["llm_analysis"] = analyses[i]["analysis"]

    else:
        for proposal in proposals:
            proposal["analysis"] = None

    # ── 9. summary + response ─────────────────────────────────────────────────
    overall_scores = [p["overall_score"] for p in proposals]
    summary = {
        "file_name":          file.filename,
        "llm_analysis":       explain,
    }

    return JSONResponse(content={
        "file_name": file.filename,
        "llm_analysis": [
            {
                "proposal_id": p["project_id"],
                "llm_analysis": p["llm_analysis"]
            }
            for p in proposals
            if p.get("llm_analysis")
        ]
    })