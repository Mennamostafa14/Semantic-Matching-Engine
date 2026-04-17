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

@nlp_router.post("/index/compare")
async def compare_proposal(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Form(5),
    chunk_size: int = Form(300),
    overlap_size: int = Form(50),
    search_limit: int = Form(50),
    min_chunks: int = Form(2),          # noise filter: ignore proposals below this
):
    """
    Semantic comparison pipeline:
      1.  Extract + clean + chunk the uploaded file.
      2.  Embed every chunk with QUERY document type.
      3.  Mean-pool chunk vectors → single query vector.
      4.  Multi-query search: also search per-chunk for top-K then merge
          (hybrid: global + local retrieval).
      5.  Extract query keywords.
      6.  Group hits → proposal → section.
      7.  Re-rank using vector score + keyword overlap + section importance.
      8.  Return structured explainable response.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _mean_pool(vecs: list[list[float]]) -> list[float]:
        arr  = np.array(vecs, dtype=np.float32)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm).tolist() if norm > 0 else mean.tolist()

    def _parse_hit(r) -> dict | None:
        """Safely convert a RetrievedDocument to a raw hit dict."""
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

    # ── 1. read file ─────────────────────────────────────────────────────────
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

    # ── 4. hybrid retrieval: global (mean-pool) + local (per-chunk) ───────────
    #
    # Why both?
    #   • Mean-pool vector captures the document's overall topic.
    #     Good for finding globally similar proposals.
    #   • Per-chunk vectors find fine-grained section-level matches.
    #     Good for finding proposals that share ONE very specific component
    #     (e.g. same objectives but different background).
    #
    # We deduplicate by (project_id, text) so the same chunk doesn't inflate
    # the score if it shows up in both search passes.

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

    # 4a. global search with mean-pooled vector
    global_vector = _mean_pool(chunk_vectors)
    _collect(
        request.app.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=global_vector,
            limit=search_limit,
        )
    )

    # 4b. local search: use only HIGH-WEIGHT section chunks (objectives,
    #     problem_definition) to avoid diluting results with boilerplate.
    #     Cap at 3 chunk queries to stay within reasonable latency.
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
    query_full_text  = " ".join(c.page_content for c in chunks)
    query_keywords   = extract_keywords(query_full_text, top_n=20)

    # ── 6 + 7. group, score, re-rank (delegated to scoring.py) ───────────────
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

    # ── 8. summary + response ─────────────────────────────────────────────────
    overall_scores = [p["overall_score"] for p in proposals]
    summary = {
        "file_name":          file.filename,
        "text_length":        sum(len(c.page_content) for c in chunks),
        "chunks_embedded":    len(chunk_vectors),
        "total_hits":         len(raw_hits),
        "projects_found":     len(proposals),
        "highest_similarity": max(overall_scores)  if overall_scores else 0,
        "average_similarity": round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0,
    }

    return JSONResponse(content={
        "signal":    ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
        "summary":   summary,
        "proposals": proposals,
    })