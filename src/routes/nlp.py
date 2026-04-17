
from fastapi import FastAPI, APIRouter, status, Request,UploadFile,Form, File
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest, ProposalRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController,ProcessController
from models import ResponseSignal
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
"""
# ── existing endpoint (unchanged) ──────────────────────────────────────────────

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(
            project_id=project.id, page_no=page_no
        )
        if len(page_chunks):
            page_no += 1

        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids = list(range(idx, idx + len(page_chunks)))
        idx += len(page_chunks)

        is_inserted = nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            do_reset=push_request.do_reset,
            chunks_ids=chunks_ids,
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value}
            )

        inserted_items_count += len(page_chunks)

    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count,
        }
    )
"""
# ── NEW: push a single proposal into the vector DB ────────────────────────────

@nlp_router.post("/index/push-proposal")
async def push_proposal(
    request: Request,
    proposal_id: str = Form(...),
    do_reset: bool = Form(False),
    file: UploadFile = File(...)
):
    process_controller = ProcessController(project_id="temp")


    file_bytes = await file.read()

    # 3. chunk
    chunks = process_controller.process_file_bytes(
        file_bytes=file_bytes,
        file_name=file.filename,
        proposal_id=proposal_id,
        chunk_size=500,      # يمكن جعلها قابلة للتكوين من settings
        overlap_size=50,
    )

    if not chunks:
        return JSONResponse(
            status_code=422,
            content={"signal":ResponseSignal.PROCESSING_FAILED.value }
        )

    # 4. NLP
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
# 2. Embed with None-filtering, keeping texts/metadata/vectors aligned

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
            """
            Deterministic integer ID scoped to a proposal+chunk pair.
            Re-pushing the same proposal overwrites only its own records.
            """
            raw = f"{proposal_id}:{chunk_index}".encode()
            return int(hashlib.md5(raw).hexdigest()[:16], 16) % (2**53)  # safe JS int range

    record_ids = [_make_record_id(proposal_id, i) for i in range(len(texts_out))]
    print(f"Proposal {proposal_id}: {len(embedded)} chunks embedded out of {len(texts)}")
    # insert
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
        "chunks": len(chunks)
    }

@nlp_router.post("/index/compare")
async def compare_proposal(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Form(5),
    chunk_size: int = Form(300),
    overlap_size: int = Form(50),
    search_limit: int = Form(50),
):
    """
    Upload a PDF or TXT file. The endpoint:
      1. Extracts and cleans text using ProcessController.
      2. Splits into chunks using ProcessController.
      3. Embeds every chunk then mean-pools to a single document vector.
      4. Searches all Qdrant collections for the most similar chunks.
      5. Groups hits by project_id, computes blended score, returns top projects.
    """
 

    def _mean_pool(vectors: list[list[float]]) -> list[float]:
        arr = np.array(vectors, dtype=np.float32)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            return (mean / norm).tolist()
        return mean.tolist()
    # ── 1. read file ────────────────────────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        return JSONResponse(status_code=400, content={"signal": "empty_file"})

    # ── 2. use ProcessController to extract, clean, and chunk ────────────────
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
            content={"signal": "extraction_failed", "detail": "No text extracted or chunking failed"}
        )

    # ── 3. embed each chunk and mean-pool ───────────────────────────────────
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    vectors = []
    for chunk in chunks:
        vec = nlp_controller.embedding_client.embed_text(
            text=chunk.page_content,
            document_type=DocumentTypeEnum.QUERY.value, 
        )
        if vec:
            vectors.append(vec)

    if not vectors:
        return JSONResponse(status_code=400, content={"signal": "embedding_error"})

    # Mean pooling + L2 normalization
    query_vector = _mean_pool(vectors)   

    # ── 4. search across all collections ────────────────────────────────────


    raw_hits = []
    collection_name = "proposals"
    results = request.app.vectordb_client.search_by_vector(
        collection_name=collection_name,
        vector=query_vector,
        limit=search_limit,
    )
    if results:
        for r in results:
            metadata = r.metadata or {}
            pid = metadata.get("proposal_id")
            if not pid:                        # ← ADD THIS: skip unidentifiable hits
                logger.warning(f"Hit with missing proposal_id skipped: {r.text[:60]!r}")
                continue
            raw_hits.append({
                    "project_id": pid,
                    "section":    metadata.get("section"), 
                    "text": r.text,
                    "score": r.score,
                })

    if not raw_hits:
        return JSONResponse(status_code=400, content={"signal": "vectordb_search_error"})
    
# ── 5. group by proposal_id → section ───────────────────────────────────

    # Structure: { proposal_id: { section: [score, ...] } }
    section_scores:  dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # Structure: { proposal_id: [(score, text), ...] } — for top_passages only
    proposal_hits:   dict[str, list[tuple[float, str]]] = defaultdict(list)

    for hit in raw_hits:
        pid     = hit["project_id"]                          # already guaranteed non-None (filtered above)
        section = (hit.get("section") or "unknown").strip()  # guard missing/None section
        score   = hit["score"]

        section_scores[pid][section].append(score)
        proposal_hits[pid].append((score, hit["text"]))

    # ── helper: compute the three score fields from a list of floats ─────────
    def _score_block(scores: list[float]) -> dict:
        avg   = sum(scores) / len(scores)
        mx    = max(scores)
        final = 0.7 * mx + 0.3 * avg
        return {
            "avg_similarity":   round(avg   * 100, 2),
            "max_similarity":   round(mx    * 100, 2),
            "final_score":      round(final * 100, 2),
            "matched_chunks":   len(scores),
        }

    # ── build proposal list ──────────────────────────────────────────────────
    proposals = []
    for pid, sections in section_scores.items():

        # proposal-level scores: flatten all chunk scores for this proposal
        all_scores = [s for sec_scores in sections.values() for s in sec_scores]
        proposal_block = _score_block(all_scores)

        # section-level scores
        sections_out = {
            section: _score_block(scores)
            for section, scores in sections.items()
        }

        # top 3 passages for the whole proposal (not per section)
        top_passages = sorted(proposal_hits[pid], key=lambda x: x[0], reverse=True)[:3]

        proposals.append({
            "project_id":      pid,
            "overall_score":   proposal_block["final_score"],
            "avg_similarity":  proposal_block["avg_similarity"],
            "max_similarity":  proposal_block["max_similarity"],
            "matched_chunks":  proposal_block["matched_chunks"],
            "sections":        sections_out,
            "top_passages": [
                {"score": round(s * 100, 2), "text": t}
                for s, t in top_passages
            ],
        })

    proposals.sort(key=lambda x: x["overall_score"], reverse=True)
    proposals = proposals[:limit]

    # ── 6. summary ──────────────────────────────────────────────────────────
    overall_scores = [p["overall_score"] for p in proposals]
    summary = {
        "file_name":          file.filename,
        "text_length":        sum(len(c.page_content) for c in chunks),
        "chunks_embedded":    len(vectors),
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
"""
    # ── 5. group by project_id and compute scores ───────────────────────────
    
    bucket = defaultdict(list)
    texts_by_project = defaultdict(list)

    for hit in raw_hits:
        pid = hit["project_id"]
        bucket[pid].append(hit["score"])
        texts_by_project[pid].append(hit["text"])

    proposals = []
    for pid, scores in bucket.items():
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        final_score = 0.7 * max_score + 0.3 * avg_score

  
        top_passages = sorted(
            zip(scores, texts_by_project[pid]),
            key=lambda x: x[0],
            reverse=True
        )[:3]

        proposals.append({
            "project_id": pid,
            "avg_similarity": round(avg_score * 100, 2),
            "max_similarity": round(max_score * 100, 2),
            "final_score": round(final_score * 100, 2),
            "matched_chunks": len(scores),
            "top_passages": [
                {"score": round(s * 100, 2), "text": t}
                for s, t in top_passages
            ],
        })

    proposals.sort(key=lambda x: x["final_score"], reverse=True)
    proposals = proposals[:limit]

    # ── 6. summary ──────────────────────────────────────────────────────────
    final_scores = [p["final_score"] for p in proposals]
    summary = {
        "file_name": file.filename,
        "text_length": sum(len(c.page_content) for c in chunks),
        "chunks_embedded": len(vectors),
        "total_hits": len(raw_hits),
        "projects_found": len(proposals),
        "highest_similarity": max(final_scores) if final_scores else 0,
        "average_similarity": round(sum(final_scores) / len(final_scores), 2) if final_scores else 0,
    }

    return JSONResponse(content={
        "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
        "summary": summary,
        "proposals": proposals,
    })
"""



