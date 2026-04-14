from fastapi import FastAPI, APIRouter, status, Request,UploadFile,Form, File
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest, ProposalRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController,ProcessController
from models import ResponseSignal
import os
# from services import ProposalService
import logging
from collections import defaultdict
import numpy as np
import uuid

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

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


# ── NEW: push a single proposal into the vector DB ────────────────────────────

@nlp_router.post("/index/push-proposal")
async def push_proposal(
    request: Request,
    proposal_id: str = Form(...),
    do_reset: bool = Form(False),
    file: UploadFile = File(...)
):
    process_controller = ProcessController(project_id="temp")

    # 1. save file
    # file_path = os.path.join(process_controller.project_path, file.filename)
    # with open(file_path, "wb") as f:
    #     f.write(await file.read())

    # # 2. load
    # file_content = process_controller.get_file_content(file_id=file.filename)

    # if not file_content:
    #     return JSONResponse(
    #         status_code=422,
    #         content={"signal": "file_processing_failed"}
    #     )
    file_bytes = await file.read()

    # 3. chunk
    chunks = process_controller.process_file_bytes(
        file_bytes=file_bytes,
        file_name=file.filename,
        proposal_id=proposal_id,
        chunk_size=300,      # يمكن جعلها قابلة للتكوين من settings
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
    metadata = [c.metadata for c in chunks]

    vectors = [
        nlp_controller.embedding_client.embed_text(
            text=t,
            document_type="document"
        )
        for t in texts
    ]

    print("Number of vectors:", len(vectors))
    print("Single vector size:", len(vectors[0]) if vectors else "None")

    # create collection
    request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=nlp_controller.embedding_client.embedding_size,
        do_reset=do_reset,
    )

    # insert
    request.app.vectordb_client.insert_many(
        collection_name=collection_name,
        texts=texts,
        metadata=metadata,
        vectors=vectors,
        record_ids=list(range(len(texts)))
    )

    return {
        "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
        "proposal_id": proposal_id,
        "chunks": len(chunks)
    }

# ── NEW: compare a proposal against everything already in the vector DB ───────



# @nlp_router.post("/index/compare")
# async def compare_proposal(request: Request, search_request: SearchRequest):

#     nlp_controller = NLPController(
#         vectordb_client=request.app.vectordb_client,
#         generation_client=request.app.generation_client,
#         embedding_client=request.app.embedding_client,
#     )

#     collection_name = "proposals"

#     # ── 1. embed query ───────────────────────────────────────
#     query_vector = nlp_controller.embedding_client.embed_text(
#         text=search_request.text,
#         document_type="query"
#     )

#     if not query_vector:
#         return JSONResponse(
#             status_code=400,
#             content={"signal": "embedding_error"}
#         )

#     # ── 2. search ────────────────────────────────────────────
#     results = request.app.vectordb_client.search_by_vector(
#         collection_name=collection_name,
#         vector=query_vector,
#         limit=50   # 👈 مهم نكبرها
#     )

#     if not results:
#         return JSONResponse(
#             status_code=400,
#             content={"signal": "vectordb_search_error"}
#         )

#     # ── 3. group by proposal_id ─────────────────────────────
#     proposal_scores = defaultdict(list)

#     for r in results:
#         proposal_id = r.metadata.get("proposal_id", "unknown")
#         proposal_scores[proposal_id].append(r.score)

#     # ── 4. compute final score ──────────────────────────────
#     proposals = []

#     for pid, scores in proposal_scores.items():
#         avg_score = sum(scores) / len(scores)
#         max_score = max(scores)

#         final_score = (0.7 * max_score) + (0.3 * avg_score)

#         proposals.append({
#             "proposal_id": pid,
#             "avg_similarity": round(avg_score * 100, 2),
#             "max_similarity": round(max_score * 100, 2),
#             "final_score": round(final_score * 100, 2),
#             "matched_chunks": len(scores),
#         })

#     # ── 5. sort ─────────────────────────────────────────────
#     proposals = sorted(
#         proposals,
#         key=lambda x: x["final_score"],
#         reverse=True
#     )[:search_request.limit]

#     # ── 6. summary ──────────────────────────────────────────
#     scores = [p["final_score"] for p in proposals]

#     summary = {
#         "total_proposals": len(proposals),
#         "highest_similarity": max(scores) if scores else 0,
#         "average_similarity": round(sum(scores) / len(scores), 2) if scores else 0,
#     }

#     return JSONResponse(
#         content={
#             "signal": "vectordb_search_success",
#             "query_text": search_request.text,
#             "summary": summary,
#             "proposals": proposals,
#         }
#     )

@nlp_router.post("/index/compare")
async def compare_proposal(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Form(5),
    chunk_size: int = Form(512),
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
    process_controller = ProcessController()   # لاحظي: modified constructor (no project_id)
    chunks = process_controller.process_file_bytes(
        file_bytes=file_bytes,
        file_name=file.filename,
        proposal_id="compare_temp",   # مؤقت، لأن المقارنة لا تحتاج proposal حقيقي
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
            document_type="document",   # لأننا نخزن المستند
        )
        if vec:
            vectors.append(vec)

    if not vectors:
        return JSONResponse(status_code=400, content={"signal": "embedding_error"})

    # Mean pooling + L2 normalization
    query_vector = _mean_pool(vectors)   # استخدمي الدالة المساعدة أو اكتبيها

    # ── 4. search across all collections ────────────────────────────────────
    all_cols = request.app.vectordb_client.list_all_collections()
    if not all_cols or not all_cols.collections:
        return JSONResponse(status_code=400, content={"signal": "no_collections_found"})


    collection_name = "proposals"

    results = request.app.vectordb_client.search_by_vector(
        collection_name=collection_name,
        vector=query_vector,
        limit=search_limit,
    )

    raw_hits = []

    if results:
        for r in results:
            proposal_id = r.metadata.get("proposal_id")

            raw_hits.append({
                "proposal_id": proposal_id,
                "text": r.text,
                "score": r.score,
            })

    if not raw_hits:
        return JSONResponse(status_code=400, content={"signal": "vectordb_search_error"})
    print("Raw hits:", raw_hits[:3])
    # ── 5. group by proposal_id and compute scores ───────────────────────────
    from collections import defaultdict
    bucket = defaultdict(list)
    texts_by_proposal = defaultdict(list)

    for hit in raw_hits:
        pid = hit["proposal_id"]
        bucket[pid].append(hit["score"])
        texts_by_proposal[pid].append(hit["text"])

    proposals = []
    for pid, scores in bucket.items():
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        final_score = 0.7 * max_score + 0.3 * avg_score

        # أفضل 3 نصوص متطابقة
        top_passages = sorted(
            zip(scores, texts_by_proposal[pid]),
            key=lambda x: x[0],
            reverse=True
        )[:3]

        proposals.append({
            "proposal_id": pid,
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