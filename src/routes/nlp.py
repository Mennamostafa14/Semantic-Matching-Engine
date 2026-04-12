from fastapi import FastAPI, APIRouter, status, Request,UploadFile,Form, File
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest, ProposalRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController,ProcessController
from models import ResponseSignal
import os
from services import ProposalService
import logging

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
        template_parser=request.app.template_parser,
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

@nlp_router.post("/index/push-proposal/{project_id}")
async def push_proposal(
    request:     Request,
    project_id:  str,
    proposal_id: str        = Form(...),
    do_reset:    bool       = Form(False),
    file:        UploadFile = File(...),
):
    # ── 1. Save the uploaded file into the project directory ──────────────
    process_controller = ProcessController(project_id=project_id)

    file_path = os.path.join(process_controller.project_path, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # ── 2. Load and extract text via the existing controller ──────────────
    file_content = process_controller.get_file_content(file_id=file.filename)

    if not file_content:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "signal": ResponseSignal.FILE_PROCESSING_FAILED.value,
            }
        )

    # ── 3. Split into chunks ───────────────────────────────────────────────
    chunks = process_controller.process_file_content(
        file_content=file_content,
        file_id=file.filename,
    )

    # ── 4. Resolve project ────────────────────────────────────────────────
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value,
            }
        )

    # ── 5. Embed and index into the vector DB ─────────────────────────────
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,

    )

    chunks_ids = list(range(len(chunks)))

    is_inserted = nlp_controller.index_into_vector_db(
        project=project,
        chunks=chunks,
        do_reset=do_reset,
        chunks_ids=chunks_ids,
    )

    if not is_inserted:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value,
            }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "proposal_id": proposal_id,
            "inserted_chunks_count": len(chunks),
        }
    )


# ── NEW: compare a proposal against everything already in the vector DB ───────

@nlp_router.post("/index/compare/{project_id}")
async def compare_proposal(request: Request, project_id: str, search_request: SearchRequest):
    """
    Embed the incoming proposal text, search the project's vector collection
    for the closest matches, and return a ranked similarity report.

    Body (SearchRequest):
        text  : str – the proposal text to compare
        limit : int – how many similar proposals to return (default: 5)

    Response includes, per match:
        text                 – the stored proposal text
        similarity_score     – raw cosine similarity in [0, 1]
        similarity_percentage – score × 100, rounded to 2 dp
        similarity_label     – human-readable band:
                               "Very high" ≥ 0.90
                               "High"      ≥ 0.75
                               "Moderate"  ≥ 0.50
                               "Low"       < 0.50
    """
    project_model = await ProjectModel.create_instance(
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

    results = nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=search_request.limit,
    )

    if not results:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value}
        )

    def _similarity_label(score: float) -> str:
        if score >= 0.90:
            return "Very high"
        if score >= 0.75:
            return "High"
        if score >= 0.50:
            return "Moderate"
        return "Low"

    matches = [
        {
            "text": r.text,
            "similarity_score": round(r.score, 4),
            "similarity_percentage": round(r.score * 100, 2),
            "similarity_label": _similarity_label(r.score),
        }
        for r in results
    ]

    # Summary statistics for the caller's convenience
    scores = [m["similarity_score"] for m in matches]
    summary = {
        "total_matches": len(matches),
        "highest_similarity": round(max(scores) * 100, 2) if scores else 0,
        "average_similarity": round((sum(scores) / len(scores)) * 100, 2) if scores else 0,
    }

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "query_text": search_request.text,
            "summary": summary,
            "matches": matches,
        }
    )

