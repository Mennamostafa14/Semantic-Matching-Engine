import os
import logging

import aiofiles
from fastapi import APIRouter, Request, UploadFile, status
from fastapi.responses import JSONResponse

from controllers import ProposalController
from models import ResponseSignal
from routes.schemes.proposal import IndexRequest, AnalyzeRequest
from helpers.config import get_settings

logger = logging.getLogger("uvicorn.error")

proposal_router = APIRouter(
    prefix="/api/v1/proposals",
    tags=["proposals"],
)


def _controller(request: Request) -> ProposalController:
    """Build a ProposalController from the app-level shared clients."""
    return ProposalController(
        vectordb_client=request.app.vectordb_client,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser,
    )


# ------------------------------------------------------------------ #
# POST /proposals/upload/{file_id}
# Upload a proposal file and return its file_id for later indexing.
# ------------------------------------------------------------------ #

@proposal_router.post("/upload")
async def upload_proposal(request: Request, file: UploadFile):
    """
    Upload a proposal file (.txt or .pdf).
    Returns the generated file_id which is used to index the proposal.
    """
    settings = get_settings()
    ctrl = _controller(request)

    is_valid, signal = ctrl.validate_file(file)
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": signal},
        )

    file_path, file_id = ctrl.generate_unique_filepath(file.filename)

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value},
        )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "file_id": file_id,
            "original_filename": file.filename,
            "size_bytes": os.path.getsize(file_path),
        },
    )


# ------------------------------------------------------------------ #
# POST /proposals/index/{file_id}
# Extract, chunk, embed, and store the uploaded file in Qdrant.
# ------------------------------------------------------------------ #

@proposal_router.post("/index/{file_id}")
async def index_proposal(request: Request, file_id: str, body: IndexRequest):
    """
    Index a previously uploaded proposal file into the vector database.
    Each text chunk is stored as a separate vector with full metadata in the payload.
    """
    ctrl = _controller(request)

    result = ctrl.index_proposal_file(
        file_id=file_id,
        proposal_title=body.proposal_title,
        chunk_size=body.chunk_size,
        overlap_size=body.overlap_size,
        do_reset=body.do_reset,
    )

    if not result["success"]:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"signal": result["reason"]},
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROPOSAL_INDEXED_SUCCESS.value,
            "proposal_id": result["proposal_id"],
            "proposal_title": result["proposal_title"],
            "chunks_indexed": result["chunks_indexed"],
        }
    )


# ------------------------------------------------------------------ #
# POST /proposals/analyze
# Compare a query proposal against the Qdrant collection.
# ------------------------------------------------------------------ #

@proposal_router.post("/analyze")
async def analyze_proposal(request: Request, body: AnalyzeRequest):
    """
    Embed the submitted proposal text, retrieve the most similar stored proposals
    via vector search, then return:
      - A ranked list of matches with similarity scores and text previews.
      - An LLM-generated narrative analysis of the similarities and differences.
    """
    ctrl = _controller(request)

    matches, full_prompt, analysis = ctrl.analyze_similarity(
        query_text=body.proposal_text,
        limit=body.limit,
    )

    if matches is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value},
        )

    if analysis is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.PROPOSAL_ANALYSIS_ERROR.value},
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROPOSAL_ANALYSIS_SUCCESS.value,
            "total_matches": len(matches),
            "matches": [
                {
                    "rank": i + 1,
                    "similarity_score": round(m.score, 6),
                    "text_preview": m.text[:400] + ("..." if len(m.text) > 400 else ""),
                }
                for i, m in enumerate(matches)
            ],
            "analysis": analysis,
        }
    )


# ------------------------------------------------------------------ #
# GET /proposals/list
# List all indexed proposals (unique, deduplicated from Qdrant payloads).
# ------------------------------------------------------------------ #

@proposal_router.get("/list")
async def list_proposals(request: Request):
    """
    Return all unique proposals currently stored in the vector database,
    extracted from Qdrant payload metadata.
    """
    ctrl = _controller(request)
    proposals = ctrl.list_proposals()

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROPOSAL_LIST_SUCCESS.value,
            "total": len(proposals),
            "proposals": proposals,
        }
    )


# ------------------------------------------------------------------ #
# DELETE /proposals/{proposal_id}
# Remove all vectors belonging to a proposal from Qdrant.
# ------------------------------------------------------------------ #

@proposal_router.delete("/{proposal_id}")
async def delete_proposal(request: Request, proposal_id: str):
    """
    Delete all vector chunks associated with a proposal from Qdrant.
    The original file on disk is NOT removed.
    """
    ctrl = _controller(request)
    ok = ctrl.delete_proposal(proposal_id)

    if not ok:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": ResponseSignal.PROPOSAL_NOT_FOUND.value},
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.PROPOSAL_DELETED_SUCCESS.value,
            "proposal_id": proposal_id,
        }
    )