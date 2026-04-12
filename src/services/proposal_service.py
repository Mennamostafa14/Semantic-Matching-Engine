import os
from uuid import uuid4
from fastapi import UploadFile, Request

from controllers import NLPController, ProcessController


class ProposalService:
    def __init__(self, project_id: str, request: Request):
        self.project_id = project_id
        self.request = request

        self.process_controller = ProcessController(project_id=project_id)

        self.nlp_controller = NLPController(
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
        )

        self.project_path = self.process_controller.project_path

    async def push_proposal(
        self,
        proposal_id: str,
        file: UploadFile,
        do_reset: bool = False,
    ):
        """
        Full pipeline:
        upload → save → load → chunk → embed → store vector DB
        """

        # 1. ensure directory exists
        os.makedirs(self.project_path, exist_ok=True)

        # 2. save file
        safe_filename = f"{uuid4()}_{file.filename}"
        file_path = os.path.join(self.project_path, safe_filename)

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # 3. load file
        file_content = self.process_controller.get_file_content(safe_filename)

        if not file_content:
            raise ValueError("Failed to load file content")

        # 4. chunking
        chunks = self.process_controller.process_file_content(
            file_content=file_content,
            file_id=safe_filename
        )

        if not chunks:
            raise ValueError("No chunks generated")

        # 5. build chunks
        proposal_chunks = []
        for i, chunk in enumerate(chunks):
            proposal_chunks.append({
                "text": chunk.page_content,
                "chunk_id": f"{proposal_id}_{i}",
                "metadata": {
                    "project_id": self.project_id,
                    "proposal_id": proposal_id,
                    "source_file": file.filename,
                },
            })

        # 6. index
        is_inserted = await self.nlp_controller.index_into_vector_db(
            project_id=self.project_id,
            proposal_id=proposal_id,
            chunks=proposal_chunks,
            do_reset=do_reset,
            chunks_ids=[c["chunk_id"] for c in proposal_chunks],
        )

        if not is_inserted:
            raise ValueError("Vector DB insertion failed")

        return {
            "proposal_id": proposal_id,
            "chunks_count": len(proposal_chunks),
            "file_name": file.filename,
        }