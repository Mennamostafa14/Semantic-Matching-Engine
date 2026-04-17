# controllers/ProcessController.py
from __future__ import annotations

import os
from typing import Optional
from collections import Counter
import numpy as np
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

from models import ProcessingEnum

from .BaseController import BaseController
from .ProjectController import ProjectController
from helpers import (
    clean_text,
    load_from_bytes,
    group_lines_by_section,
    chunk_section_docs,
    enrich_chunk_metadata,
    SECTION_WEIGHTS,
)


class ProcessController(BaseController):

    def __init__(self, project_id: str = None):
        super().__init__()
        self.project_id = project_id

    # =========================================================================
    # Legacy helpers (kept for backward compatibility)
    # =========================================================================

    def get_file_extension(self, file_id: str) -> str:
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):
        if not self.project_id:
            return None
        project_path = ProjectController().get_project_path(project_id=self.project_id)
        file_path = os.path.join(project_path, file_id)
        if not os.path.exists(file_path):
            return None
        file_ext = self.get_file_extension(file_id=file_id)
        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")
        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        return None

    def get_file_content(self, file_id: str):
        loader = self.get_file_loader(file_id=file_id)
        return loader.load() if loader else None

    # =========================================================================
    # Core public API — pure orchestration, no internal logic
    # =========================================================================

    def process_file_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        proposal_id: str = None,
        chunk_size: int = 300,
        overlap_size: int = 50,
    ) -> list[Document]:
        """
        Full semantic preprocessing pipeline.

        Steps
        -----
        1. Load PDF/TXT from raw bytes via a temp file.
        2. Clean extracted text.
        3. Group lines into section buckets (structure-aware split).
        4. Build one Document per section with rich metadata.
        5. Chunk within each section only (semantic boundary preservation).
        6. Enrich every chunk with positional and weight metadata.

        Returns a flat list of LangChain Document chunks ready for
        embedding and insertion into Qdrant.
        """
        # 1. Load
        raw_docs = load_from_bytes(file_bytes, file_name)
        if not raw_docs:
            return []

        # 2. Clean
        full_text = "\n".join(
            clean_text(doc.page_content)
            for doc in raw_docs
            if doc.page_content
        )
        if not full_text.strip():
            return []

        # 3. Section-aware grouping
        section_texts = group_lines_by_section(full_text)

        # 4. Build one Document per non-empty section
        effective_proposal_id = proposal_id or self.project_id
        section_docs = [
            Document(
                page_content=text,
                metadata={
                    "section": section,
                    "proposal_id": effective_proposal_id,
                    "source_file": file_name,
                    "section_weight": SECTION_WEIGHTS.get(section, 0.4),
                },
            )
            for section, text in section_texts.items()
            if text.strip()
        ]
        if not section_docs:
            return []

        # 5. Chunk within sections
        chunks = chunk_section_docs(section_docs, chunk_size, overlap_size)
        sections = [c.metadata["section"] for c in chunks]
        print("SECTION DISTRIBUTION:", Counter(sections))

        # 6. Enrich metadata
        enrich_chunk_metadata(chunks)

        print(f"[ProcessController] {file_name}: {len(chunks)} chunks from {len(section_docs)} sections")
        if chunks:
            print(f"  First chunk ({chunks[0].metadata['section']}): {chunks[0].page_content[:120]!r}")

        return chunks

    # =========================================================================
    # Proposal-level embedding aggregation (optional, Qdrant-ready)
    # =========================================================================

    def build_proposal_embedding(
        self,
        chunks: list[Document],
        embed_fn,           # Callable[[str], list[float]]
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Aggregate per-chunk embeddings into a single proposal-level vector
        using section-weight-based averaging.

        Parameters
        ----------
        chunks   : Output of process_file_bytes().
        embed_fn : Callable that takes a string → 1-D list/array of floats.
        normalize: L2-normalize the result (recommended for cosine similarity).

        Returns
        -------
        1-D numpy array, or None if chunks is empty.

        Example
        -------
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = controller.build_proposal_embedding(
            chunks, embed_fn=lambda t: model.encode(t).tolist()
        )
        qdrant_client.upsert("proposals", [PointStruct(id=pid, vector=vec.tolist())])
        """
        if not chunks:
            return None

        vectors, weights = [], []
        for chunk in chunks:
            weight = chunk.metadata.get("section_weight", 0.4)
            vec = np.asarray(embed_fn(chunk.page_content), dtype=np.float32)
            vectors.append(vec * weight)
            weights.append(weight)

        total_weight = sum(weights) or 1.0
        proposal_vec = np.sum(vectors, axis=0) / total_weight

        if normalize:
            norm = np.linalg.norm(proposal_vec)
            if norm > 0:
                proposal_vec = proposal_vec / norm

        return proposal_vec