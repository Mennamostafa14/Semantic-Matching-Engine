
# src/controllers/NLPController
from .BaseController import BaseController
from .ProcessController import ProcessController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
from fastapi import UploadFile
from helpers.proposal_analysis import generate_similarity_analysis
from helpers.scoring import build_proposals, extract_keywords
from helpers.retrieval_utils import _mean_pool,_parse_hit
from fastapi.responses import JSONResponse

class NLPController(BaseController):
        # ثابتة (لو مش في settings)
    DEFAULT_LIMIT = 5
    DEFAULT_CHUNK_SIZE = 256
    DEFAULT_OVERLAP_SIZE = 40
    DEFAULT_SEARCH_LIMIT = 50
    DEFAULT_MIN_CHUNKS = 2
    DEFAULT_EXPLAIN_TOP_N = 3


    def __init__(self, vectordb_client, generation_client, 
                 embedding_client):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client


    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: manage items
        texts = [ c.page_content for c in chunks ]
        metadata = [ c.metadata for c in  chunks]
        vectors = []

        for text in texts:

            dense_vector = self.embedding_client.embed_text(
                text=text,
                document_type=DocumentTypeEnum.DOCUMENT.value
            )

            # sparse_vector = build_sparse_vector_from_keywords(text)
            vectors.append(dense_vector)
            # vectors.append({
            #     "dense": dense_vector,
            #     "keywords": sparse_vector
            # })

        # step3: create collection if not exists
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        collection_name = self.create_collection_name(project_id=project.project_id)

        vector = self.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.QUERY.value,
        )

        if not vector or len(vector) == 0:
            return False

        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit,
            query_text=text,          # ← hybrid search
        )

        if not results:
            return False

        return results
    
    async def compare_documents_scores_only(self, file: UploadFile):

        # ── 1. read file ─────────────────────
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse(status_code=400, content={"signal": "empty_file"})

        # ── 2. chunking ──────────────────────
        process_controller = ProcessController()
        chunks = process_controller.process_file_bytes(
            file_bytes=file_bytes,
            file_name=file.filename,
            proposal_id="compare_temp",
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            overlap_size=self.DEFAULT_OVERLAP_SIZE,
        )

        if not chunks:
            return JSONResponse(
                status_code=400,
                content={"signal": "extraction_failed",
                        "detail": "No text extracted or chunking failed"},
            )

        # ── 3. embed every chunk independently ───────────────────────────────
        texts = [chunk.page_content for chunk in chunks]

        chunk_vectors = self.embedding_client.embed_text(
            text=texts,
            document_type=DocumentTypeEnum.QUERY.value,
        )

        if not chunk_vectors:
            return JSONResponse(status_code=400, content={"signal": "embedding_error"})

        # ── 4. chunk-vs-chunk hybrid search ──────────────────────────────────
        all_raw_hits = []

        for vec, chunk_text in zip(chunk_vectors, texts):
            hits = self.vectordb_client.search_by_vector(
                collection_name="proposals",
                vector=vec,
                limit=10,
                query_text=chunk_text,
            )
            if hits:
                for h in hits:
                    pid = h.metadata.get("proposal_id") if h.metadata else None
                    if not pid:
                        continue
                    all_raw_hits.append({
                        "project_id": pid,
                        "section":    h.metadata.get("section", "unknown"),
                        "text":       h.text,
                        "score":      h.score,
                    })

        if not all_raw_hits:
            return {"file_name": file.filename, "results": []}

        # ── 5. keyword extraction ─────────────────────────────────────────────
        query_text     = " ".join(c.page_content for c in chunks)
        query_keywords = extract_keywords(query_text, top_n=20)

        # ── 6. build proposals ────────────────────────────────────────────────
        proposals = build_proposals(
            raw_hits=all_raw_hits,
            query_keywords=query_keywords,
            min_chunks=self.DEFAULT_MIN_CHUNKS,
            limit=self.DEFAULT_LIMIT,
        )

        if not proposals:
            return {"file_name": file.filename, "results": []}

        # ── 7. LLM analysis ───────────────────────────────────────────────────  ← جديد
        query_context = {
            "proposal_id": "query",
            "keywords":    {"overlap": query_keywords},
            "top_passages": [{"score": 100, "text": query_text[:500]}],
        }
        for i in range(min(self.DEFAULT_EXPLAIN_TOP_N, len(proposals))):
            result = generate_similarity_analysis(
                query_context,
                proposals[i],
                self.generation_client
            )
            proposals[i]["llm_analysis"] = result if result else None

        # ── 8. response ───────────────────────────────────────────────────────
        return {
            "file_name": file.filename,
            "results": [
                {
                    "proposal_id":   p["project_id"],
                    "overall_score": p["overall_score"],
                    "llm_analysis": {
                        "similarity_score": p["overall_score"],  # ← من الـ vector مش من الـ LLM
                        "explanation":      p.get("llm_analysis", {}).get("explanation"),
                        "key_similarities": p.get("llm_analysis", {}).get("key_similarities", []),
                    } if p.get("llm_analysis") else None,
                }
                for p in proposals
            ],
        }
    # async def compare_documents_scores_only(self, file: UploadFile):

    #     # ── 1. read file ─────────────────────
    #     file_bytes = await file.read()
    #     if not file_bytes:
    #         return JSONResponse(status_code=400, content={"signal": "empty_file"})

    #     # ── 2. chunking ──────────────────────
    #     process_controller = ProcessController()
    #     chunks = process_controller.process_file_bytes(
    #         file_bytes=file_bytes,
    #         file_name=file.filename,
    #         proposal_id="compare_temp",
    #         chunk_size=self.DEFAULT_CHUNK_SIZE,
    #         overlap_size=self.DEFAULT_OVERLAP_SIZE,
    #     )

    #     if not chunks:
    #         return JSONResponse(
    #             status_code=400,
    #             content={"signal": "extraction_failed",
    #                      "detail": "No text extracted or chunking failed"},
    #         )

    #     # ── 3. embed every chunk independently ───────────────────────────────
    #     texts = [chunk.page_content for chunk in chunks]

    #     chunk_vectors = self.embedding_client.embed_text(
    #         text=texts,
    #         document_type=DocumentTypeEnum.QUERY.value,
    #     )

    #     if not chunk_vectors:
    #         return JSONResponse(status_code=400, content={"signal": "embedding_error"})

    #     # ── 4. chunk-vs-chunk hybrid search ──────────────────────────────────
    #     all_raw_hits = []

    #     for vec, chunk_text in zip(chunk_vectors, texts):
    #         hits = self.vectordb_client.search_by_vector(
    #             collection_name="proposals",
    #             vector=vec,
    #             limit=10,
    #             query_text=chunk_text,     # ← sparse leg uses chunk text
    #         )
    #         if hits:
    #             for h in hits:
    #                 pid = h.metadata.get("proposal_id") if h.metadata else None
    #                 if not pid:
    #                     continue
    #                 all_raw_hits.append({
    #                     "project_id": pid,
    #                     "section":    h.metadata.get("section", "unknown"),
    #                     "text":       h.text,
    #                     "score":      h.score,
    #                 })

    #     if not all_raw_hits:
    #         return {"file_name": file.filename, "results": []}

    #     # ── 5. keyword extraction ─────────────────────────────────────────────
    #     query_text     = " ".join(c.page_content for c in chunks)
    #     query_keywords = extract_keywords(query_text, top_n=20)

    #     # ── 6. build proposals ────────────────────────────────────────────────
    #     proposals = build_proposals(
    #         raw_hits=all_raw_hits,
    #         query_keywords=query_keywords,
    #         min_chunks=self.DEFAULT_MIN_CHUNKS,
    #         limit=self.DEFAULT_LIMIT,
    #     )

    #     if not proposals:
    #         return {"file_name": file.filename, "results": []}

    #     # ── 7. response ───────────────────────────────────────────────────────
    #     print("NEW CHUNK-VS-CHUNK VERSION")
    #     for p in proposals[:5]:
    #         print(
    #             f"Proposal={p['project_id']}, "
    #             f"Score={p['overall_score']}, "
    #             f"Chunks={p.get('matched_chunks')},"
                
    #         )
    #         print("Overall:", p["overall_score"])
    #         print("Keywords:", p.get("keyword_overlap"))
    #         print("Importance:", p.get("importance_bonus"))
    #         print("Sections:", p.get("sections"))
    #     return {
    #         "file_name": file.filename,
    #         "results": [
    #             {
    #                 "proposal_id":      p["project_id"],
    #                 "overall_score":    p["overall_score"],
    #                 "matched_sections": p["sections"],
    #                 "keywords_overlap": p["keywords"]["overlap"],
    #                 "top_passages":     p["top_passages"],
    #             }
    #             for p in proposals
    #         ],
    #     }

    # async def compare_documents(self, file: UploadFile):

    #     # ── 1. read file ─────────────────────
    #     file_bytes = await file.read()
    #     if not file_bytes:
    #         return JSONResponse(status_code=400, content={"signal": "empty_file"})

    #     # ── 2. chunking ──────────────────────
    #     process_controller = ProcessController()
    #     chunks = process_controller.process_file_bytes(
    #         file_bytes=file_bytes,
    #         file_name=file.filename,
    #         proposal_id="compare_temp",
    #         chunk_size=self.DEFAULT_CHUNK_SIZE,
    #         overlap_size=self.DEFAULT_OVERLAP_SIZE,
    #     )

    #     if not chunks:
    #         return JSONResponse(
    #             status_code=400,
    #             content={"signal": "extraction_failed", "detail": "No text extracted or chunking failed"},
    #         )

    #     # ── 3. embedding ─────────────────────
    #     chunk_vectors: list[list[float]] = []
    #     for chunk in chunks:
    #         vec = self.embedding_client.embed_text(
    #             text=chunk.page_content,
    #             document_type=DocumentTypeEnum.QUERY.value,
    #         )
    #         if vec:
    #             chunk_vectors.append(vec)

    #     if not chunk_vectors:
    #         return JSONResponse(status_code=400, content={"signal": "embedding_error"})


    #     # ── 4. search ────────────────────────
    #     query_vector = _mean_pool(chunk_vectors)

    #     raw_hits = self.vectordb_client.search_by_vector(
    #         collection_name="proposals",
    #         vector=query_vector,
    #         limit=self.DEFAULT_SEARCH_LIMIT
    #     )

    #     if not raw_hits:
    #         return None

    #     # ── 5. build proposals ───────────────
    #     query_text = " ".join(c.page_content for c in chunks)
    #     query_keywords = extract_keywords(query_text, top_n=20)
    #     raw_hits = [
    #         {
    #             "project_id": h.metadata.get("proposal_id"),
    #             "section": getattr(h, "section", "unknown"),
    #             "text": h.text,
    #             "score": h.score,
    #         }
    #         for h in raw_hits
    #     ]
    #     proposals = build_proposals(
    #         raw_hits=raw_hits,
    #         query_keywords=query_keywords,
    #         min_chunks=self.DEFAULT_MIN_CHUNKS,
    #         limit=self.DEFAULT_LIMIT,
    #     )

    #     if not proposals:
    #         return None

    #     # ── 6. LLM (mandatory) ───────────────
    #     query_context = {
    #         "proposal_id": "query",
    #         "keywords": {
    #             "overlap": query_keywords
    #         },
    #         "text": query_text[:1000],
    #     }

    #     for i in range(min(self.DEFAULT_EXPLAIN_TOP_N, len(proposals))):
    #         result = generate_similarity_analysis(
    #             query_context,
    #             proposals[i],
    #             self.generation_client
    #         )

    #         proposals[i]["llm_analysis"] = (
    #             result if result else None
    #         )

    #     # ── 7. response ───────────────────────────────────────────────────────
    #     return {
    #         "file_name": file.filename,
    #         "results": [
    #             {
    #                 "proposal_id":      p["project_id"],
    #                 "vector_score":     p["overall_score"],
    #                 "matched_sections": list(p["sections"].keys()),
    #                 "keywords_overlap": p["keywords"]["overlap"],
    #                 "top_passages":     p["top_passages"],
    #                 "llm_analysis": {
    #                     "similarity_score": p.get("llm_analysis", {}).get("similarity_score"),
    #                     "explanation":      p.get("llm_analysis", {}).get("explanation"),
    #                     "key_similarities": p.get("llm_analysis", {}).get("key_similarities", []),
    #                 } if p.get("llm_analysis") else None,
    #             }
    #             for p in proposals
    #         ],
    #     }

    # async def compare_documents(self, file: UploadFile):

    #     # ── 1. read file ─────────────────────
    #     file_bytes = await file.read()
    #     if not file_bytes:
    #         return JSONResponse(status_code=400, content={"signal": "empty_file"})

    #     # ── 2. chunking ──────────────────────
    #     process_controller = ProcessController()
    #     chunks = process_controller.process_file_bytes(
    #         file_bytes=file_bytes,
    #         file_name=file.filename,
    #         proposal_id="compare_temp",
    #         chunk_size=self.DEFAULT_CHUNK_SIZE,
    #         overlap_size=self.DEFAULT_OVERLAP_SIZE,
    #     )

    #     if not chunks:
    #         return JSONResponse(
    #             status_code=400,
    #             content={"signal": "extraction_failed",
    #                      "detail": "No text extracted or chunking failed"},
    #         )

    #     # ── 3. embed every chunk independently ───────────────────────────────
    #     texts = [chunk.page_content for chunk in chunks]

    #     chunk_vectors = self.embedding_client.embed_text(
    #         text=texts,
    #         document_type=DocumentTypeEnum.QUERY.value,
    #     )

    #     if not chunk_vectors:
    #         return JSONResponse(status_code=400, content={"signal": "embedding_error"})

    #     # ── 4. chunk-vs-chunk hybrid search ──────────────────────────────────
    #     all_raw_hits = []

    #     for vec, chunk_text in zip(chunk_vectors, texts):
    #         hits = self.vectordb_client.search_by_vector(
    #             collection_name="proposals",
    #             vector=vec,
    #             limit=10,
    #             query_text=chunk_text,     # ← sparse leg uses chunk text
    #         )
    #         if hits:
    #             for h in hits:
    #                 pid = h.metadata.get("proposal_id") if h.metadata else None
    #                 if not pid:
    #                     continue
    #                 all_raw_hits.append({
    #                     "project_id": pid,
    #                     "section":    h.metadata.get("section", "unknown"),
    #                     "text":       h.text,
    #                     "score":      h.score,
    #                 })

    #     if not all_raw_hits:
    #         return {"file_name": file.filename, "results": []}

    #     # ── 5. keyword extraction ─────────────────────────────────────────────
    #     query_text     = " ".join(c.page_content for c in chunks)
    #     query_keywords = extract_keywords(query_text, top_n=20)

    #     # ── 6. build proposals ────────────────────────────────────────────────
    #     proposals = build_proposals(
    #         raw_hits=all_raw_hits,
    #         query_keywords=query_keywords,
    #         min_chunks=self.DEFAULT_MIN_CHUNKS,
    #         limit=self.DEFAULT_LIMIT,
    #     )

    #     if not proposals:
    #         return {"file_name": file.filename, "results": []}

    #     # ── 7. LLM analysis ───────────────────────────────────────────────────
    #     query_context = {
    #         "overall_score": 100,
    #         "sections": {},
    #         "keywords": {"overlap": query_keywords},
    #         "top_passages": [],
    #     }

    #     for i in range(min(self.DEFAULT_EXPLAIN_TOP_N, len(proposals))):
    #         result = generate_similarity_analysis(
    #             query_context,
    #             proposals[i],
    #             self.generation_client,
    #         )
    #         proposals[i]["llm_analysis"] = result if result else None

    #     # ── 8. response ───────────────────────────────────────────────────────
        
    #     return {
    #         "file_name": file.filename,
    #         "results": [
    #             {
    #                 "proposal_id":      p["project_id"],
    #                 "similarity_score": p.get("llm_analysis", {}).get("similarity_score") if p.get("llm_analysis") else None,
    #                 "explanation":      p.get("llm_analysis", {}).get("explanation") if p.get("llm_analysis") else None,
    #                 "key_similarities": p.get("llm_analysis", {}).get("key_similarities", []) if p.get("llm_analysis") else [],
    #                 "vector_score":     p["overall_score"],
    #                 "keywords_overlap": p["keywords"]["overlap"],
    #                 "top_passages":     p["top_passages"],
    #             }
    #             for p in proposals
    #         ],
    #     }