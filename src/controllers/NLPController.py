
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
    DEFAULT_CHUNK_SIZE = 300
    DEFAULT_OVERLAP_SIZE = 50
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
        vectors = [
            self.embedding_client.embed_text(text=text, 
                                             document_type=DocumentTypeEnum.DOCUMENT.value)
            for text in texts
        ]

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

        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: get text embedding vector
        vector = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        # step3: do semantic search
        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        if not results:
            return False

        return results
    
    def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        
        answer, full_prompt, chat_history = None, None, None

        # step1: retrieve related documents
        retrieved_documents = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc.text,
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        # step3: Construct Generation Client Prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history
    

    def search_all_collections(self, text: str, limit: int = 50):
        """
        Search across every collection in the vector DB.
        Returns a flat list of RetrievedDocument-like dicts
        with an extra 'project_id' key derived from the collection name.
        """
        collections = self.vectordb_client.list_all_collections()
        vector = self.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.QUERY.value
        )
        if not vector or len(vector) == 0:
            return None

        all_results = []
        for col in collections.collections:          # QdrantClient returns CollectionsResponse
            col_name = col.name
            results = self.vectordb_client.search_by_vector(
                collection_name=col_name,
                vector=vector,
                limit=limit,
            )
            if results:
                project_id = col_name.replace("collection_", "", 1)
                for r in results:
                    all_results.append({
                        "project_id": project_id,
                        "text": r.text,
                        "score": r.score,
                    })
        return all_results
    
    async def compare_documents(self, file: UploadFile):

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
                content={"signal": "extraction_failed", "detail": "No text extracted or chunking failed"},
            )

        # ── 3. embedding ─────────────────────
        chunk_vectors: list[list[float]] = []
        for chunk in chunks:
            vec = self.embedding_client.embed_text(
                text=chunk.page_content,
                document_type=DocumentTypeEnum.QUERY.value,
            )
            if vec:
                chunk_vectors.append(vec)

        if not chunk_vectors:
            return JSONResponse(status_code=400, content={"signal": "embedding_error"})


        # ── 4. search ────────────────────────
        query_vector = _mean_pool(chunk_vectors)

        raw_hits = self.vectordb_client.search_by_vector(
            collection_name="proposals",
            vector=query_vector,
            limit=self.DEFAULT_SEARCH_LIMIT
        )

        if not raw_hits:
            return None

        # ── 5. build proposals ───────────────
        query_text = " ".join(c.page_content for c in chunks)
        query_keywords = extract_keywords(query_text, top_n=20)
        raw_hits = [
            {
                "project_id": h.metadata.get("proposal_id"),
                "section": getattr(h, "section", "unknown"),
                "text": h.text,
                "score": h.score,
            }
            for h in raw_hits
        ]
        proposals = build_proposals(
            raw_hits=raw_hits,
            query_keywords=query_keywords,
            min_chunks=self.DEFAULT_MIN_CHUNKS,
            limit=self.DEFAULT_LIMIT,
        )

        if not proposals:
            return None

        # ── 6. LLM (mandatory) ───────────────
        query_context = {
            "proposal_id": "query",
            "keywords": {
                "overlap": query_keywords
            },
            "text": query_text[:1000],
        }

        for i in range(min(self.DEFAULT_EXPLAIN_TOP_N, len(proposals))):
            result = generate_similarity_analysis(
                query_context,
                proposals[i],
                self.generation_client
            )

            proposals[i]["llm_analysis"] = (
                result.get("analysis") if result else None
            )

        # ── 7. response ──────────────────────
        return {
            "file_name": file.filename,
            "results": [
                {
                    "proposal_id": p["project_id"],
                    "llm_analysis": p.get("llm_analysis")
                }
                for p in proposals
                if p.get("llm_analysis")
            ]
        }