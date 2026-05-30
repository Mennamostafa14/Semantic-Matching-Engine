from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List
from models.db_schemes import RetrievedDocument


class QdrantDBProvider(VectorDBInterface):

    SPARSE_VECTOR_NAME = "keywords"

    def __init__(self, db_path: str, distance_method: str):
        self.client = None
        self.db_path = db_path
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)

    def list_all_collections(self) -> List:
        return self.client.get_collections()

    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)

    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            return self.client.delete_collection(collection_name=collection_name)

    def create_collection(self, collection_name: str,
                          embedding_size: int,
                          do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)

        if not self.is_collection_existed(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method,
                ),
                # ── sparse vector config for keyword-based search ──
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: models.SparseVectorParams()
                },
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Sparse vector helper
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sparse_vector(text: str) -> models.SparseVector:
        """
        Build a simple TF sparse vector from raw text.
        - Tokenise (alpha, length ≥ 3, lowercase)
        - Count term frequencies
        - Use a stable hash as the sparse index
        """
        import re, math
        from collections import Counter

        tokens = re.findall(r"[a-zA-Z\u0600-\u06FF]{3,}", text.lower())
        tf = Counter(tokens)
        if not tf:
            return models.SparseVector(indices=[], values=[])

        total = sum(tf.values())
        indices, values = [], []
        for term, count in tf.items():
            idx = abs(hash(term)) % (2 ** 24)   # stable 24-bit index
            indices.append(idx)
            values.append(round(count / total, 6))

        return models.SparseVector(indices=indices, values=values)

    # ------------------------------------------------------------------
    # insert_many  (dense + sparse)
    # ------------------------------------------------------------------
    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadata: list = None,
                    record_ids: list = None, batch_size: int = 50):

        if metadata is None:
            metadata = [None] * len(texts)
        if record_ids is None:
            record_ids = list(range(len(texts)))

        for i in range(0, len(texts), batch_size):
            end = i + batch_size
            b_texts     = texts[i:end]
            b_vectors   = vectors[i:end]
            b_metadata  = metadata[i:end]
            b_ids       = record_ids[i:end]

            points = []
            for x in range(len(b_texts)):
                sparse_vec = self._build_sparse_vector(b_texts[x])
                points.append(
                    models.PointStruct(
                        id=b_ids[x],
                        vector={
                            "": b_vectors[x],                          # dense
                            self.SPARSE_VECTOR_NAME: sparse_vec,       # sparse
                        },
                        payload={
                            "text": b_texts[x],
                            **(b_metadata[x] or {}),
                        },
                    )
                )

            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch: {e}")
                return False

        self.logger.info(f"Inserted {len(texts)} records into '{collection_name}'")
        return True

    # ------------------------------------------------------------------
    # insert_one  (dense + sparse)
    # ------------------------------------------------------------------
    def insert_one(self, collection_name: str, text: str, vector: list,
                   metadata: dict = None, record_id: str = None):

        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Collection not found: {collection_name}")
            return False

        sparse_vec = self._build_sparse_vector(text)
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=record_id,
                        vector={
                            "": vector,
                            self.SPARSE_VECTOR_NAME: sparse_vec,
                        },
                        payload={"text": text, **(metadata or {})},
                    )
                ],
                wait=True,
            )
        except Exception as e:
            self.logger.error(f"Error while inserting record: {e}")
            return False

        return True

    # ------------------------------------------------------------------
    # search_by_vector  — hybrid (dense + sparse) with RRF fusion
    # ------------------------------------------------------------------
    def search_by_vector(self, collection_name: str, vector: list,
                         limit: int = 5, query_text: str = None):
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).

        Parameters
        ----------
        vector      : dense embedding of the query
        limit       : number of results to return
        query_text  : raw query text used to build the sparse vector.
                      If None, falls back to dense-only search.
        """
        try:
            if query_text:
                sparse_vec = self._build_sparse_vector(query_text)

                results = self.client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        # ── dense leg ──
                        models.Prefetch(
                            query=vector,
                            using="",
                            limit=limit * 4,
                        ),
                        # ── sparse leg ──
                        models.Prefetch(
                            query=models.SparseVector(
                                indices=sparse_vec.indices,
                                values=sparse_vec.values,
                            ),
                            using=self.SPARSE_VECTOR_NAME,
                            limit=limit * 4,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=limit,
                    with_payload=True,
                ).points

            else:
                # fallback: dense-only (old behaviour)
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=limit,
                    with_payload=True,
                )

            if not results:
                return None

            return [
                RetrievedDocument(
                    score=r.score,
                    text=r.payload.get("text"),
                    metadata={k: v for k, v in r.payload.items() if k != "text"},
                )
                for r in results
            ]

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return None