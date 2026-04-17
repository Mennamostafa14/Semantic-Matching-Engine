# helpers/metadata.py
from langchain.schema import Document


def enrich_chunk_metadata(chunks: list[Document]) -> None:
    """
    Add global positional metadata to every chunk **in-place**.

    Fields added / ensured:
    - chunk_index     : 0-based position in the final flat chunk list
    - total_chunks    : total number of chunks for this document
    - chunk_position  : relative position in [0.0, 1.0]; useful for
                        position-aware re-ranking (e.g. penalise tail chunks
                        that are often references/appendices)
    - section_weight  : propagated from the parent section Document so the
                        value survives the LangChain text splitter (which may
                        copy but sometimes drops custom metadata keys)
    """
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i,
            "total_chunks": total,
            "chunk_position": round(i / total, 4) if total > 1 else 0.0,
            "section_weight": chunk.metadata.get("section_weight", 0.4),
        })#