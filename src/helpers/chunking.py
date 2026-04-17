# helpers/chunking.py
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_section_docs(
    section_docs: list[Document],
    chunk_size: int,
    overlap_size: int,
) -> list[Document]:
    """
    Chunk each section Document independently using RecursiveCharacterTextSplitter.

    Splitting per-section (rather than on the full text) guarantees that no
    chunk ever spans two different sections, preserving semantic boundaries.

    Returns a flat list of Document chunks with all source metadata intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    for doc in section_docs:
        all_chunks.extend(splitter.split_documents([doc]))

    return all_chunks