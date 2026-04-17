# helpers/__init__.py
# Public re-exports so callers can do:
#   from helpers import clean_text, detect_section, ...
from .text_cleaner import clean_text
from .config import Settings
from .section_detector import (
    SectionSpec,
    SECTION_REGISTRY,
    SECTION_WEIGHTS,
    DEFAULT_SECTION,
    detect_section,
    group_lines_by_section,
)
from .file_loader import load_from_bytes
from .chunking import chunk_section_docs
from .metadata import enrich_chunk_metadata

__all__ = [
    "clean_text",
    "SectionSpec",
    "SECTION_REGISTRY",
    "SECTION_WEIGHTS",
    "DEFAULT_SECTION",
    "detect_section",
    "group_lines_by_section",
    "load_from_bytes",
    "chunk_section_docs",
    "enrich_chunk_metadata",
]