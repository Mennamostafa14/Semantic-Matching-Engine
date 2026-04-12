from pydantic import BaseModel, Field
from typing import Optional


class IndexRequest(BaseModel):
    proposal_title: str = Field(
        ...,
        min_length=1,
        description="Human-readable title stored alongside the proposal vectors.",
    )
    chunk_size: Optional[int] = Field(
        512, ge=50, le=4096,
        description="Character length of each text chunk.",
    )
    overlap_size: Optional[int] = Field(
        50, ge=0, le=500,
        description="Overlap in characters between consecutive chunks.",
    )
    do_reset: Optional[bool] = Field(
        False,
        description="Wipe the entire proposals collection before indexing.",
    )


class AnalyzeRequest(BaseModel):
    proposal_text: str = Field(
        ...,
        min_length=10,
        description="Full text of the proposal to compare against stored proposals.",
    )
    limit: Optional[int] = Field(
        5, ge=1, le=20,
        description="Number of most-similar proposals to retrieve and analyse.",
    )