from pydantic import BaseModel,Field
from typing import Optional

class PushRequest(BaseModel):
    do_reset:Optional[bool] = False
    
class SearchRequest(BaseModel):
    text:str
    limit:Optional[int]=5

class ProposalRequest(BaseModel):
    proposal_id: str = Field(..., description="Stable unique identifier for this proposal")
    do_reset: bool = Field(False, description="Wipe the collection before inserting")