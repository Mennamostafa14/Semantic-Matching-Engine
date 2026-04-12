from pydantic import BaseModel,Field,validator
from bson.objectid import ObjectId
from typing import Optional


class DataChunk(BaseModel):
    id: Optional[ObjectId]=Field(None,alias="_id")
    chunk_text:str = Field(...,min_length=1)
    chunk_metadata:dict
    chunk_order:int =Field(...,gt=0)
    chunk_project_id:ObjectId
    chunk_asset_id:ObjectId

    # to ignore ObjectId error
    class Config:
        arbitrary_types_allowed=True
    
    @classmethod
    def get_index(cls):
        return [
            {
                "key":[
                    ("chunk_project_id",1)
                ],
                "name":"chunk_project_id_index_1",
                "unique":False
            }
        ]

# schema for the rerieved data from database after semantic search
class RetrievedDocument(BaseModel):
    text:str
    score:float