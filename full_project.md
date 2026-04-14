# Project Export: D:\AI-Projects\Semantic-Matching-Engine\src

================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\.env
================================================================================

``` 
APP_NAME="semantic-match"
APP_VERSION="0.1"


FILE_ALLOWED_TYPES=["text/plain","application/pdf"]
FILE_MAX_SIZE=10

FILE_DEFAULT_CHUNK_SIZE=512000 #3512KB

MONGODB_URL="mongodb://menna:menna@localhost:27008"
MONGODB_DATABASE= "semantic-match"
# -----------------------------------------


#======================= LLM Config ==========================
GENERATION_BACKEND="GEMINI"
EMBEDDING_BACKEND="GEMINI"


GEMINI_API_KEY="AIzaSyATpdwxPe4XINRN1HejCNJGEwM4BdtVj4w"
GENERATION_MODEL_ID="gemini-1.5-flash"
EMBEDDING_MODEL_ID="gemini-embedding-2-preview"
EMBEDDING_MODEL_SIZE=3072

INPUT_DAFAULT_MAX_CHARACTERS=1024
GENERATION_DAFAULT_MAX_TOKENS=200
GENERATION_DAFAULT_TEMPERATURE=0.1

#======================= Vector DB Config ==========================
VECTOR_DB_BACKEND="QDRANT"
VECTOR_DB_PATH="qdrant_db"
VECTOR_DB_DISTANCE_METHOD="cosine"



``` 



================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\main.py
================================================================================

``` 
from fastapi import FastAPI
from routes import data, nlp
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory


app = FastAPI()

async def startup_span():
    settings = get_settings()
    # app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    # app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]

    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings)

    # generation client
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)
    
    # vector db client
    app.vectordb_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.vectordb_client.connect()



async def shutdown_span():
    # app.mongo_conn.close()
    app.vectordb_client.disconnect()

app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)


app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\requirements.txt
================================================================================

``` 
fastapi==0.110.2
uvicorn[standard]==0.29.0

python-multipart==0.0.9
python-dotenv==1.0.1
aiofiles==23.2.1


pydantic>=2.9.0,<3.0.0
pydantic-settings==2.2.1

# Gemini
google-genai==1.72.0

# Vector DB
qdrant-client==1.10.1

# File processing
PyMuPDF==1.24.3

# MongoDB
motor==3.4.0
pymongo==4.8.0  
pydantic-mongo==2.3.0

langchain-community>=0.0.10

numpy>=1.24
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\assets\.gitignore
================================================================================

``` 
files
database

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\assets\.gitkeep
================================================================================

``` 

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\assets\database\qdrant_db\.lock
================================================================================

``` 
tmp lock file
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\assets\database\qdrant_db\meta.json
================================================================================

``` 
{"collections": {"collection_1": {"vectors": {"size": 3072, "distance": "Cosine", "hnsw_config": null, "quantization_config": null, "on_disk": null, "datatype": null, "multivector_config": null}, "shard_number": null, "sharding_method": null, "replication_factor": null, "write_consistency_factor": null, "on_disk_payload": null, "hnsw_config": null, "wal_config": null, "optimizers_config": null, "init_from": null, "quantization_config": null, "sparse_vectors": null}, "proposals": {"vectors": {"size": 3072, "distance": "Cosine", "hnsw_config": null, "quantization_config": null, "on_disk": null, "datatype": null, "multivector_config": null}, "shard_number": null, "sharding_method": null, "replication_factor": null, "write_consistency_factor": null, "on_disk_payload": null, "hnsw_config": null, "wal_config": null, "optimizers_config": null, "init_from": null, "quantization_config": null, "sparse_vectors": null}}, "aliases": {}}
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\controllers\BaseController.py
================================================================================

``` 
from helpers.config import get_settings, Settings
import os
import random
import string

class BaseController:
    
    def __init__(self):

        self.app_settings = get_settings()
        
        self.base_dir = os.path.dirname( os.path.dirname(__file__) )
        self.files_dir = os.path.join(
            self.base_dir,
            "assets/files"
        )

        self.database_dir=os.path.join(
            self.base_dir,
            "assets/database"
        )
        
    def generate_random_string(self, length: int=12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    def get_database_path(self,db_name:str):
        database_path=os.path.join(
            self.database_dir,db_name
        )
        if not os.path.exists(database_path):
            os.makedirs(database_path)
        return database_path
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\controllers\DataController.py
================================================================================

``` 
from .BaseController import BaseController
from .ProjectController import ProjectController
from fastapi import UploadFile
from models import ResponseSignal
import re
import os

class DataController(BaseController):
    
    def __init__(self):
        super().__init__()
        self.size_scale = 1048576 # convert MB to bytes

    def validate_uploaded_file(self, file: UploadFile):

        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        if file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value

        return True, ResponseSignal.FILE_VALIDATED_SUCCESS.value

    


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\controllers\NLPController.py
================================================================================

``` 
from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json

class NLPController(BaseController):

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
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\controllers\ProcessController.py
================================================================================

``` 
# controllers/ProcessController.py
from .BaseController import BaseController
from .ProjectController import ProjectController
import os
import re
import tempfile
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from models import ProcessingEnum

class ProcessController(BaseController):

    def __init__(self, project_id: str = None):
        super().__init__()
        self.project_id = project_id



    def get_file_extension(self, file_id: str):
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
        if loader:
            return loader.load()
        return None

    
    def clean_text(self, text: str) -> str:
  
        # Remove lone page numbers
        text = re.sub(r'(?i)(page\s*\d+|\-\s*\d+\s*\-)', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        # Collapse runs of dashes/underscores
        text = re.sub(r'[-_]{3,}', ' ', text)
        # Replace non-breaking spaces
        text = re.sub(r'[\xa0\u2000-\u200f\u2028\u2029]+', ' ', text)
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple spaces/tabs
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Strip each line
        lines = [line.strip() for line in text.splitlines()]
        # Drop noise lines
        lines = [
            line for line in lines
            if len(line) > 2 and not re.fullmatch(r'[\d\s\W]+', line)
        ]
        return '\n'.join(lines).strip()

    def process_file_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        proposal_id: str = None,     
        chunk_size: int = 300,
        overlap_size: int = 50,
    ) -> list[Document]:
        
        file_ext = os.path.splitext(file_name)[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            if file_ext == '.txt':
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif file_ext == '.pdf':
                loader = PyMuPDFLoader(tmp_path)
            else:
                return []  

            documents = loader.load()
        finally:

            os.unlink(tmp_path)

        if not documents:
            return []


        cleaned_docs = []
        for doc in documents:
            cleaned = self.clean_text(doc.page_content)
            if cleaned:
                cleaned_docs.append(
                    Document(
                        page_content=cleaned,
                        metadata=doc.metadata.copy()  
                    )
                )

        if not cleaned_docs:
            return []


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        chunks = text_splitter.split_documents(cleaned_docs)


        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "proposal_id": proposal_id if proposal_id else self.project_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_file": file_name,
            })

        return chunks
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\controllers\ProjectController.py
================================================================================

``` 
from .BaseController import BaseController
from fastapi import UploadFile
from models import ResponseSignal
import os

class ProjectController(BaseController):
    
    def __init__(self):
        super().__init__()

    def get_project_path(self, project_id: str):
        project_dir = os.path.join(
            self.files_dir,
            project_id
        )

        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        return project_dir

    
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\controllers\__init__.py
================================================================================

``` 
from .DataController import DataController
from .ProjectController import ProjectController
from .ProcessController import ProcessController
from .NLPController import NLPController
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\helpers\config.py
================================================================================

``` 
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str

    GEMINI_API_KEY:str

    
    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    MONGODB_URL: str
    MONGODB_DATABASE: str

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str



    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None
    INPUT_DAFAULT_MAX_CHARACTERS: int = None
    GENERATION_DAFAULT_MAX_TOKENS: int = None
    GENERATION_DAFAULT_TEMPERATURE: float = None

    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: str = None


    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\helpers\__init__.py
================================================================================

``` 

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\AssetModel.py
================================================================================

``` 
from .BaseDataModel import BaseDataModel
from .db_schemes import Asset
from .enums.DataBaseEnum import DataBaseEnum
from bson import ObjectId

class AssetModel(BaseDataModel):

    def __init__(self, db_client):
        super().__init__(db_client=db_client)
        self.collection=self.db_client[DataBaseEnum.COLLECTION_ASSET_NAME.value]

    @classmethod
    async def create_instance(cls,db_client:object):
        instance=cls(db_client)
        await instance.init_collection()
        return instance
    
    async def init_collection(self):
        all_collections= await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_ASSET_NAME.value not in all_collections:
            self.collection=self.db_client[DataBaseEnum.COLLECTION_ASSET_NAME.value]
            indexes=Asset.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"]
                )
    async def create_asset(self,asset:Asset):
        result= await self.collection.insert_one(asset.dict(by_alias=True,exclude_unset=True))
        asset.id= result.inserted_id
        return asset
    
    async def get_all_project_assets(self,asset_project_id:str,asset_type:str):
        records= await self.collection.find({
            "asset_project_id":ObjectId(asset_project_id) if isinstance(asset_project_id,str) else asset_project_id,
            "asset_type": asset_type,
        }).to_list(length=None)

        return [
            Asset(**record)
            for record in records
        ]
    
    async def get_asset_record(self,asset_project_id:str,asset_name:str):
        record= await self.collection.find_one({
            "asset_project_id":ObjectId(asset_project_id) if isinstance(asset_project_id,str) else asset_project_id,
            "asset_name":asset_name,
        })

        if record:
            return Asset(**record)
        return None
 

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\BaseDataModel.py
================================================================================

``` 
from helpers.config import get_settings,Settings

class BaseDataModel:
    def __init__(self,db_client:object):
        self.db_client=db_client
        self.app_settings=get_settings()
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\ChunkModel.py
================================================================================

``` 
from .BaseDataModel import BaseDataModel
from .db_schemes import DataChunk
from .enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId
from pymongo import InsertOne

class ChunkModel(BaseDataModel):
    def __init__(self, db_client:object):
        super().__init__(db_client=db_client)
        self.collection=self.db_client[DataBaseEnum.COLLECTION_CHUNK_NAME.value]
    
    @classmethod
    async def create_instance(cls,db_client:object):
        instance=cls(db_client)
        await instance.init_collection()
        return instance
    
    async def init_collection(self):
        all_collections= await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_PROJECT_NAME.value not in all_collections:
            self.collection=self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]
            indexes=DataChunk.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"]
                )
    async def create_chunk(self,chunk:DataChunk):
        result= await self.collection.insert_one(chunk.dict(by_alias=True,exclude_unset=True))
        chunk._id=result.inserted_id
        return chunk
    
    async def get_chunk(self,chunk_id:str):
        result= await self.collection.find_one({
            "_id":ObjectId(chunk_id)
        })
        if result is None:
            return None
        return DataChunk(**result)
    
    async def insert_many_chunks(self,chunks:list,batch_size:int=100):
        for i in range(0,len(chunks),batch_size):
            batch= chunks[i:i+batch_size]
            operations=[
                InsertOne(chunk.dict(by_alias=True,exclude_unset=True))
                for chunk in batch
            ]
            await self.collection.bulk_write(operations)
        return len(chunks)

    async def delete_chunks_by_project_id(self,project_id:ObjectId):
        result=await self.collection.delete_many({
            "chunk_project_id":project_id
        })
        return result.deleted_count

    
    async def get_poject_chunks(self, project_id: ObjectId, page_no: int=1, page_size: int=50):
        records = await self.collection.find({
                    "chunk_project_id": project_id
                }).skip(
                    (page_no-1) * page_size
                ).limit(page_size).to_list(length=None)

        return [
            DataChunk(**record)
            for record in records
        ]
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\ProjectModel.py
================================================================================

``` 
from .BaseDataModel import BaseDataModel
from .db_schemes import Project
from .enums.DataBaseEnum import DataBaseEnum


class ProjectModel(BaseDataModel):

    def __init__(self, db_client):
        super().__init__(db_client=db_client)
        self.collection=self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]
    
    @classmethod
    async def create_instance(cls,db_client:object):
        instance=cls(db_client)
        await instance.init_collection()
        return instance
    
    async def init_collection(self):
        all_collections= await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_PROJECT_NAME.value not in all_collections:
            self.collection=self.db_client[DataBaseEnum.COLLECTION_PROJECT_NAME.value]
            indexes=Project.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"]
                )

    async def create_project(self,project:Project):
        result= await self.collection.insert_one(project.dict(by_alias=True,exclude_unset=True))
        project.id=result.inserted_id
        return project
    
    async def get_project_or_create_one(self,project_id:str):
        record= await self.collection.find_one({
            "project_id": project_id
        })
        if record is None:
            # create new project
            project=Project(project_id=project_id)
            project= await self.create_project(project=project)
            return project
        return Project(**record)
    
    async def get_all_projects(self,page:int=1,page_size:int=10):
        # count total number of documents
        total_documents=await self.collection.count_documents({})

        # calculate total number of pages
        total_pages=total_documents//page_size
        if total_documents%page_size>0:
            total_pages+=1
        
        cursor= self.collection.find().skip((page-1) * page_size).limit(page_size)
        projects=[]
        async for document in cursor:
            projects.append(
                Project(**document)
            )
            return projects,total_pages



``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\__init__.py
================================================================================

``` 
from .enums.ResponseEnums import ResponseSignal
from .enums.ProcessingEnum import ProcessingEnum
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\db_schemes\asset.py
================================================================================

``` 
from pydantic import BaseModel,Field,validator
from typing import Optional
from bson.objectid import ObjectId
from datetime import datetime

class Asset(BaseModel):
    id: Optional[ObjectId]=Field(None,alias="_id")
    asset_project_id:ObjectId
    asset_type:str=Field(...,min_length=1)
    asset_name:str= Field(...,min_length=1)
    asset_size:int=Field(ge=0,default=None)
    asset_pushed_at:datetime=Field(default=datetime.utcnow)
    asset_config:dict=Field(default=None)
    
    # to ignore ObjectId error
    class Config:
        arbitrary_types_allowed=True
    
    @classmethod
    def get_indexes(cls):
        return [
            {
                "key":[
                    ("asset_project_id",1) 
                ],
                "name":"asset_project_id_index_1",
                "unique":False
            },
            {
                "key":[
                    ("asset_project_id",1), 
                    ("asset_name",1)
                ],
                "name":"asset_project_id_name_index_1",
                "unique":True
            } 
        ]

   
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\db_schemes\data_chunk.py
================================================================================

``` 
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
    metadata:Optional[dict]=None
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\db_schemes\project.py
================================================================================

``` 
from pydantic import BaseModel,Field,validator
from typing import Optional
from bson.objectid import ObjectId
class Project(BaseModel):
    id: Optional[ObjectId]=Field(None,alias="_id")
    project_id:str = Field(...,min_length=1)

    @validator('project_id')
    def validate_project_id(cls,value):
        if not value.isalnum():
            raise ValueError('project_id must be alphanumeric')
        return value
    
    # to ignore ObjectId error
    class Config:
        arbitrary_types_allowed=True
    
    @classmethod
    def get_indexes(cls):
        return [
            {
                "key":[
                    ("project_id",1) 
                ],
                "name":"project_id_index_1",
                "unique":True
            }
        ]


``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\db_schemes\__init__.py
================================================================================

``` 
from .project import Project
from .data_chunk import DataChunk,RetrievedDocument
from .asset import Asset
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\enums\AssetTypeEnum.py
================================================================================

``` 
from enum import Enum

class AssetTypeEnum(Enum):
    FILE="file"
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\enums\DataBaseEnum.py
================================================================================

``` 
from enum import Enum

class DataBaseEnum(Enum):

    COLLECTION_PROJECT_NAME="projects"
    COLLECTION_CHUNK_NAME="chunks"
    COLLECTION_ASSET_NAME="assets"

    
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\enums\ProcessingEnum.py
================================================================================

``` 
from enum import Enum

class ProcessingEnum(Enum):

    TXT = ".txt"
    PDF = ".pdf"
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\enums\ResponseEnums.py
================================================================================

``` 
from enum import Enum
 
 
class ResponseSignal(Enum):
 
    # File handling
    FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_VALIDATED_SUCCESS = "file_validated_success"
    FILE_UPLOAD_SUCCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"
 
    # Processing
    PROCESSING_SUCCESS = "processing_success"
    PROCESSING_FAILED = "processing_failed"
    FILE_ID_ERROR = "no_file_found_with_this_id"
 
    # Vector DB
    INSERT_INTO_VECTORDB_SUCCESS = "insert_into_vectordb_success"
    INSERT_INTO_VECTORDB_ERROR = "insert_into_vectordb_error"
    VECTORDB_SEARCH_ERROR = "vectordb_search_error"
    VECTORDB_SEARCH_SUCCESS = "vectordb_search_success"
 
    # Proposals
    PROPOSAL_INDEXED_SUCCESS = "proposal_indexed_success"
    PROPOSAL_INDEXED_ERROR = "proposal_indexed_error"
    PROPOSAL_NOT_FOUND = "proposal_not_found"
    PROPOSAL_ANALYSIS_SUCCESS = "proposal_analysis_success"
    PROPOSAL_ANALYSIS_ERROR = "proposal_analysis_error"
    PROPOSAL_DELETED_SUCCESS = "proposal_deleted_success"
    PROPOSAL_LIST_SUCCESS = "proposal_list_success"
    PROPOSAL_SIMILARITY_ERROR="proposal_similarity_error"
    PROPOSAL_SIMILARITY_SUCESS="proposal_similarity_sucess"
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\models\enums\__init__.py
================================================================================

``` 
from .ResponseEnums import ResponseSignal
from .ProcessingEnum import ProcessingEnum
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\routes\base.py
================================================================================

``` 

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\routes\data.py
================================================================================

``` 
from fastapi import FastAPI, APIRouter, Depends, UploadFile, status,Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
import aiofiles
from models import ResponseSignal
import logging
from .schemes.data import ProcessRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.db_schemes import DataChunk,Asset
from models.AssetModel import AssetModel
from models.enums.AssetTypeEnum import AssetTypeEnum

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1", "data"],
)

@data_router.post("/upload/{project_id}")
async def upload_data(request:Request,project_id: str, file: UploadFile,
                      app_settings: Settings = Depends(get_settings)):
    project_model= await ProjectModel.create_instance(
        db_client=request.app.db_client 
    )    
    
    project=await project_model.get_project_or_create_one(
        project_id=project_id
    )
    # validate the file properties
    data_controller = DataController()

    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error while uploading file: {e}")

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )
    # store the assets into database
    asset_model= await AssetModel.create_instance(
        db_client=request.app.db_client
    )
    asset_resource=Asset(
        asset_project_id=project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path)
 )
    asset_record=await asset_model.create_asset(asset=asset_resource)

    return JSONResponse(
            content={
                "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
                "file_id": str(asset_record.id)
            }
        )

@data_router.post("/process/{project_id}")
async def process_endpoint(request:Request,project_id: str, process_request: ProcessRequest):

    
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset= process_request.do_reset

    project_model= await ProjectModel.create_instance(
        db_client=request.app.db_client 
    )    
    
    project=await project_model.get_project_or_create_one(
        project_id=project_id
    )

    asset_model= await AssetModel.create_instance(
        db_client=request.app.db_client
    )

    project_files_ids={}
    if process_request.file_id:
        asset_record= await asset_model.get_asset_record(
            asset_project_id=project_id,
            asset_name=process_request.file_id
        )
        if asset_record is None:
            return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal":ResponseSignal.FILE_ID_ERROR.value
        })
        project_files_ids={
            asset_record.id:asset_record.asset_name
        }

    else:
        project_files= await asset_model.get_all_project_assets(
            asset_project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value,
        )
        project_files_ids={
            record.id: record.asset_name
            for record in project_files
        }
    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal":ResponseSignal.NO_FILES_ERROR.value
        }
    )       
    process_controller = ProcessController(project_id=project_id)
    
    no_records=0
    no_files=0

    chunk_model=await ChunkModel.create_instance(
        db_client=request.app.db_client
    )
    
    if do_reset==1:
        _= await chunk_model.delete_chunks_by_project_id(
            project_id=project.id
        )
    for asset_id,file_id in project_files_ids.items():
        file_content = process_controller.get_file_content(file_id=file_id)
        if file_content is None:
            logger.error(f"Error while processing file: {file_id}")
            continue

        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_id,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )

        if file_chunks is None or len(file_chunks) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.PROCESSING_FAILED.value
                }
            )

        file_chunks_records=[
            DataChunk(
            chunk_text= chunk.page_content,
            chunk_metadata=chunk.metadata,
            chunk_order=i+1,
            chunk_project_id=project.id,
            chunk_asset_id=asset_id
            )
            for i, chunk in enumerate(file_chunks)
        ]
        chunk_model=await ChunkModel.create_instance(
            db_client=request.app.db_client
        )

        if do_reset==1:
            _= await chunk_model.delete_chunks_by_project_id(
                project_id=project.id
            )


        no_records+=await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files+=1
    return JSONResponse(
        content={
            "signal":ResponseSignal.PROCESSING_SUCCESS.value,
            "inserted_chunks":no_records,
            "processed_files":no_files
        }
    )

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\routes\nlp.py
================================================================================

``` 
from fastapi import FastAPI, APIRouter, status, Request,UploadFile,Form, File
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest, ProposalRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController,ProcessController
from models import ResponseSignal
import os
import logging
from collections import defaultdict
import numpy as np

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

# ── existing endpoint (unchanged) ──────────────────────────────────────────────

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
        )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(
            project_id=project.id, page_no=page_no
        )
        if len(page_chunks):
            page_no += 1

        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids = list(range(idx, idx + len(page_chunks)))
        idx += len(page_chunks)

        is_inserted = nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            do_reset=push_request.do_reset,
            chunks_ids=chunks_ids,
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value}
            )

        inserted_items_count += len(page_chunks)

    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count,
        }
    )


# ── NEW: push a single proposal into the vector DB ────────────────────────────

@nlp_router.post("/index/push-proposal")
async def push_proposal(
    request: Request,
    proposal_id: str = Form(...),
    do_reset: bool = Form(False),
    file: UploadFile = File(...)
):
    process_controller = ProcessController(project_id="temp")

    # 1. save file
    # file_path = os.path.join(process_controller.project_path, file.filename)
    # with open(file_path, "wb") as f:
    #     f.write(await file.read())

    # # 2. load
    # file_content = process_controller.get_file_content(file_id=file.filename)

    # if not file_content:
    #     return JSONResponse(
    #         status_code=422,
    #         content={"signal": "file_processing_failed"}
    #     )
    file_bytes = await file.read()

    # 3. chunk
    chunks = process_controller.process_file_bytes(
        file_bytes=file_bytes,
        file_name=file.filename,
        proposal_id=proposal_id,
        chunk_size=300,      # يمكن جعلها قابلة للتكوين من settings
        overlap_size=50,
    )

    if not chunks:
        return JSONResponse(
            status_code=422,
            content={"signal":ResponseSignal.PROCESSING_FAILED.value }
        )

    # 4. NLP
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    collection_name = "proposals"

    texts = [c.page_content for c in chunks]
    metadata = [c.metadata for c in chunks]

    vectors = [
        nlp_controller.embedding_client.embed_text(
            text=t,
            document_type="document"
        )
        for t in texts
    ]

    # create collection
    request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=nlp_controller.embedding_client.embedding_size,
        do_reset=do_reset,
    )

    # insert
    request.app.vectordb_client.insert_many(
        collection_name=collection_name,
        texts=texts,
        metadata=metadata,
        vectors=vectors,
        record_ids=list(range(len(texts)))
    )

    return {
        "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
        "proposal_id": proposal_id,
        "chunks": len(chunks)
    }

@nlp_router.post("/index/compare")
async def compare_proposal(
    request: Request,
    file: UploadFile = File(...),
    limit: int = Form(5),
    chunk_size: int = Form(512),
    overlap_size: int = Form(50),
    search_limit: int = Form(50),
):
    """
    Upload a PDF or TXT file. The endpoint:
      1. Extracts and cleans text using ProcessController.
      2. Splits into chunks using ProcessController.
      3. Embeds every chunk then mean-pools to a single document vector.
      4. Searches all Qdrant collections for the most similar chunks.
      5. Groups hits by project_id, computes blended score, returns top projects.
    """
 

    def _mean_pool(vectors: list[list[float]]) -> list[float]:
        arr = np.array(vectors, dtype=np.float32)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            return (mean / norm).tolist()
        return mean.tolist()
    # ── 1. read file ────────────────────────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        return JSONResponse(status_code=400, content={"signal": "empty_file"})

    # ── 2. use ProcessController to extract, clean, and chunk ────────────────
    process_controller = ProcessController()   
    chunks = process_controller.process_file_bytes(
        file_bytes=file_bytes,
        file_name=file.filename,
        proposal_id="compare_temp",   
        chunk_size=chunk_size,
        overlap_size=overlap_size,
    )

    if not chunks:
        return JSONResponse(
            status_code=400,
            content={"signal": "extraction_failed", "detail": "No text extracted or chunking failed"}
        )

    # ── 3. embed each chunk and mean-pool ───────────────────────────────────
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    vectors = []
    for chunk in chunks:
        vec = nlp_controller.embedding_client.embed_text(
            text=chunk.page_content,
            document_type="document", 
        )
        if vec:
            vectors.append(vec)

    if not vectors:
        return JSONResponse(status_code=400, content={"signal": "embedding_error"})

    # Mean pooling + L2 normalization
    query_vector = _mean_pool(vectors)   

    # ── 4. search across all collections ────────────────────────────────────
    all_cols = request.app.vectordb_client.list_all_collections()
    if not all_cols or not all_cols.collections:
        return JSONResponse(status_code=400, content={"signal": "no_collections_found"})

    raw_hits = []
    for col in all_cols.collections:
        col_name = col.name
        results = request.app.vectordb_client.search_by_vector(
            collection_name=col_name,
            vector=query_vector,
            limit=search_limit,
        )
        if results:
            project_id = col_name.replace("collection_", "", 1)
            for r in results:
                raw_hits.append({
                    "project_id": project_id,
                    "text": r.text,
                    "score": r.score,
                })

    if not raw_hits:
        return JSONResponse(status_code=400, content={"signal": "vectordb_search_error"})

    # ── 5. group by project_id and compute scores ───────────────────────────
    from collections import defaultdict
    bucket = defaultdict(list)
    texts_by_project = defaultdict(list)

    for hit in raw_hits:
        pid = hit["project_id"]
        bucket[pid].append(hit["score"])
        texts_by_project[pid].append(hit["text"])

    proposals = []
    for pid, scores in bucket.items():
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        final_score = 0.7 * max_score + 0.3 * avg_score

  
        top_passages = sorted(
            zip(scores, texts_by_project[pid]),
            key=lambda x: x[0],
            reverse=True
        )[:3]

        proposals.append({
            "project_id": pid,
            "avg_similarity": round(avg_score * 100, 2),
            "max_similarity": round(max_score * 100, 2),
            "final_score": round(final_score * 100, 2),
            "matched_chunks": len(scores),
            "top_passages": [
                {"score": round(s * 100, 2), "text": t}
                for s, t in top_passages
            ],
        })

    proposals.sort(key=lambda x: x["final_score"], reverse=True)
    proposals = proposals[:limit]

    # ── 6. summary ──────────────────────────────────────────────────────────
    final_scores = [p["final_score"] for p in proposals]
    summary = {
        "file_name": file.filename,
        "text_length": sum(len(c.page_content) for c in chunks),
        "chunks_embedded": len(vectors),
        "total_hits": len(raw_hits),
        "projects_found": len(proposals),
        "highest_similarity": max(final_scores) if final_scores else 0,
        "average_similarity": round(sum(final_scores) / len(final_scores), 2) if final_scores else 0,
    }

    return JSONResponse(content={
        "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
        "summary": summary,
        "proposals": proposals,
    })
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\routes\__init__.py
================================================================================

``` 

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\routes\schemes\data.py
================================================================================

``` 
from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str=None
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
    do_reset: Optional[int] = 0
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\routes\schemes\nlp.py
================================================================================

``` 
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

class CompareRequest(BaseModel):
    limit: Optional[int] = 5
    chunk_size: Optional[int] = 512
    overlap_size: Optional[int] = 50
    search_limit: Optional[int] = 50   # how many Qdrant results to pull before grouping
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\routes\schemes\__init__.py
================================================================================

``` 

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\llm\LLMEnums.py
================================================================================

``` 
from enum import Enum


class LLMEnums(Enum):
    GEMINI = "GEMINI"


class GeminiRoleEnum(Enum):
    USER = "user"
    MODEL = "model"


class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\llm\LLMInterface.py
================================================================================

``` 
from abc import ABC, abstractmethod
from typing import List

class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str):
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None):
        pass

    @abstractmethod
    def embed_text(self, text: str, document_type: str = None):
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str):
        pass


``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\llm\LLMProviderFactory.py
================================================================================

``` 
from .LLMEnums import LLMEnums
from .providers import GeminiProvider

class LLMProviderFactory:
    def __init__(self,config:dict):
        self.config=config

    def create(self,provider:str):
        if provider==LLMEnums.GEMINI.value:
            return GeminiProvider(
            api_key=self.config.GEMINI_API_KEY,
            default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
            default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
            default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
        )

        raise ValueError(f"Unsupported LLM provider: {provider}")
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\llm\__init__.py
================================================================================

``` 

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\llm\providers\GeminiProvider.py
================================================================================

``` 
from ..LLMInterface import LLMInterface
import logging
from google import genai


class GeminiProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1
    ):

        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = genai.Client(api_key=self.api_key)

        self.logger = logging.getLogger(__name__)

    # -------------------------
    # Model configuration
    # -------------------------
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int = None):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    # -------------------------
    # Text preprocessing
    # -------------------------
    def process_text(self, text: str):
        if not text:
            return ""
        return text[:self.default_input_max_characters].strip()

    # -------------------------
    # Generation (LLM)
    # -------------------------
    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None
    ):

        if not self.client:
            self.logger.error("Gemini client is not initialized")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model is not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        try:
            response = self.client.models.generate_content(
                model=self.generation_model_id,
                contents=self.process_text(prompt),
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens
                }
            )

            if not response or not response.text:
                self.logger.error("Empty response from Gemini generation")
                return None

            return response.text

        except Exception as e:
            self.logger.error(f"Gemini generation error: {e}")
            return None

    # -------------------------
    # Embeddings
    # -------------------------
    def embed_text(self, text: str, document_type: str = None):

        if not self.client:
            self.logger.error("Gemini client is not initialized")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model is not set")
            return None

        try:
            response = self.client.models.embed_content(
                model=self.embedding_model_id,
                contents=self.process_text(text)
            )

            if not response or not response.embeddings:
                self.logger.error("Empty embedding response from Gemini")
                return None

            return response.embeddings[0].values

        except Exception as e:
            self.logger.error(f"Gemini embedding error: {e}")
            return None

    # -------------------------
    # Prompt helper (optional)
    # -------------------------
    def construct_prompt(self, prompt: str, role: str = None):
        return self.process_text(prompt)
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\llm\providers\__init__.py
================================================================================

``` 
from .GeminiProvider import GeminiProvider
``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\vectordb\VectorDBEnums.py
================================================================================

``` 
from enum import Enum

class VectorDBEnums(Enum):
    QDRANT="QDRANT"

class DistanceMethodEnums(Enum):
    COSINE="cosine"
    DOT="dot"

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\vectordb\VectorDBInterface.py
================================================================================

``` 
from abc import ABC,abstractmethod
from typing import List
from models.db_schemes import RetrievedDocument

class VectorDBInterface(ABC):

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def is_collection_existed(self,collection_name:str) -> bool:
        pass

    @abstractmethod
    def list_all_collections(self) -> List:
        pass

    @abstractmethod
    def get_collection_info(self,collection_name:str) -> dict:
        pass

    @abstractmethod
    def delete_collection(self,collection_name:str):
        pass

    @abstractmethod
    def create_collection(self,collection_name:str,
                          embedding_size:int,
                          do_reset:bool = False):
        pass


    @abstractmethod
    def insert_one(self,collection_name:str,text:str,vector:list,
                   metadata:dict=None,
                   record_id:str=None):
        pass

    @abstractmethod
    def insert_many(self,collection_name:str,texts:list,
                    vectors:list,metadata:list=None,
                   record_ids:list=None,batch_size:int=50):
        pass

    @abstractmethod
    def search_by_vector(self,collection_name:str,vector:list,limit:int) -> List[RetrievedDocument]:
        pass


``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\vectordb\VectorDBProviderFactory.py
================================================================================

``` 
from .providers import QdrantDBProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController
class VectorDBProviderFactory:
    def __init__(self,config):
        self.config=config
        self.base_controller=BaseController()

    def create(self,provider:str):
        if provider==VectorDBEnums.QDRANT.value:
            db_path=self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)
            return QdrantDBProvider(
                db_path=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            )
        return None

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\vectordb\__init__.py
================================================================================

``` 

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\vectordb\providers\QdrantDBProvider.py
================================================================================

``` 
from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List
from models.db_schemes import RetrievedDocument

class QdrantDBProvider(VectorDBInterface):

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
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )

            return True
        
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        try:
            _ = self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=[record_id],
                        vector=vector,
                        payload={
                            "text": text, "metadata": metadata
                        }
                    )
                ],
                wait=True
            )
        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False

        return True
    
    def insert_many(self, collection_name: str, texts: list, 
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 50):
        
        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = [
                models.Record(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={
                        "text": batch_texts[x], "metadata": batch_metadata[x]
                    }
                )

                for x in range(len(batch_texts))
            ]

            try:
                _ = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records,
                    wait=True
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch: {e}")
                return False

        return True
        
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):

        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            with_payload=True
        )

        if not results or len(results) == 0:
            return None
        
        return [
            RetrievedDocument(**{
                "score": result.score,
                "text": result.payload["text"],
                "metadata":result.payload.get("metadata")
            })
            for result in results
        ]

``` 


================================================================================
## FILE: D:\AI-Projects\Semantic-Matching-Engine\src\stores\vectordb\providers\__init__.py
================================================================================

``` 
from .QdrantDBProvider import QdrantDBProvider
``` 


