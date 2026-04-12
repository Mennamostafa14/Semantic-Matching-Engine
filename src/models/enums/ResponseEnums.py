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