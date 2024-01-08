from .hnsw_indexer import PatientDataHnswDocumentIndex
from .weaviate_indexer import PatientDataWeaviateDocumentIndex
from .base_indexer import PatientDataIndex

index_options = {
    'PatientDataHnswDocumentIndex': PatientDataHnswDocumentIndex,
    'PatientDataWeaviateDocumentIndex': PatientDataWeaviateDocumentIndex
}
