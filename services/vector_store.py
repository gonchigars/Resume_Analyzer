# resume_analyzer/services/vector_store.py
from typing import List
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings
from utils.logging_config import logger

class VectorStoreService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pc = Pinecone(api_key=api_key)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.embeddings = self._setup_embeddings()
        
    def _setup_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL,
            model_kwargs={'device': settings.EMBEDDINGS_DEVICE}
        )
    
    def initialize_store(self):
        try:
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(...)
            else:
                logger.info(f"Using existing index: {self.index_name}")
            return self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Pinecone connection failed: {str(e)}")
            raise

    
    def list_documents(self) -> List[str]:
        """List all documents in the vector store"""
        try:
            if self.index_name not in self.pc.list_indexes().names():
                return []
            
            index = self.pc.Index(self.index_name)
            query_response = index.query(
                vector=[0] * settings.PINECONE_DIMENSION,
                top_k=10000,
                include_metadata=True
            )
            
            unique_docs = {
                match.metadata["source"] 
                for match in query_response.matches 
                if "source" in match.metadata
            }
            return sorted(list(unique_docs))
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def delete_document(self, filename: str) -> bool:
        """Delete a document from the vector store"""
        try:
            logger.info(f"Attempting to delete document: {filename}")
            
            if self.index_name not in self.pc.list_indexes().names():
                logger.warning(f"Index {self.index_name} not found")
                return False
            
            index = self.pc.Index(self.index_name)
            
            query_response = index.query(
                vector=[0.0] * settings.PINECONE_DIMENSION,
                top_k=100,
                include_metadata=True
            )
            
            vectors_to_delete = [
                match.id for match in query_response.matches 
                if match.metadata.get('source') == filename
            ]
            
            if vectors_to_delete:
                delete_response = index.delete(ids=vectors_to_delete)
                logger.info(f"Successfully deleted vectors for {filename}")
                return True
            else:
                logger.warning(f"No vectors found for document: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def create_from_documents(self, documents):
        """Create or update vector store from documents"""
        try:
            index = self.initialize_store()
            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name,
                pinecone_api_key=self.api_key
            )
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise