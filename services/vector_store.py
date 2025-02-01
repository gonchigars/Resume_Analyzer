# services/vector_store.py
import os
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader

from config.settings import settings
from utils.logging_config import logger
from services.document_processor import EnhancedDocumentProcessor

class VectorStoreService:
    """Service for managing vector store operations using Pinecone"""
    
    def __init__(self, api_key: str):
        """
        Initialize the VectorStoreService.
        
        Args:
            api_key (str): Pinecone API key
        """
        self.api_key = api_key
        self.pc = Pinecone(api_key=api_key)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.embeddings = self._setup_embeddings()
        self.document_processor = EnhancedDocumentProcessor()
        self._index = None

    def _setup_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Set up the HuggingFace embeddings model.
        
        Returns:
            HuggingFaceEmbeddings: Configured embeddings model
        """
        try:
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDINGS_MODEL,
                model_kwargs={'device': settings.EMBEDDINGS_DEVICE}
            )
        except Exception as e:
            logger.error(f"Error setting up embeddings: {str(e)}")
            raise

    def initialize_store(self) -> Optional[Any]:
        """
        Initialize or connect to Pinecone index.
        
        Returns:
            Optional[Any]: Initialized Pinecone index or None if initialization fails
        """
        try:
            existing_indexes = self.pc.list_indexes()
            
            if self.index_name not in existing_indexes.names():
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.PINECONE_DIMENSION,
                    metric=settings.PINECONE_METRIC,
                    spec=ServerlessSpec(
                        cloud=settings.PINECONE_CLOUD,
                        region=settings.PINECONE_REGION
                    )
                )
                # Wait for index to be ready
                while self.index_name not in self.pc.list_indexes().names():
                    time.sleep(1)
            else:
                logger.info(f"Using existing index: {self.index_name}")
            
            self._index = self.pc.Index(self.index_name)
            if self._index:
                logger.info("Successfully connected to Pinecone index")
                return self._index
            return None
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {str(e)}")
            raise

    def list_documents(self) -> List[str]:
        """
        List all unique document sources stored in the vector database.
        
        Returns:
            List[str]: List of document source paths
        """
        try:
            if not self._index:
                self._index = self.pc.Index(self.index_name)
            
            # Query for all vectors and fetch metadata
            query_response = self._index.query(
                vector=[0] * settings.PINECONE_DIMENSION,  # Dummy vector for metadata query
                top_k=10000,  # Adjust based on your needs
                include_metadata=True
            )
            
            # Extract unique source files from metadata
            unique_sources = set()
            for match in query_response['matches']:
                if 'metadata' in match and 'source' in match['metadata']:
                    unique_sources.add(match['metadata']['source'])
            
            return sorted(list(unique_sources))
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    def delete_document(self, document_path: str) -> bool:
        """
        Delete all vectors associated with a specific document.
        
        Args:
            document_path (str): Path of the document to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            if not self._index:
                self._index = self.pc.Index(self.index_name)

            logger.info(f"Attempting to delete document: {document_path}")
            
            # Find all vectors associated with this document
            query_response = self._index.query(
                vector=[0] * settings.PINECONE_DIMENSION,
                top_k=10000,
                include_metadata=True
            )
            
            # Get IDs of vectors to delete
            ids_to_delete = [
                match['id'] for match in query_response['matches']
                if match.get('metadata', {}).get('source') == document_path
            ]
            
            if ids_to_delete:
                # Delete vectors in batches
                batch_size = 100
                for i in range(0, len(ids_to_delete), batch_size):
                    batch = ids_to_delete[i:i + batch_size]
                    self._index.delete(ids=batch)
                
                logger.info(f"Successfully deleted {len(ids_to_delete)} vectors for {document_path}")
                return True
            else:
                logger.warning(f"No vectors found for document: {document_path}")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting document {document_path}: {str(e)}")
            return False

    def create_from_documents(self, documents: List[Any]) -> Optional[PineconeVectorStore]:
        """
        Create or update vector store from documents with enhanced processing.
        
        Args:
            documents: List of document objects to process
            
        Returns:
            Optional[PineconeVectorStore]: Initialized vector store or None if creation fails
        """
        try:
            logger.info("Starting document processing...")
            processed_chunks = []
            
            for doc in documents:
                try:
                    source = doc.metadata.get('source', 'Unknown')
                    logger.info(f"Processing document: {source}")
                    
                    # Create chunks with metadata
                    chunk = {
                        "text": doc.page_content,
                        "metadata": {
                            "source": source,
                            "start_index": doc.metadata.get('start_index', 0),
                            "chunk_type": "document",
                            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    processed_chunks.append(chunk)
                    
                except Exception as e:
                    logger.error(f"Error processing document {source}: {str(e)}")
                    continue
            
            if not processed_chunks:
                raise ValueError("No documents were successfully processed")
            
            logger.info(f"Successfully processed {len(processed_chunks)} chunks")
            
            # Create embeddings
            logger.info("Creating embeddings...")
            texts = [chunk["text"] for chunk in processed_chunks]
            metadatas = [chunk["metadata"] for chunk in processed_chunks]
            
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Created {len(embeddings)} embeddings")
            
            # Initialize Pinecone
            index = self.initialize_store()
            if not index:
                raise ValueError("Failed to initialize Pinecone index")
            
            # Batch upload to Pinecone
            logger.info("Uploading to Pinecone...")
            batch_size = 100
            total_uploaded = 0
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadatas[i:i + batch_size]
                
                vectors = [
                    (f"vec_{total_uploaded + j}", emb, {"text": text, **meta})
                    for j, (emb, text, meta) in enumerate(zip(batch_embeddings, batch_texts, batch_metadata))
                ]
                
                index.upsert(vectors=vectors)
                total_uploaded += len(batch_texts)
                logger.info(f"Uploaded batch: {total_uploaded}/{len(texts)} vectors")
            
            logger.info("Successfully created vector store")
            return PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                text_key="text"
            )
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query (str): Query text to search for
            k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with metadata
        """
        try:
            if not self._index:
                raise ValueError("Vector store not initialized")
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform similarity search
            results = self._index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    'text': match['metadata'].get('text', ''),
                    'score': match['score'],
                    'metadata': {
                        k: v for k, v in match['metadata'].items()
                        if k != 'text'
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise