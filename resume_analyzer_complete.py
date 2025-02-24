"""
Resume Analyzer - Complete Project
Generated on: 2025-02-24 09:16:58
This file contains all merged Python code from the Resume Analyzer project.
"""

# ====================== Import Statements ======================
import os
import sys
import asyncio
import streamlit as st
import warnings
import time
from typing import Dict, Any, List, Optional, Union
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
import pinecone
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
import torch
import aiohttp



# ====================== CONFIG ======================

# File: config\settings.py
# resume_analyzer/config/settings.py

class Settings:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # API Keys
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        
        # DeepSeek Model Settings
        self.DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-r1:nitro')
        self.DEEPSEEK_TEMPERATURE = float(os.getenv('DEEPSEEK_TEMPERATURE', '0.5'))
        self.DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://openrouter.ai/api/v1')
        
        # Pinecone Settings
        self.PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'resume-analysis')
        self.PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '768'))
        self.PINECONE_METRIC = os.getenv('PINECONE_METRIC', 'cosine')
        self.PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
        self.PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
        
        # Embeddings Settings
        self.EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        self.EMBEDDINGS_DEVICE = os.getenv('EMBEDDINGS_DEVICE', 'cpu').strip()  # Added .strip() to remove any whitespace
        
        # Document Processing Settings
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
        
        # Data Directory
        self.RESUME_DIR = os.getenv('RESUME_DIR', 'resumes')

# Create a single instance to be imported by other modules
settings = Settings()

# Make settings available for import
__all__ = ['settings']


# File: config\__init__.py
# resume_analyzer/config/__init__.py
"""Configuration package for the Resume Analysis System."""



# ====================== UTILS ======================

# File: utils\logging_config.py

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler('app_debug.log')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = setup_logging()


# File: utils\__init__.py
# resume_analyzer/utils/__init__.py
"""Utilities package for the Resume Analysis System."""



# ====================== MODELS ======================

# File: models\deepseek_llm.py
# models/deepseek_llm.py

class OpenRouterDeepSeek:
    """Class for interacting with OpenRouter's DeepSeek model"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenRouter API key"""
        self.api_key = api_key
        self.api_base = settings.DEEPSEEK_API_BASE
        self.model = settings.DEEPSEEK_MODEL
        self.temperature = settings.DEEPSEEK_TEMPERATURE

    async def ainvoke(self, prompt: str) -> str:
        """
        Asynchronously invoke the DeepSeek model with a prompt
        
        Args:
            prompt (str): The prompt to send to the model
            
        Returns:
            str: The model's response text
            
        Raises:
            Exception: If the API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://resume-analyzer.example.com",  # Replace with your domain
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_base,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            raise

# File: models\setup.py

setup(
    name="resume_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "python-dotenv",
        "langchain",
        "langchain-community",
        "langchain-core",
        "pinecone",  # Updated from pinecone-client
        "sentence-transformers",
        "transformers",
        "torch",
        "pypdf",
        "huggingface-hub",
        "nltk",
        "unstructured",  # For document loading
        "markdown",      # For markdown processing
        "aiohttp",      # For async HTTP requests
    ],
)

# File: models\__init__.py
# models/__init__.py
"""Models package for the Resume Analysis System."""


__all__ = ['OpenRouterDeepSeek']


# ====================== SERVICES ======================

# File: services\document_processor.py

class EnhancedDocumentProcessor:
    """Enhanced document processing with smart chunking strategies"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create smart chunks based on document content"""
        # First try header-based splitting for structured documents
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        try:
            md_chunks = markdown_splitter.split_text(text)
            if len(md_chunks) > 1:  # If we found headers
                return self._process_markdown_chunks(md_chunks, metadata)
        except Exception as e:
            logger.warning(f"Markdown splitting failed: {str(e)}")
        
        # Fall back to sentence-based splitting
        return self._create_semantic_chunks(text, metadata)
    
    def _process_markdown_chunks(
        self, 
        md_chunks: List[Any], 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process markdown chunks with headers"""
        processed_chunks = []
        
        for chunk in md_chunks:
            chunk_metadata = metadata.copy()
            # Add header information to metadata
            for header_level, header_text in chunk.metadata.items():
                chunk_metadata[f"header_{header_level}"] = header_text
            
            processed_chunks.append({
                "text": chunk.page_content,
                "metadata": chunk_metadata
            })
        
        return processed_chunks
    
    def _create_semantic_chunks(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create chunks based on semantic boundaries"""
        # First split into sentences
        sentences = sent_tokenize(text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > settings.CHUNK_SIZE:
                if current_chunk:  # Save current chunk if it exists
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "metadata": metadata.copy()
                    })
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "metadata": metadata.copy()
            })
        
        return chunks
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract relevant metadata from document content"""
        metadata = {
            "source": filename,
            "char_count": len(text),
            "estimated_read_time": len(text.split()) / 200  # Assuming 200 wpm reading speed
        }
        
        # Try to identify document sections
        sections = []
        current_section = ""
        for line in text.split('\n'):
            if line.strip().startswith('#'):
                if current_section:
                    sections.append(current_section)
                current_section = line.strip('#').strip()
        if current_section:
            sections.append(current_section)
        
        if sections:
            metadata["sections"] = sections
        
        return metadata

# File: services\resume_analyzer.py
# resume_analyzer/services/resume_analyzer.py


class ResumeAnalyzer:
    def __init__(self, openrouter_api_key: str, pinecone_api_key: str):
        """Initialize the Resume Analyzer with required API keys"""
        logger.info("Initializing ResumeAnalyzer")
        self.openrouter_api_key = openrouter_api_key
        self.vector_store = VectorStoreService(pinecone_api_key)
        self.llm = OpenRouterDeepSeek(api_key=openrouter_api_key)
        
        # Initialize vector store and process existing resumes
        try:
            self.vectorstore = self._initialize_vectorstore()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _initialize_vectorstore(self):
        """Initialize vector store and process existing resumes"""
        # Initialize Pinecone connection
        index = self.vector_store.initialize_store()
        
        # Create VectorStore instance
        if index:
            return PineconeVectorStore(
                index=index,
                embedding=self.vector_store.embeddings,
                text_key="text"
            )
        return None

    def process_new_resumes(self, data_dir: str = settings.RESUME_DIR) -> Tuple[bool, str]:
        """Process only newly added resumes"""
        try:
            logger.info(f"Processing new resumes from directory: {data_dir}")
            
            if not os.path.exists(data_dir):
                logger.error(f"Directory not found: {data_dir}")
                return False, "Resume directory not found"
            
            # Get list of files and currently stored resumes
            files = [f for f in os.listdir(data_dir) 
                    if os.path.isfile(os.path.join(data_dir, f))]
            stored_resumes = self.vector_store.list_documents()
            
            # Filter for new files
            new_files = [f for f in files 
                        if os.path.join(data_dir, f) not in stored_resumes]
            
            if not new_files:
                logger.info("No new resumes to process")
                return True, "No new resumes to process"
            
            logger.info(f"Found {len(new_files)} new files to process")
            
            # Process new files
            documents = []
            for file in new_files:
                file_path = os.path.join(data_dir, file)
                try:
                    if file.lower().endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    else:
                        loader = TextLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            
            if not documents:
                return False, "No new resumes could be processed"
            
            # Create chunks and update vector store
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Update vector store
            self.vectorstore = self.vector_store.create_from_documents(chunks)
            
            return True, f"Successfully processed {len(new_files)} new resumes"
            
        except Exception as e:
            logger.error(f"Error processing new resumes: {str(e)}")
            return False, f"Error processing new resumes: {str(e)}"

    def list_stored_resumes(self) -> List[str]:
        """Return list of resumes stored in the vector database"""
        return self.vector_store.list_documents()
    
    def delete_resume(self, resume_filename: str) -> bool:
        """Delete a specific resume from the vector database"""
        success = self.vector_store.delete_document(resume_filename)
        if success:
            # Reinitialize vector store after deletion
            try:
                self.vectorstore = self._initialize_vectorstore()
            except Exception as e:
                logger.error(f"Error reinitializing vector store after deletion: {str(e)}")
        return success
    async def analyze_resumes(self, job_requirements: str) -> Dict[str, Any]:
        """
        Analyze resumes against job requirements
        
        Args:
            job_requirements: String containing job requirements
            
        Returns:
            Dictionary containing analysis results and metadata
            
        Raises:
            ValueError: If vector store is not initialized
            Exception: If analysis fails
        """
        try:
            if not self.vectorstore:
                logger.error("Vector store not initialized")
                raise ValueError("Vector store not initialized. Please process resumes first.")

            logger.info("Starting resume analysis...")
            # Retrieve relevant resume chunks
            docs = self.vectorstore.similarity_search(job_requirements, k=10)
            logger.info(f"Retrieved {len(docs)} relevant chunks")
            
            # Group chunks by resume
            resume_contents = {}
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                if source not in resume_contents:
                    resume_contents[source] = []
                resume_contents[source].append(doc.page_content)
            
            logger.info(f"Analyzing {len(resume_contents)} resumes")
            
            # Analyze each resume
            analysis_results = []
            for resume_file, contents in resume_contents.items():
                logger.info(f"Analyzing resume: {resume_file}")
                full_content = "\n".join(contents)
                
                analysis_prompt = f"""
                Act as an expert HR analyst. Analyze this resume against the following job requirements. 
                Provide a structured analysis in valid JSON format.
                
                Job Requirements:
                {job_requirements}
                
                Resume Content:
                {full_content}
                
                Response Instructions:
                1. Match Score: Provide a number between 0-100
                2. List 3-5 key matching qualifications
                3. List 2-3 missing requirements
                4. List 2-3 additional relevant skills
                5. Calculate years of relevant experience
                6. Write a 2-3 sentence summary
                
                Required JSON Structure:
                {{
                    "match_score": <number>,
                    "qualifications_match": ["qual1", "qual2", "qual3"],
                    "missing_requirements": ["req1", "req2"],
                    "additional_skills": ["skill1", "skill2"],
                    "years_experience": <number>,
                    "summary": "Brief summary text"
                }}
                
                The response must be ONLY valid JSON with no additional text.
                """
                
                try:
                    result = await self.llm.ainvoke(analysis_prompt)
                    logger.debug(f"Raw LLM response: {result}")
                    
                    # Clean and parse the response
                    result_text = result.strip()
                    if not result_text.startswith('{'):
                        result_text = result_text[result_text.find('{'):]
                    if not result_text.endswith('}'):
                        result_text = result_text[:result_text.rfind('}')+1]
                    
                    parsed_result = json.loads(result_text)
                    
                    # Validate required fields
                    required_fields = [
                        'match_score', 
                        'qualifications_match', 
                        'missing_requirements',
                        'additional_skills', 
                        'years_experience', 
                        'summary'
                    ]
                    
                    for field in required_fields:
                        if field not in parsed_result:
                            raise ValueError(f"Missing required field: {field}")
                    
                    parsed_result["resume_file"] = resume_file
                    analysis_results.append(parsed_result)
                    logger.info(f"Successfully analyzed resume: {resume_file}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {resume_file}: {str(e)}")
                    continue
            
            if not analysis_results:
                raise Exception("No resumes could be analyzed successfully")
            
            # Sort results by match score
            sorted_results = sorted(
                analysis_results, 
                key=lambda x: x.get('match_score', 0), 
                reverse=True
            )
            
            logger.info("Analysis completed successfully")
            return {
                "analysis": sorted_results,
                "total_resumes": len(resume_contents)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing resumes: {str(e)}")
            raise

# File: services\vector_store.py
# services/vector_store.py


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

# File: services\__init__.py
# resume_analyzer/services/__init__.py
"""Services package for the Resume Analysis System."""


__all__ = ['ResumeAnalyzer', 'VectorStoreService', 'EnhancedDocumentProcessor']


# ====================== ROOT ======================

# File: app.py

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now use absolute imports

# Suppress warnings
warnings.filterwarnings('ignore')

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'resume_analyzer' not in st.session_state:
        st.session_state.resume_analyzer = None
    if 'job_requirements' not in st.session_state:
        st.session_state.job_requirements = ""
    if 'refresh_key' not in st.session_state:
        st.session_state.refresh_key = 0
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

def handle_file_upload(uploaded_files):
    """Handle uploaded resume files"""
    if uploaded_files:
        try:
            os.makedirs(settings.RESUME_DIR, exist_ok=True)
            for file in uploaded_files:
                file_path = os.path.join(settings.RESUME_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"✅ Saved {len(uploaded_files)} resumes")
            return True
        except Exception as e:
            st.error(f"Error saving files: {str(e)}")
            logger.error(f"File upload error: {str(e)}")
            return False

def display_stored_resumes():
    """Display and manage stored resumes"""
    if st.session_state.resume_analyzer is None:
        return

    stored_resumes = st.session_state.resume_analyzer.list_stored_resumes()

    if stored_resumes:
        st.sidebar.markdown("---")
        st.sidebar.header("Stored Resumes")
        for resume in stored_resumes:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.write(f"📄 {os.path.basename(resume)}")
            with col2:
                if st.button("🗑️", key=f"del_{resume}"):
                    if st.session_state.resume_analyzer.delete_resume(resume):
                        st.success(f"Deleted {resume}")
                        time.sleep(1)
                        st.rerun()
    else:
        st.sidebar.info("No resumes stored")

def main():
    st.set_page_config(
        page_title="Resume Analysis System",
        page_icon="📄",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("📄 Resume Analysis System")
    
    # System Status Display
    with st.expander("System Status", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("API Status:")
            st.write(f"OpenRouter Key: {'✅' if settings.OPENROUTER_API_KEY else '❌'}")
            st.write(f"Pinecone Key: {'✅' if settings.PINECONE_API_KEY else '❌'}")
        with col2:
            st.write("System Status:")
            st.write(f"System Initialized: {'✅' if st.session_state.resume_analyzer else '❌'}")
            if st.session_state.processing_status:
                success, message = st.session_state.processing_status
                st.write(f"Last Process Status: {'✅' if success else '❌'} {message}")
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        # Initialize System Button
        if st.button("Initialize System", use_container_width=True):
            with st.spinner("Initializing..."):
                try:
                    st.session_state.resume_analyzer = ResumeAnalyzer(
                        openrouter_api_key=settings.OPENROUTER_API_KEY,
                        pinecone_api_key=settings.PINECONE_API_KEY
                    )
                    st.success("✅ System initialized!")
                    st.session_state.processing_status = None
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # File Upload Section
        st.header("Resume Upload")
        uploaded_files = st.file_uploader(
            "Upload Resumes (PDF, TXT)", 
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        if uploaded_files:
            if handle_file_upload(uploaded_files):
                if st.button("Process Uploaded Resumes", use_container_width=True):
                    with st.spinner("Processing resumes..."):
                        try:
                            success, message = st.session_state.resume_analyzer.process_resumes()
                            if success:
                                st.success(f"✅ {message}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ Error processing resumes: {str(e)}")
        
        # Display stored resumes in sidebar
        display_stored_resumes()
    
    # Main Content Area
    if st.session_state.resume_analyzer:
        st.header("Job Requirements")
        job_requirements = st.text_area(
            "Enter the job requirements:",
            value=st.session_state.job_requirements,
            height=200
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("Analyze Resumes", type="primary")
        
        if analyze_button and job_requirements:
            st.session_state.job_requirements = job_requirements
            with st.spinner("Analyzing resumes..."):
                try:
                    results = asyncio.run(
                        st.session_state.resume_analyzer.analyze_resumes(job_requirements)
                    )
                    if results:
                        st.success("Analysis completed!")
                        display_analysis_results(results)
                    else:
                        st.warning("No results to display")
                except Exception as e:
                    st.error(f"❌ Error analyzing resumes: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")
        elif analyze_button:
            st.warning("Please enter job requirements first")
    else:
        st.info("👈 Please initialize the system using the sidebar controls")

def display_analysis_results(results):
    if not results or not results.get('analysis'):
        st.warning("No analysis results to display")
        return
    
    st.write("### Analysis Results")
    
    # Display summary stats
    st.write(f"Total Resumes Analyzed: {results['total_resumes']}")
    
    # Display individual results
    for result in results['analysis']:
        with st.expander(f"📄 {os.path.basename(result['resume_file'])} (Match Score: {result['match_score']}%)"):
            st.write("**Key Matching Qualifications:**")
            for qual in result['qualifications_match']:
                st.write(f"- {qual}")
                
            st.write("\n**Missing Requirements:**")
            for req in result['missing_requirements']:
                st.write(f"- {req}")
                
            st.write("\n**Additional Relevant Skills:**")
            for skill in result['additional_skills']:
                st.write(f"- {skill}")
                
            st.write(f"\n**Years of Experience:** {result['years_experience']}")
            st.write(f"\n**Summary:** {result['summary']}")

if __name__ == "__main__":
    main()

# File: Merge.py

def should_include_file(file_path: str, excluded_dirs: Set[str], excluded_extensions: Set[str], excluded_files: Set[str]) -> bool:
    """
    Check if a file should be included in the merge.
    """
    path = Path(file_path)
    
    # Check if file is in excluded_files
    if path.name.lower() in excluded_files:
        return False
        
    # Check if any parent directory is in excluded_dirs
    if any(part in excluded_dirs for part in path.parts):
        return False
        
    # Check file extension
    if path.suffix in excluded_extensions:
        return False
        
    # Check if it's a hidden file
    if path.name.startswith('.'):
        return False
        
    return True

def get_all_files(excluded_dirs: Set[str], excluded_extensions: Set[str], excluded_files: Set[str]) -> List[str]:
    """
    Get all files in current directory that should be included.
    """
    all_files = []
    
    for root, _, files in os.walk('.'):
        for file in files:
            file_path = os.path.join(root, file)
            if should_include_file(file_path, excluded_dirs, excluded_extensions, excluded_files):
                all_files.append(file_path)
                
    return sorted(all_files)  # Sort for consistent output

def create_merged_file() -> None:
    """
    Merge all files from current directory into merged_project.txt
    """
    output_file = 'merged_project.txt'
    
    # Define exclusions
    excluded_dirs = {'__pycache__', 'venv', '.git', '.idea', 'node_modules', '.pytest_cache'}
    excluded_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib'}
    excluded_files = {'readme.md', 'merge.py', output_file.lower(), '.gitignore', 'requirements.txt', '.env', '.env.template'}
    
    files = get_all_files(excluded_dirs, excluded_extensions, excluded_files)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write header
        outfile.write(f"# Merged File Contents\n")
        outfile.write(f"# Generated on: {datetime.datetime.now()}\n")
        outfile.write(f"# Source Directory: {os.path.abspath('.')}\n\n")
        
        for file_path in files:
            try:
                # Write file header
                outfile.write(f"\n{'='*80}\n")
                outfile.write(f"# File: {file_path}\n")
                outfile.write(f"{'='*80}\n\n")
                
                # Write file contents
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    
                # Add newline after each file
                outfile.write('\n')
                
            except Exception as e:
                outfile.write(f"# Error reading file {file_path}: {str(e)}\n")

def main():
    create_merged_file()
    print("Files merged successfully into merged_project.txt")

if __name__ == "__main__":
    main()

# File: resume_analyzer_complete.py
"""
Resume Analyzer - Complete Project
Generated on: 2025-02-24 09:16:58
This file contains all merged Python code from the Resume Analyzer project.
"""

# ====================== Import Statements ======================



# ====================== CONFIG ======================

# File: config\settings.py
# resume_analyzer/config/settings.py

class Settings:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # API Keys
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        
        # DeepSeek Model Settings
        self.DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-r1:nitro')
        self.DEEPSEEK_TEMPERATURE = float(os.getenv('DEEPSEEK_TEMPERATURE', '0.5'))
        self.DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://openrouter.ai/api/v1')
        
        # Pinecone Settings
        self.PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'resume-analysis')
        self.PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '768'))
        self.PINECONE_METRIC = os.getenv('PINECONE_METRIC', 'cosine')
        self.PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
        self.PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
        
        # Embeddings Settings
        self.EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        self.EMBEDDINGS_DEVICE = os.getenv('EMBEDDINGS_DEVICE', 'cpu').strip()  # Added .strip() to remove any whitespace
        
        # Document Processing Settings
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
        
        # Data Directory
        self.RESUME_DIR = os.getenv('RESUME_DIR', 'resumes')

# Create a single instance to be imported by other modules
settings = Settings()

# Make settings available for import
__all__ = ['settings']


# File: config\__init__.py
# resume_analyzer/config/__init__.py
"""Configuration package for the Resume Analysis System."""



# ====================== UTILS ======================

# File: utils\logging_config.py

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler('app_debug.log')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = setup_logging()


# File: utils\__init__.py
# resume_analyzer/utils/__init__.py
"""Utilities package for the Resume Analysis System."""



# ====================== MODELS ======================

# File: models\deepseek_llm.py
# models/deepseek_llm.py

class OpenRouterDeepSeek:
    """Class for interacting with OpenRouter's DeepSeek model"""
    
    def __init__(self, api_key: str):
        """Initialize with OpenRouter API key"""
        self.api_key = api_key
        self.api_base = settings.DEEPSEEK_API_BASE
        self.model = settings.DEEPSEEK_MODEL
        self.temperature = settings.DEEPSEEK_TEMPERATURE

    async def ainvoke(self, prompt: str) -> str:
        """
        Asynchronously invoke the DeepSeek model with a prompt
        
        Args:
            prompt (str): The prompt to send to the model
            
        Returns:
            str: The model's response text
            
        Raises:
            Exception: If the API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://resume-analyzer.example.com",  # Replace with your domain
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_base,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {error_text}")
                    
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            raise

# File: models\setup.py

setup(
    name="resume_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "python-dotenv",
        "langchain",
        "langchain-community",
        "langchain-core",
        "pinecone",  # Updated from pinecone-client
        "sentence-transformers",
        "transformers",
        "torch",
        "pypdf",
        "huggingface-hub",
        "nltk",
        "unstructured",  # For document loading
        "markdown",      # For markdown processing
        "aiohttp",      # For async HTTP requests
    ],
)

# File: models\__init__.py
# models/__init__.py
"""Models package for the Resume Analysis System."""


__all__ = ['OpenRouterDeepSeek']


# ====================== SERVICES ======================

# File: services\document_processor.py

class EnhancedDocumentProcessor:
    """Enhanced document processing with smart chunking strategies"""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create smart chunks based on document content"""
        # First try header-based splitting for structured documents
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        
        try:
            md_chunks = markdown_splitter.split_text(text)
            if len(md_chunks) > 1:  # If we found headers
                return self._process_markdown_chunks(md_chunks, metadata)
        except Exception as e:
            logger.warning(f"Markdown splitting failed: {str(e)}")
        
        # Fall back to sentence-based splitting
        return self._create_semantic_chunks(text, metadata)
    
    def _process_markdown_chunks(
        self, 
        md_chunks: List[Any], 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process markdown chunks with headers"""
        processed_chunks = []
        
        for chunk in md_chunks:
            chunk_metadata = metadata.copy()
            # Add header information to metadata
            for header_level, header_text in chunk.metadata.items():
                chunk_metadata[f"header_{header_level}"] = header_text
            
            processed_chunks.append({
                "text": chunk.page_content,
                "metadata": chunk_metadata
            })
        
        return processed_chunks
    
    def _create_semantic_chunks(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create chunks based on semantic boundaries"""
        # First split into sentences
        sentences = sent_tokenize(text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > settings.CHUNK_SIZE:
                if current_chunk:  # Save current chunk if it exists
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "metadata": metadata.copy()
                    })
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "metadata": metadata.copy()
            })
        
        return chunks
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract relevant metadata from document content"""
        metadata = {
            "source": filename,
            "char_count": len(text),
            "estimated_read_time": len(text.split()) / 200  # Assuming 200 wpm reading speed
        }
        
        # Try to identify document sections
        sections = []
        current_section = ""
        for line in text.split('\n'):
            if line.strip().startswith('#'):
                if current_section:
                    sections.append(current_section)
                current_section = line.strip('#').strip()
        if current_section:
            sections.append(current_section)
        
        if sections:
            metadata["sections"] = sections
        
        return metadata

# File: services\resume_analyzer.py
# resume_analyzer/services/resume_analyzer.py


class ResumeAnalyzer:
    def __init__(self, openrouter_api_key: str, pinecone_api_key: str):
        """Initialize the Resume Analyzer with required API keys"""
        logger.info("Initializing ResumeAnalyzer")
        self.openrouter_api_key = openrouter_api_key
        self.vector_store = VectorStoreService(pinecone_api_key)
        self.llm = OpenRouterDeepSeek(api_key=openrouter_api_key)
        
        # Initialize vector store and process existing resumes
        try:
            self.vectorstore = self._initialize_vectorstore()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _initialize_vectorstore(self):
        """Initialize vector store and process existing resumes"""
        # Initialize Pinecone connection
        index = self.vector_store.initialize_store()
        
        # Create VectorStore instance
        if index:
            return PineconeVectorStore(
                index=index,
                embedding=self.vector_store.embeddings,
                text_key="text"
            )
        return None

    def process_new_resumes(self, data_dir: str = settings.RESUME_DIR) -> Tuple[bool, str]:
        """Process only newly added resumes"""
        try:
            logger.info(f"Processing new resumes from directory: {data_dir}")
            
            if not os.path.exists(data_dir):
                logger.error(f"Directory not found: {data_dir}")
                return False, "Resume directory not found"
            
            # Get list of files and currently stored resumes
            files = [f for f in os.listdir(data_dir) 
                    if os.path.isfile(os.path.join(data_dir, f))]
            stored_resumes = self.vector_store.list_documents()
            
            # Filter for new files
            new_files = [f for f in files 
                        if os.path.join(data_dir, f) not in stored_resumes]
            
            if not new_files:
                logger.info("No new resumes to process")
                return True, "No new resumes to process"
            
            logger.info(f"Found {len(new_files)} new files to process")
            
            # Process new files
            documents = []
            for file in new_files:
                file_path = os.path.join(data_dir, file)
                try:
                    if file.lower().endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    else:
                        loader = TextLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
            
            if not documents:
                return False, "No new resumes could be processed"
            
            # Create chunks and update vector store
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Update vector store
            self.vectorstore = self.vector_store.create_from_documents(chunks)
            
            return True, f"Successfully processed {len(new_files)} new resumes"
            
        except Exception as e:
            logger.error(f"Error processing new resumes: {str(e)}")
            return False, f"Error processing new resumes: {str(e)}"

    def list_stored_resumes(self) -> List[str]:
        """Return list of resumes stored in the vector database"""
        return self.vector_store.list_documents()
    
    def delete_resume(self, resume_filename: str) -> bool:
        """Delete a specific resume from the vector database"""
        success = self.vector_store.delete_document(resume_filename)
        if success:
            # Reinitialize vector store after deletion
            try:
                self.vectorstore = self._initialize_vectorstore()
            except Exception as e:
                logger.error(f"Error reinitializing vector store after deletion: {str(e)}")
        return success
    async def analyze_resumes(self, job_requirements: str) -> Dict[str, Any]:
        """
        Analyze resumes against job requirements
        
        Args:
            job_requirements: String containing job requirements
            
        Returns:
            Dictionary containing analysis results and metadata
            
        Raises:
            ValueError: If vector store is not initialized
            Exception: If analysis fails
        """
        try:
            if not self.vectorstore:
                logger.error("Vector store not initialized")
                raise ValueError("Vector store not initialized. Please process resumes first.")

            logger.info("Starting resume analysis...")
            # Retrieve relevant resume chunks
            docs = self.vectorstore.similarity_search(job_requirements, k=10)
            logger.info(f"Retrieved {len(docs)} relevant chunks")
            
            # Group chunks by resume
            resume_contents = {}
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                if source not in resume_contents:
                    resume_contents[source] = []
                resume_contents[source].append(doc.page_content)
            
            logger.info(f"Analyzing {len(resume_contents)} resumes")
            
            # Analyze each resume
            analysis_results = []
            for resume_file, contents in resume_contents.items():
                logger.info(f"Analyzing resume: {resume_file}")
                full_content = "\n".join(contents)
                
                analysis_prompt = f"""
                Act as an expert HR analyst. Analyze this resume against the following job requirements. 
                Provide a structured analysis in valid JSON format.
                
                Job Requirements:
                {job_requirements}
                
                Resume Content:
                {full_content}
                
                Response Instructions:
                1. Match Score: Provide a number between 0-100
                2. List 3-5 key matching qualifications
                3. List 2-3 missing requirements
                4. List 2-3 additional relevant skills
                5. Calculate years of relevant experience
                6. Write a 2-3 sentence summary
                
                Required JSON Structure:
                {{
                    "match_score": <number>,
                    "qualifications_match": ["qual1", "qual2", "qual3"],
                    "missing_requirements": ["req1", "req2"],
                    "additional_skills": ["skill1", "skill2"],
                    "years_experience": <number>,
                    "summary": "Brief summary text"
                }}
                
                The response must be ONLY valid JSON with no additional text.
                """
                
                try:
                    result = await self.llm.ainvoke(analysis_prompt)
                    logger.debug(f"Raw LLM response: {result}")
                    
                    # Clean and parse the response
                    result_text = result.strip()
                    if not result_text.startswith('{'):
                        result_text = result_text[result_text.find('{'):]
                    if not result_text.endswith('}'):
                        result_text = result_text[:result_text.rfind('}')+1]
                    
                    parsed_result = json.loads(result_text)
                    
                    # Validate required fields
                    required_fields = [
                        'match_score', 
                        'qualifications_match', 
                        'missing_requirements',
                        'additional_skills', 
                        'years_experience', 
                        'summary'
                    ]
                    
                    for field in required_fields:
                        if field not in parsed_result:
                            raise ValueError(f"Missing required field: {field}")
                    
                    parsed_result["resume_file"] = resume_file
                    analysis_results.append(parsed_result)
                    logger.info(f"Successfully analyzed resume: {resume_file}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {resume_file}: {str(e)}")
                    continue
            
            if not analysis_results:
                raise Exception("No resumes could be analyzed successfully")
            
            # Sort results by match score
            sorted_results = sorted(
                analysis_results, 
                key=lambda x: x.get('match_score', 0), 
                reverse=True
            )
            
            logger.info("Analysis completed successfully")
            return {
                "analysis": sorted_results,
                "total_resumes": len(resume_contents)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing resumes: {str(e)}")
            raise

# File: services\vector_store.py
# services/vector_store.py


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



# File: run.py

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the Streamlit app
if hasattr(app, 'main'):
    app.main()

# File: setup.py

setup(
    name="resume_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "python-dotenv",
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-pinecone",  # Added this package
        "pinecone>=0.8.0",
        "sentence-transformers",
        "transformers",
        "torch",
        "pypdf",
        "huggingface-hub",
        "nltk",
        "unstructured",
        "markdown",
        "aiohttp",
    ],
)

# File: test.py
st.write("Hello World")

# File: verify.py

print("All critical packages imported successfully!")

# File: __init__.py
# resume_analyzer/services/__init__.py
"""Services package for the Resume Analysis System."""


__all__ = ['EnhancedDocumentProcessor', 'VectorStoreService', 'ResumeAnalyzer']

