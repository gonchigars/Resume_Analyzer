# resume_analyzer/services/resume_analyzer.py
import json
import os
from typing import Dict, Any, List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
import streamlit as st
import time

from config.settings import settings
from utils.logging_config import logger
from models.deepseek_llm import OpenRouterDeepSeek
from services.vector_store import VectorStoreService
from langchain_pinecone import PineconeVectorStore

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