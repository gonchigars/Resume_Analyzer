# resume_analyzer/services/resume_analyzer.py
import json
from typing import Dict, Any, List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader, PyPDFLoader
import streamlit as st
import time

from config import settings
from utils.logging_config import logger
from models.deepseek_llm import OpenRouterDeepSeek
from services.vector_store import VectorStoreService

class ResumeAnalyzer:
    def __init__(self, openrouter_api_key: str, pinecone_api_key: str):
        """Initialize the Resume Analyzer with required API keys"""
        self.openrouter_api_key = openrouter_api_key
        self.vector_store = VectorStoreService(pinecone_api_key)
        self.llm = OpenRouterDeepSeek(api_key=openrouter_api_key)
        self.vectorstore = None
    
    @st.cache_data(ttl=10)
    def list_stored_resumes(_self) -> List[str]:
        """Return list of resumes stored in the vector database"""
        return _self.vector_store.list_documents()
    
    def delete_resume(self, resume_filename: str) -> bool:
        """Delete a specific resume from the vector database"""
        success = self.vector_store.delete_document(resume_filename)
        if success:
            time.sleep(2)  # Allow time for Pinecone to update
            self.list_stored_resumes.clear()  # Clear the cache
        return success

    def process_resumes(self, data_dir: str = settings.RESUME_DIR) -> tuple[bool, str]:
        """
        Process and store resumes in the vector database
        
        Args:
            data_dir: Directory containing resume files
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Set up document loaders
            loaders = [
                DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
                DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader),
                DirectoryLoader(data_dir, glob="**/*.csv", loader_cls=CSVLoader)
            ]
            
            documents = []
            for loader in loaders:
                try:
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error loading documents: {str(e)}")
                    continue
            
            if not documents:
                raise ValueError("No resumes were loaded")
            
            # Create chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create or update vector store
            self.vectorstore = self.vector_store.create_from_documents(chunks)
            
            # Clear the cache after processing new resumes
            self.list_stored_resumes.clear()
            
            return True, f"Processed {len(documents)} resumes"
            
        except Exception as e:
            logger.error(f"Error processing resumes: {str(e)}")
            return False, str(e)

    async def analyze_resumes(self, job_requirements: str) -> Dict[str, Any]:
        """
        Analyze resumes against job requirements
        
        Args:
            job_requirements: String containing job requirements
            
        Returns:
            Dictionary containing analysis results and metadata
            
        Raises:
            Exception: If analysis fails
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vector store not initialized. Please process resumes first.")

            # Retrieve relevant resume chunks
            docs = self.vectorstore.similarity_search(job_requirements, k=10)
            
            # Group chunks by resume
            resume_contents = {}
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                if source not in resume_contents:
                    resume_contents[source] = []
                resume_contents[source].append(doc.page_content)
            
            # Analyze each resume
            analysis_results = []
            for resume_file, contents in resume_contents.items():
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
                    
                    # Clean and parse the response
                    result_text = result.strip()
                    if not result_text.startswith('{'):
                        result_text = result_text[result_text.find('{'):]
                    if not result_text.endswith('}'):
                        result_text = result_text[:result_text.rfind('}')+1]
                    
                    parsed_result = json.loads(result_text)
                    
                    # Validate required fields
                    required_fields = ['match_score', 'qualifications_match', 'missing_requirements', 
                                     'additional_skills', 'years_experience', 'summary']
                    for field in required_fields:
                        if field not in parsed_result:
                            raise ValueError(f"Missing required field: {field}")
                    
                    parsed_result["resume_file"] = resume_file
                    analysis_results.append(parsed_result)
                    logger.info(f"Successfully analyzed resume: {resume_file}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {resume_file}: {str(e)}")
                    continue
            
            # Sort results by match score
            sorted_results = sorted(analysis_results, key=lambda x: x.get('match_score', 0), reverse=True)
            
            return {
                "analysis": sorted_results,
                "total_resumes": len(resume_contents)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing resumes: {str(e)}")
            raise