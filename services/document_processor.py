from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
import nltk
from nltk.tokenize import sent_tokenize
from utils.logging_config import logger
from config.settings import settings

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