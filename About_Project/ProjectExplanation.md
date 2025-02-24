# Resume Analyzer: Complete Technical Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Configuration](#configuration)
5. [Services](#services)
6. [Frontend Implementation](#frontend-implementation)
7. [External Integrations](#external-integrations)
8. [Error Handling & Logging](#error-handling--logging)

## Project Overview

The Resume Analyzer is an AI-powered system that analyzes resumes against job requirements. It uses advanced NLP techniques, vector databases, and machine learning to provide intelligent resume matching and analysis.

### Key Features

- Resume processing and analysis
- Job requirement matching
- Vector-based similarity search
- AI-powered content analysis
- User-friendly web interface

## Project Structure

```
project_root/
├── app.py                 # Main Streamlit application
├── run.py                # Application launcher
├── config/               # Configuration files
│   ├── settings.py       # System settings
│   └── __init__.py
├── models/              # AI models
│   ├── deepseek_llm.py  # LLM integration
│   └── __init__.py
├── services/            # Core services
│   ├── document_processor.py
│   ├── resume_analyzer.py
│   ├── vector_store.py
│   └── __init__.py
└── utils/              # Utility functions
    ├── logging_config.py
    └── __init__.py
```

## Core Components

### 1. Settings (config/settings.py)

```python
class Settings:
    def __init__(self):
        # API Keys
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

        # DeepSeek Model Settings
        self.DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-r1:nitro')
        self.DEEPSEEK_TEMPERATURE = float(os.getenv('DEEPSEEK_TEMPERATURE', '0.5'))

        # Pinecone Settings
        self.PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'resume-analysis')
        self.PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '768'))

        # Document Processing Settings
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
```

### 2. Document Processor (services/document_processor.py)

The EnhancedDocumentProcessor handles document parsing and chunking:

```python
class EnhancedDocumentProcessor:
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Creates smart chunks based on document content"""
        # Try header-based splitting first
        try:
            md_chunks = self._try_markdown_splitting(text)
            if md_chunks:
                return self._process_markdown_chunks(md_chunks, metadata)
        except Exception:
            pass

        # Fall back to semantic chunking
        return self._create_semantic_chunks(text, metadata)
```

### 3. Vector Store Service (services/vector_store.py)

Manages vector embeddings and similarity search:

```python
class VectorStoreService:
    def __init__(self, api_key: str):
        self.pc = Pinecone(api_key=api_key)
        self.embeddings = self._setup_embeddings()

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Performs similarity search in vector store"""
        query_embedding = self.embeddings.embed_query(query)
        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        return self._format_results(results)
```

### 4. Resume Analyzer (services/resume_analyzer.py)

Core analysis service that coordinates the entire process:

```python
class ResumeAnalyzer:
    async def analyze_resumes(self, job_requirements: str) -> Dict[str, Any]:
        """Analyzes resumes against job requirements"""
        # Get relevant resume chunks
        docs = self.vectorstore.similarity_search(job_requirements, k=10)

        # Group by resume
        resume_contents = self._group_by_resume(docs)

        # Analyze each resume
        analysis_results = []
        for resume_file, contents in resume_contents.items():
            result = await self._analyze_single_resume(
                contents,
                job_requirements
            )
            analysis_results.append(result)

        return {
            "analysis": sorted(analysis_results, key=lambda x: x['match_score'], reverse=True),
            "total_resumes": len(resume_contents)
        }
```

### 5. DeepSeek LLM Integration (models/deepseek_llm.py)

Handles interaction with the AI model:

```python
class OpenRouterDeepSeek:
    async def ainvoke(self, prompt: str) -> str:
        """Asynchronously invokes the DeepSeek model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_base, headers=headers, json=payload) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
```

## Frontend Implementation (app.py)

### 1. Main Application Structure

```python
def main():
    st.set_page_config(page_title="Resume Analysis System", layout="wide")
    initialize_session_state()

    # Build UI
    with st.sidebar:
        build_sidebar()
    build_main_content()
```

### 2. File Upload and Processing

```python
def handle_file_upload(uploaded_files):
    """Handles resume file uploads"""
    if uploaded_files:
        try:
            os.makedirs(settings.RESUME_DIR, exist_ok=True)
            for file in uploaded_files:
                save_uploaded_file(file)
            process_uploaded_files()
        except Exception as e:
            handle_upload_error(e)
```

### 3. Analysis Interface

```python
def build_analysis_interface():
    """Builds the analysis interface"""
    st.header("Job Requirements")
    job_requirements = st.text_area(
        "Enter the job requirements:",
        value=st.session_state.job_requirements,
        height=200
    )

    if st.button("Analyze Resumes") and job_requirements:
        run_analysis(job_requirements)
```

## Error Handling & Logging

### 1. Logging Configuration

```python
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Add handlers
    ch = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler('app_debug.log')

    # Set formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
```

### 2. Error Handling

```python
def handle_operation_error(operation: str, error: Exception):
    """Centralized error handling"""
    logger.error(f"Error during {operation}: {str(error)}")
    st.error(f"❌ Error: {str(error)}")

    if isinstance(error, APIError):
        handle_api_error(error)
    elif isinstance(error, FileNotFoundError):
        handle_file_error(error)
    else:
        handle_general_error(error)
```

## External Integrations

### 1. Pinecone Integration

The system uses Pinecone for vector storage and similarity search:

```python
def initialize_pinecone():
    """Initialize Pinecone connection"""
    try:
        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )

        # Create index if it doesn't exist
        if settings.PINECONE_INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.PINECONE_DIMENSION,
                metric=settings.PINECONE_METRIC
            )

        return pinecone.Index(settings.PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Pinecone initialization failed: {str(e)}")
        raise
```

### 2. DeepSeek Integration

Integration with the DeepSeek language model via OpenRouter:

```python
async def analyze_resume_content(content: str, requirements: str) -> Dict[str, Any]:
    """Analyze resume content using DeepSeek"""
    prompt = construct_analysis_prompt(content, requirements)

    try:
        response = await deepseek.ainvoke(prompt)
        return parse_llm_response(response)
    except Exception as e:
        logger.error(f"DeepSeek analysis failed: {str(e)}")
        raise
```

## Key Features Implementation

### 1. Resume Processing

```python
def process_resume(file_path: str) -> Dict[str, Any]:
    """Process a single resume file"""
    # Load document
    if file_path.endswith('.pdf'):
        text = extract_pdf_text(file_path)
    else:
        text = extract_text_file(file_path)

    # Create chunks
    chunks = document_processor.create_chunks(text)

    # Extract metadata
    metadata = document_processor.extract_metadata(text, file_path)

    # Create embeddings
    embeddings = vector_store.create_embeddings(chunks)

    return {
        'chunks': chunks,
        'metadata': metadata,
        'embeddings': embeddings
    }
```

### 2. Analysis Pipeline

```python
async def analyze_resume(resume_content: str, job_requirements: str) -> Dict[str, Any]:
    """Complete resume analysis pipeline"""
    # Step 1: Extract relevant information
    relevant_chunks = vector_store.similarity_search(
        job_requirements,
        resume_content
    )

    # Step 2: Analyze with LLM
    analysis = await deepseek.analyze_content(
        relevant_chunks,
        job_requirements
    )

    # Step 3: Process and structure results
    structured_results = process_analysis_results(analysis)

    return structured_results
```

### 3. Results Processing

```python
def process_analysis_results(raw_results: Dict[str, Any]) -> Dict[str, Any]:
    """Process and structure analysis results"""
    return {
        'match_score': calculate_match_score(raw_results),
        'qualifications': extract_qualifications(raw_results),
        'missing_requirements': identify_missing_requirements(raw_results),
        'experience': calculate_experience(raw_results),
        'summary': generate_summary(raw_results)
    }
```

## Usage Example

```python
# Initialize the system
analyzer = ResumeAnalyzer(
    openrouter_api_key=settings.OPENROUTER_API_KEY,
    pinecone_api_key=settings.PINECONE_API_KEY
)

# Process resumes
success, message = analyzer.process_new_resumes()

# Analyze against job requirements
results = await analyzer.analyze_resumes(job_requirements)

# Display results
display_analysis_results(results)
```

This documentation covers the main components and functionality of the Resume Analyzer project. Each component is designed to be modular and maintainable, with clear separation of concerns and robust error handling.
