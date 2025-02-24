# Resume Analyzer: Complete Streamlit Implementation Guide

## Table of Contents

1. [Project Structure](#project-structure)
2. [Main Application (app.py)](#main-application)
3. [Core Components](#core-components)
4. [Page Layout](#page-layout)
5. [Implementation Details](#implementation-details)
6. [Code Examples](#code-examples)

## Project Structure

```
project_root/
│
├── app.py                # Main Streamlit application
├── run.py               # Streamlit app launcher
├── test.py              # Streamlit test file
│
├── config/
│   ├── settings.py      # Configuration affecting Streamlit
│   └── __init__.py
│
├── services/            # Backend services called by Streamlit
│   ├── resume_analyzer.py
│   ├── vector_store.py
│   └── document_processor.py
│
└── utils/
    └── logging_config.py # Logging used in Streamlit
```

## Main Application

### app.py - Core Structure

```python
import streamlit as st
import os
import sys
import asyncio
import warnings
from typing import Dict, Any, List

def main():
    # Page Configuration
    st.set_page_config(
        page_title="Resume Analysis System",
        page_icon="📄",
        layout="wide"
    )

    initialize_session_state()

    # Main UI Components
    st.title("📄 Resume Analysis System")

    # Side Bar
    with st.sidebar:
        build_sidebar()

    # Main Content
    build_main_content()

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

if __name__ == "__main__":
    main()
```

## Core Components

### 1. Session State Management

```python
def initialize_session_state():
    """Initialize all required session state variables"""
    states = {
        'resume_analyzer': None,
        'job_requirements': "",
        'refresh_key': 0,
        'processing_status': None
    }

    for key, default_value in states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
```

### 2. File Upload Handler

```python
def handle_file_upload(uploaded_files):
    """Process uploaded resume files"""
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
```

### 3. Resume Display

```python
def display_stored_resumes():
    """Display and manage stored resumes in sidebar"""
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
                        st.rerun()
```

## Page Layout

### 1. Sidebar Structure

```python
def build_sidebar():
    st.header("System Controls")

    # Initialize System Button
    if st.button("Initialize System", use_container_width=True):
        initialize_system()

    # File Upload Section
    st.header("Resume Upload")
    handle_file_upload_section()

    # Display stored resumes
    display_stored_resumes()
```

### 2. Main Content Structure

```python
def build_main_content():
    # System Status Display
    with st.expander("System Status", expanded=True):
        display_system_status()

    # Job Requirements Section
    st.header("Job Requirements")
    job_requirements = st.text_area(
        "Enter the job requirements:",
        value=st.session_state.job_requirements,
        height=200
    )

    # Analysis Controls
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("Analyze Resumes", type="primary")

    # Handle Analysis
    if analyze_button and job_requirements:
        handle_resume_analysis(job_requirements)
```

## Implementation Details

### 1. System Status Display

```python
def display_system_status():
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
```

### 2. Analysis Results Display

```python
def display_analysis_results(results):
    if not results or not results.get('analysis'):
        st.warning("No analysis results to display")
        return

    st.write("### Analysis Results")
    st.write(f"Total Resumes Analyzed: {results['total_resumes']}")

    for result in results['analysis']:
        with st.expander(f"📄 {os.path.basename(result['resume_file'])} "
                        f"(Match Score: {result['match_score']}%)"):
            display_result_details(result)

def display_result_details(result):
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
```

## Code Examples

### 1. Async Operations

```python
async def analyze_resumes(job_requirements):
    with st.spinner("Analyzing resumes..."):
        try:
            results = await st.session_state.resume_analyzer.analyze_resumes(
                job_requirements
            )
            if results:
                st.success("Analysis completed!")
                display_analysis_results(results)
            else:
                st.warning("No results to display")
        except Exception as e:
            st.error(f"❌ Error analyzing resumes: {str(e)}")
            logger.error(f"Analysis error: {str(e)}")
```

### 2. File Processing

```python
def process_uploaded_resumes():
    with st.spinner("Processing resumes..."):
        try:
            success, message = st.session_state.resume_analyzer.process_new_resumes()
            if success:
                st.success(f"✅ {message}")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ {message}")
        except Exception as e:
            st.error(f"❌ Error processing resumes: {str(e)}")
```

### Best Practices

1. **State Management**

   - Initialize all session state variables at startup
   - Use session state for data persistence
   - Clear state when appropriate

2. **Error Handling**

   - Wrap operations in try-except blocks
   - Provide clear error messages
   - Use appropriate status indicators

3. **User Experience**

   - Show loading indicators for long operations
   - Provide clear feedback
   - Maintain consistent layout

4. **Performance**
   - Cache expensive computations
   - Handle large files efficiently
   - Use async operations for API calls

### Running the Application

To run the Streamlit application:

```bash
streamlit run app.py
```

This will start the web server and open the application in your default browser. The application will be accessible at `http://localhost:8501` by default.
