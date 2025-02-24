import startup  # Import startup configuration first
import streamlit as st
import asyncio
import nest_asyncio
from typing import Dict, Any
import os
import sys
import warnings
import time
from typing import List, Optional, Union
import logging
import torch

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Disable PyTorch warnings and class inspection
import torch._classes
torch._classes.__getattr__ = lambda *args: None

# Import project modules
from config.settings import settings
from services.resume_analyzer import ResumeAnalyzer

def initialize_async_environment():
    """Initialize the async environment properly"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

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
    if 'async_loop' not in st.session_state:
        st.session_state.async_loop = initialize_async_environment()

async def initialize_resume_analyzer():
    """Asynchronously initialize the resume analyzer"""
    try:
        return ResumeAnalyzer(
            openrouter_api_key=settings.OPENROUTER_API_KEY,
            pinecone_api_key=settings.PINECONE_API_KEY
        )
    except Exception as e:
        st.error(f"Failed to initialize Resume Analyzer: {str(e)}")
        raise

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

def main():
    # Apply nest_asyncio for nested event loops
    nest_asyncio.apply()
    
    # Initialize session state
    initialize_session_state()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Resume Analysis System",
        page_icon="📄",
        layout="wide"
    )
    
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
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        # Initialize System Button
        if st.button("Initialize System", use_container_width=True):
            with st.spinner("Initializing..."):
                try:
                    loop = st.session_state.async_loop
                    st.session_state.resume_analyzer = loop.run_until_complete(
                        initialize_resume_analyzer()
                    )
                    st.success("✅ System initialized!")
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
                            success, message = st.session_state.resume_analyzer.process_new_resumes()
                            if success:
                                st.success(f"✅ {message}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"❌ Error processing resumes: {str(e)}")
        
        # Display stored resumes
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
                    loop = st.session_state.async_loop
                    results = loop.run_until_complete(
                        st.session_state.resume_analyzer.analyze_resumes(job_requirements)
                    )
                    if results:
                        st.success("Analysis completed!")
                        display_analysis_results(results)
                    else:
                        st.warning("No results to display")
                except Exception as e:
                    st.error(f"❌ Error analyzing resumes: {str(e)}")
        elif analyze_button:
            st.warning("Please enter job requirements first")
    else:
        st.info("👈 Please initialize the system using the sidebar controls")

if __name__ == "__main__":
    main()