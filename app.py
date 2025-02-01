import os
import sys
import asyncio
import streamlit as st
import warnings
import time
from typing import Dict, Any

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now use absolute imports
from config.settings import settings
from utils.logging_config import logger
from services.resume_analyzer import ResumeAnalyzer

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