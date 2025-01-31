# app.py
import os
import sys
import asyncio
import streamlit as st
import warnings
import time
from typing import Dict, Any
import json  # Add this import
import streamlit.components.v1 as components


# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now use absolute imports
from config.settings import settings
from utils.logging_config import logger
from services.resume_analyzer import ResumeAnalyzer

# Suppress torch warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')

def log_to_console(message: str):
    js_code = f"""
    <script>
        console.log({json.dumps(message)});
    </script>
    """
    components.html(js_code)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'resume_analyzer' not in st.session_state:
        st.session_state.resume_analyzer = None
    if 'job_requirements' not in st.session_state:
        st.session_state.job_requirements = ""
    if 'refresh_key' not in st.session_state:
        st.session_state.refresh_key = 0

def handle_file_upload(uploaded_files):
    """Handle uploaded resume files"""
    if uploaded_files:
        os.makedirs(settings.RESUME_DIR, exist_ok=True)
        for file in uploaded_files:
            with open(os.path.join(settings.RESUME_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"✅ Saved {len(uploaded_files)} resumes")

def display_stored_resumes():
    """Display and manage stored resumes"""
    stored_resumes = st.session_state.resume_analyzer.list_stored_resumes()
    if stored_resumes:
        st.write(f"Stored Resumes: {len(stored_resumes)}")
        for resume in stored_resumes:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(os.path.basename(resume))
            with col2:
                if st.button("Delete", key=f"del_{resume}_{st.session_state.refresh_key}"):
                    if st.session_state.resume_analyzer.delete_resume(resume):
                        st.success("Resume deleted")
                        st.session_state.refresh_key += 1
                        time.sleep(2)
                        st.rerun()
    else:
        st.info("No resumes in database")

def display_analysis_results(results: Dict[str, Any]):
    """Display resume analysis results"""
    st.header(f"Analysis Results ({results['total_resumes']} resumes)")
    
    for i, result in enumerate(results['analysis'], 1):
        with st.expander(
            f"#{i} - {os.path.basename(result['resume_file'])} "
            f"(Match Score: {result['match_score']}%)"
        ):
            st.markdown("### Summary")
            st.write(result['summary'])
            
            st.markdown("### Matching Qualifications")
            for qual in result['qualifications_match']:
                st.markdown(f"- {qual}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Missing Requirements")
                for req in result['missing_requirements']:
                    st.markdown(f"- {req}")
            
            with col2:
                st.markdown("### Additional Skills")
                for skill in result['additional_skills']:
                    st.markdown(f"- {skill}")
            
            st.markdown(f"**Years of Relevant Experience:** {result['years_experience']}")

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Resume Analysis System",
        page_icon="📄",
        layout="wide"
    )
    st.write("## System Status")
    st.write(f"OpenRouter Key: {'✅' if settings.OPENROUTER_API_KEY else '❌'}")
    st.write(f"Pinecone Key: {'✅' if settings.PINECONE_API_KEY else '❌'}")
    
    # Usage in your code:
    log_to_console("System initialized successfully")
    
    initialize_session_state()
    
    st.title("📄 Resume Analysis System")
    
    with st.sidebar:
        # Database Management Section
        if st.session_state.resume_analyzer:
            st.header("Resume Database")
            display_stored_resumes()
        
        st.header("System Controls")
        
        uploaded_files = st.file_uploader(
            "Upload Resumes (PDF, TXT)", 
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        handle_file_upload(uploaded_files)
        
        if st.button("Initialize System", use_container_width=True):
            with st.spinner("Initializing..."):
                try:
                    st.session_state.resume_analyzer = ResumeAnalyzer(
                        openrouter_api_key=settings.OPENROUTER_API_KEY,
                        pinecone_api_key=settings.PINECONE_API_KEY
                    )
                    st.success("✅ System initialized!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if st.session_state.resume_analyzer and st.button("Process Resumes", use_container_width=True):
            with st.spinner("Processing resumes..."):
                success, message = st.session_state.resume_analyzer.process_resumes()
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.refresh_key += 1
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
    
    if st.session_state.resume_analyzer:
        st.header("Job Requirements")
        job_requirements = st.text_area(
            "Enter the job requirements:",
            value=st.session_state.job_requirements,
            height=200
        )
        
        if st.button("Analyze Resumes", type="primary"):
            if not job_requirements:
                st.warning("Please enter job requirements first")
            else:
                st.session_state.job_requirements = job_requirements
                with st.spinner("Analyzing resumes..."):
                    try:
                        results = asyncio.run(
                            st.session_state.resume_analyzer.analyze_resumes(job_requirements)
                        )
                        display_analysis_results(results)
                    except Exception as e:
                        st.error(f"❌ Error analyzing resumes: {str(e)}")
    else:
        st.info("👈 Please initialize the system using the sidebar controls")

if __name__ == "__main__":
    main()