import os
import datetime

def merge_python_files(root_dir: str, output_file: str) -> None:
    """
    Merge all Python files in the project into a single file.
    
    Args:
        root_dir: Root directory of the project
        output_file: Output file path
    """
    
    # Header template for the merged file
    header = f'''"""
Resume Analyzer - Complete Project
Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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

'''

    # Order of directories to process (to handle dependencies)
    dir_order = [
        'config',
        'utils',
        'models',
        'services',
        ''  # Root directory
    ]
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write header
        outfile.write(header + '\n')
        
        # Process each directory in order
        for directory in dir_order:
            current_dir = os.path.join(root_dir, directory)
            if not os.path.exists(current_dir):
                continue
                
            # Write section header
            outfile.write(f'\n# ====================== {directory.upper() if directory else "ROOT"} ======================\n\n')
            
            # Get all Python files in current directory
            python_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and f != 'merge_project.py']
            
            # Process each Python file
            for py_file in python_files:
                file_path = os.path.join(current_dir, py_file)
                
                # Write file header
                outfile.write(f'# File: {os.path.join(directory, py_file)}\n')
                
                # Read and write file content
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                    # Remove existing imports
                    lines = content.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if not (line.startswith('import ') or line.startswith('from ')):
                            filtered_lines.append(line)
                    
                    outfile.write('\n'.join(filtered_lines) + '\n\n')

if __name__ == "__main__":
    # Project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Output file path
    output_path = os.path.join(project_root, 'resume_analyzer_complete.py')
    
    try:
        merge_python_files(project_root, output_path)
        print(f"Successfully merged all Python files into: {output_path}")
    except Exception as e:
        print(f"Error merging files: {str(e)}")