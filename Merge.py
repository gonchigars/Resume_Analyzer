import os
from pathlib import Path
from typing import List, Set
import datetime

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