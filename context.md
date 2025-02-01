# Resume Analyzer Project Context

## Overview

The Resume Analyzer project is a Python-based application designed to analyze resumes and provide insights. It includes a web interface component built with JavaScript.

## Key Files and Directories

### app.py

- **Purpose**: Main application file that likely initializes and runs the resume analysis service.
- **Role**: Entry point for the application.

### AnalysisDashboard.js

- **Purpose**: JavaScript file for the web interface, possibly handling the UI logic and data visualization.
- **Role**: Frontend component for displaying analysis results.

### config/

- **Purpose**: Contains configuration settings for the application.
- **Key Files**:
  - `settings.py`: Configuration settings for the application.

### models/

- **Purpose**: Placeholder for machine learning models used in the resume analysis.
- **Key Files**: Currently empty, but will likely contain model files in the future.

### resumes/

- **Purpose**: Contains sample resumes for testing and analysis.
- **Key Files**:
  - `Alex.txt`
  - `David.txt`
  - `Shashank.txt`

### services/

- **Purpose**: Contains service files that handle the core logic of the resume analysis.
- **Key Files**:
  - `document_processor.py`: Processes documents (resumes) for analysis.
  - `resume_analyzer.py`: Analyzes resumes and extracts relevant information.
  - `vector_store.py`: Manages vector storage for resume data.

### utils/

- **Purpose**: Contains utility files that support the application.
- **Key Files**:
  - `logging_config.py`: Configuration for logging within the application.

## Additional Files

- `__init__.py`: Initializes Python packages.
- `.gitignore`: Specifies files and directories to ignore in version control.
- `env_template`: Template for environment variables.
- `LICENSE`: License information for the project.
- `Merge.py`: Script for merging projects.
- `merged_project.txt`: Output of merged projects.
- `projectstructure.md`: Documentation of the project structure.
- `ReadMe.md`: General documentation and instructions for the project.
- `run.py`: Script for running the application.
- `setup.py`: Script for setting up the application.
- `test.py`: Script for running tests.
- `verify.py`: Script for verifying the application.

## Usage

To run the application, execute the `run.py` script. For development, you might need to set up the environment variables using `env_template` and install dependencies using `setup.py`.

## Development

The project is structured to separate concerns, with services handling the core logic, utilities providing support functions, and configuration managing settings. This separation makes the project easier to maintain and extend.

### Initialize System

- **Trigger**: Clicking the "Initialize System" button in the Streamlit interface.
- **Functionality**: Initializes the `ResumeAnalyzer` by creating an instance of it with the required API keys (`OPENROUTER_API_KEY` and `PINECONE_API_KEY`) and stores it in the Streamlit session state.
- **Purpose**: Prepares the system for resume analysis by setting up the necessary services and configurations.
