# Resume Analysis System

A system for analyzing resumes against job requirements using AI and vector similarity search.

## Features

- PDF, TXT, and CSV resume processing
- AI-powered resume analysis
- Vector similarity search using Pinecone
- Match scoring against job requirements
- Interactive web interface with Streamlit

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)
- OpenRouter API key
- Pinecone API key (free tier available)

## Dependencies

Key packages used in this project:

- `streamlit` - Web interface
- `langchain` - AI/LLM framework
- `langchain-pinecone` - Vector store integration
- `pinecone` - Vector database
- `sentence-transformers` - Text embeddings
- `deepseek` - LLM model via OpenRouter
- `pypdf` - PDF processing
- `torch` - Machine learning backend

## Development Setup

1. **Clone the repository**

   ```bash
   git clone [your-repo-url]
   cd Resume_Analyzer
   ```

2. **Create and activate a virtual environment**

   Windows:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS/Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install in development mode**

   ```bash
   pip install -e .
   ```

   The `-e` flag installs the package in "editable" mode, meaning:

   - Changes to the source code take effect immediately
   - No need to reinstall after code changes
   - Python will use your local development files
   - Great for testing and development

4. **Set up environment variables**

   Copy the environment template:

   ```bash
   cp env_template .env
   ```

   Edit `.env` and add your API keys:

   - OPENROUTER_API_KEY from [OpenRouter](https://openrouter.ai/)
   - PINECONE_API_KEY from [Pinecone](https://www.pinecone.io/)

## Running the Application

1. **Start the Streamlit app**

   ```bash
   streamlit run run.py
   ```

   Or alternatively:

   ```bash
   python -m streamlit run run.py
   ```

2. **Access the web interface**
   - Open your browser to `http://localhost:8501`
   - The first time you run the app, you'll need to:
     1. Initialize the system
     2. Upload some resumes
     3. Process the resumes
     4. Enter job requirements to analyze

## Development Workflow

1. **Make code changes**

   - Edit any Python files in the project
   - Changes take effect immediately due to `-e` installation
   - No need to restart Streamlit for most changes

2. **Test your changes**

   - Run the verify script to check dependencies:
     ```bash
     python verify.py
     ```
   - Use the web interface to test functionality

3. **View logs**
   - Check the Streamlit interface for basic logs
   - More detailed logs are in the terminal window

## Project Structure

```
resume_analyzer/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── models/
│   ├── __init__.py
│   └── deepseek_llm.py
├── services/
│   ├── __init__.py
│   ├── resume_analyzer.py
│   └── vector_store.py
├── utils/
│   ├── __init__.py
│   └── logging_config.py
└── app.py
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Troubleshooting

- If you see import errors, ensure you've installed the package with `pip install -e .`
- If API calls fail, check your `.env` file and API keys
- For Pinecone errors, verify your index settings in the configuration
- For package conflicts:
  ```bash
  # Clean reinstall (Windows)
  deactivate
  rmdir /s /q venv
  python -m venv venv
  .\venv\Scripts\activate
  pip install -e .
  ```
- For `ModuleNotFoundError`, ensure all dependencies are properly installed:
  ```bash
  pip install -r requirements.txt  # If available
  # or
  pip install -e .  # Uses setup.py
  ```

## License

MIT
