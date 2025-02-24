from setuptools import setup, find_packages

setup(
    name="resume_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.29.0",
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.10",
        "langchain-pinecone>=0.0.2",
        "langchain-huggingface>=0.0.2",  # Added for HuggingFace integration
        "pinecone-client>=3.0.0",  # Updated package name and version
        "sentence-transformers>=2.2.2",
        "transformers>=4.36.0",
        "--extra-index-url https://download.pytorch.org/whl/cu121",  # Added PyTorch CUDA channel
        "torch==2.2.0",  # Fixed version
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        "pypdf>=3.17.0",
        "huggingface-hub>=0.20.0",
        "nltk>=3.8.1",
        "unstructured>=0.11.0",
        "markdown>=3.5.0",
        "aiohttp>=3.9.0",
        "nest-asyncio>=1.5.8",
        "asyncio>=3.4.3",
        "packaging>=23.2",  # Added for better package compatibility
        "typing-extensions>=4.9.0",  # Added for type hints support
    ],
    python_requires=">=3.9,<3.12",  # Restricted Python version range
)