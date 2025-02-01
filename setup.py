from setuptools import setup, find_packages

setup(
    name="resume_analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "python-dotenv",
        "langchain",
        "langchain-community",
        "langchain-core",
        "pinecone-client",
        "sentence-transformers",
        "transformers",
        "torch",
        "pypdf",
        "huggingface-hub",
        "nltk",
        "unstructured",  # For document loading
        "markdown",      # For markdown processing
    ],
)