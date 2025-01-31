# resume_analyzer/config/settings.py
import os
from dotenv import load_dotenv

class Settings:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # API Keys
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        
        # DeepSeek Model Settings
        self.DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-r1:nitro')
        self.DEEPSEEK_TEMPERATURE = float(os.getenv('DEEPSEEK_TEMPERATURE', '0.5'))
        self.DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://openrouter.ai/api/v1')
        
        # Pinecone Settings
        self.PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'resume-analysis')
        self.PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '768'))
        self.PINECONE_METRIC = os.getenv('PINECONE_METRIC', 'cosine')
        self.PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
        self.PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
        
        # Embeddings Settings
        self.EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        self.EMBEDDINGS_DEVICE = os.getenv('EMBEDDINGS_DEVICE', 'cpu')
        
        # Document Processing Settings
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '100'))
        
        # Data Directory
        self.RESUME_DIR = os.getenv('RESUME_DIR', 'resumes')

# Create a single instance to be imported by other modules
settings = Settings()

# Make settings available for import
__all__ = ['settings']