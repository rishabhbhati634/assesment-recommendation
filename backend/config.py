import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Perplexity API Configuration
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL")
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # File paths
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    PROCESSED_CSV = os.path.join(DATA_DIR, "processed_assessments_with_embeddings.csv")
    EMBEDDINGS_FILE = os.path.join(DATA_DIR, "assessments_embeddings.npy")
    FAISS_INDEX = os.path.join(DATA_DIR, "faiss_index.bin")
    
    # Recommendation settings
    MIN_RESULTS = 5
    MAX_RESULTS = 10
    
    # Feature flags
    USE_PERPLEXITY = True
    USE_CHUNKING = True
    
    @classmethod
    def validate(cls):
        if cls.USE_PERPLEXITY and not cls.PERPLEXITY_API_KEY:
            print("⚠️ WARNING: PERPLEXITY_API_KEY not set. Disabling Perplexity.")
            cls.USE_PERPLEXITY = False
        return cls.USE_PERPLEXITY