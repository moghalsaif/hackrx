import os
from pathlib import Path
from typing import Dict, Any, Optional

class RAGFlowConfig:
    """Configuration class for the RAGFlow Insurance Policy System"""
    
    # Project Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
    
    # Document Processing
    PDF_FILE = "BAJHLIP23020V012223.pdf"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    MIN_CHUNK_SIZE = 100
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    EMBEDDING_DIMENSION = 768
    
    # Vector Database Configuration
    VECTOR_DB_TYPE = "chroma"  # Options: chroma, pinecone, milvus
    COLLECTION_NAME = "insurance_policy_chunks"
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")  # Options: openai, anthropic, local, ollama
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    
    # Local Model Configuration
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", str(MODELS_DIR / "llama3"))
    LOCAL_MODEL_TYPE = os.getenv("LOCAL_MODEL_TYPE", "transformers")  # Options: transformers, ollama, llamacpp
    LOCAL_MODEL_DEVICE = os.getenv("LOCAL_MODEL_DEVICE", "auto")  # Options: auto, cpu, cuda
    LOCAL_MODEL_PRECISION = os.getenv("LOCAL_MODEL_PRECISION", "fp16")  # Options: fp32, fp16, int8, int4
    
    # Ollama Configuration (if using Ollama)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")
    
    # Model Loading Configuration
    MAX_MEMORY_GB = int(os.getenv("MAX_MEMORY_GB", "8"))
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Query Processing
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
    ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    
    # Output Configuration
    OUTPUT_FORMAT = "json"
    INCLUDE_CONFIDENCE_SCORES = True
    REQUIRE_CLAUSE_REFERENCES = True
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_AUDIT_LOGGING = os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR, cls.VECTOR_DB_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    @classmethod
    def get_openai_api_key(cls) -> Optional[str]:
        """Get OpenAI API key from environment"""
        return os.getenv("OPENAI_API_KEY")
    
    @classmethod
    def get_anthropic_api_key(cls) -> Optional[str]:
        """Get Anthropic API key from environment"""
        return os.getenv("ANTHROPIC_API_KEY")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        if cls.LLM_PROVIDER == "openai" and not cls.get_openai_api_key():
            raise ValueError("OpenAI API key not found in environment variables")
        
        if cls.LLM_PROVIDER == "anthropic" and not cls.get_anthropic_api_key():
            raise ValueError("Anthropic API key not found in environment variables")
        
        if cls.LLM_PROVIDER == "local":
            if cls.LOCAL_MODEL_TYPE == "transformers" and not Path(cls.LOCAL_MODEL_PATH).exists():
                print(f"Warning: Local model path not found: {cls.LOCAL_MODEL_PATH}")
                print("Please ensure you have downloaded and placed the Llama 3 model files.")
        
        if not Path(cls.PDF_FILE).exists():
            raise FileNotFoundError(f"PDF file not found: {cls.PDF_FILE}")
        
        return True
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            "provider": cls.LLM_PROVIDER,
            "model": cls.LLM_MODEL,
            "temperature": cls.LLM_TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "local_path": cls.LOCAL_MODEL_PATH,
            "local_type": cls.LOCAL_MODEL_TYPE,
            "device": cls.LOCAL_MODEL_DEVICE,
            "precision": cls.LOCAL_MODEL_PRECISION,
            "use_gpu": cls.USE_GPU,
            "max_memory_gb": cls.MAX_MEMORY_GB,
            "ollama_url": cls.OLLAMA_BASE_URL,
            "ollama_model": cls.OLLAMA_MODEL_NAME
        }

# Create directories on import
RAGFlowConfig.create_directories() 