import os
from dotenv import load_dotenv
from rag.config import MAX_CONTEXT_LENGTHS

load_dotenv()

# Database configuration
CONNECTION_STRING = f"postgresql://postgres:postgres@localhost:5432/postgres"

# Embedding model
EMBEDDING_MODEL_NAME = "thenlper/gte-base"

# LLM model
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# Other settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
DATA_DIR = './data'
DOCS_URL = 'https://docs.ray.io/en/master/'
