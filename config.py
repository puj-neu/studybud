import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = BASE_DIR.joinpath('data', 'vector_store')

# Create directories
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Configure matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# LLM Settings
DEFAULT_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
RETRIEVER_K = 7
