import logging
import os

# Hide Hugging Face Hub & Tokenizer warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Suppress Transformers & Sentence-Transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("langchain_huggingface").setLevel(logging.ERROR)

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen3-0.6B"

# Text Chunking Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# System Paths
DATA_DIR = "data"
FAISS_INDEX_PATH = "faiss_index"

# PDF test file
PDF_FILENAME = "PHILIPPINE-HISTORY-SOURCE-BOOK-FINAL-SEP022021.pdf"

# LLM directory
MODELS_DIR = "models"

# Automatically create the data folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
