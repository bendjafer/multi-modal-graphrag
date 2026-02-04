"""Configuration management using environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Model Configuration
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
ENTITY_EXTRACTION_MODEL = os.getenv("ENTITY_EXTRACTION_MODEL", "sciphi/triplex")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen2.5:14b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Image filtering parameters
MIN_IMAGE_SIZE = (
    int(os.getenv("MIN_IMAGE_WIDTH", "100")),
    int(os.getenv("MIN_IMAGE_HEIGHT", "100"))
)
MAX_IMAGE_SIZE = (
    int(os.getenv("MAX_IMAGE_WIDTH", "4000")),
    int(os.getenv("MAX_IMAGE_HEIGHT", "4000"))
)

# Path Configuration
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", "results")
IMAGES_SUBDIR = os.getenv("IMAGES_SUBDIR", "images")
MARKDOWN_SUBDIR = os.getenv("MARKDOWN_SUBDIR", "markdown")
INPUT_PDF_FOLDER = os.getenv("INPUT_PDF_FOLDER", "dummy_pdfs")

# Rate Limiting Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
BATCH_PAUSE = int(os.getenv("BATCH_PAUSE", "60"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
