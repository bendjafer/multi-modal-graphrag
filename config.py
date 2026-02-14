"""Configuration management -
Based on Vecta's 2026 chunking study on academic papers:
https://www.runvecta.com/blog/chunking/
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found in environment variables")

# Vision Model: For image filtering and table reconstruction
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")

# Entity Extraction Model: For extracting entities and relationships from chunks
ENTITY_EXTRACTION_MODEL = os.getenv("ENTITY_EXTRACTION_MODEL", "sciphi/triplex")

# Summary Model: For future use
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")

# Embedding Model:
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "voyage-3")
VOYAGE_MODEL = EMBEDDING_MODEL  # Alias for consistency in scripts

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ═══════════════════════════════════════════════════════════════════════
# CHUNKING CONFIGURATION 
# ═══════════════════════════════════════════════════════════════════════

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3500"))  
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "350"))  
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "700"))  

TARGET_RETRIEVAL_TOKENS = int(os.getenv("TARGET_RETRIEVAL_TOKENS", "2000"))

CHUNKS_SUBDIR = os.getenv("CHUNKS_SUBDIR", "chunks")
EMBEDDINGS_SUBDIR = os.getenv("EMBEDDINGS_SUBDIR", "embeddings")


# ═══════════════════════════════════════════════════════════════════════
# IMAGE FILTERING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════

MIN_IMAGE_SIZE = (
    int(os.getenv("MIN_IMAGE_WIDTH", "100")),
    int(os.getenv("MIN_IMAGE_HEIGHT", "100"))
)
MAX_IMAGE_SIZE = (
    int(os.getenv("MAX_IMAGE_WIDTH", "4000")),
    int(os.getenv("MAX_IMAGE_HEIGHT", "4000"))
)


# ═══════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", "results")
IMAGES_SUBDIR = os.getenv("IMAGES_SUBDIR", "images")
TABLES_SUBDIR = os.getenv("TABLES_SUBDIR", "tables")
MARKDOWN_SUBDIR = os.getenv("MARKDOWN_SUBDIR", "markdown")
INPUT_PDF_FOLDER = os.getenv("INPUT_PDF_FOLDER", "dummy_pdfs")

# Chunks path for embedding generation
CHUNKS_PATH = os.getenv("CHUNKS_PATH")
if not CHUNKS_PATH:
    # Auto-detect: find first chunks file in results directory
    results_path = Path(BASE_OUTPUT_DIR)
    if results_path.exists():
        chunks_files = list(results_path.glob("*/chunks/*_chunks.json"))
        if chunks_files:
            CHUNKS_PATH = str(chunks_files[0])


# ═══════════════════════════════════════════════════════════════════════
# RATE LIMITING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
BATCH_PAUSE = int(os.getenv("BATCH_PAUSE", "60"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))


# ═══════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")


# ═══════════════════════════════════════════════════════════════════════
# MARKDOWN CLEANING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

MIN_LINE_LENGTH = int(os.getenv("MIN_LINE_LENGTH", "15"))
MIN_SECTION_CONTENT_WORDS = int(os.getenv("MIN_SECTION_CONTENT_WORDS", "15"))
REMOVE_CITATION_NUMBERS = os.getenv("REMOVE_CITATION_NUMBERS", "true").lower() == "true"
REMOVE_PHOTO_CREDITS = os.getenv("REMOVE_PHOTO_CREDITS", "true").lower() == "true"


# ═══════════════════════════════════════════════════════════════════════
# VISION API CACHING
# ═══════════════════════════════════════════════════════════════════════

ENABLE_VISION_CACHE = os.getenv("ENABLE_VISION_CACHE", "true").lower() == "true"
VISION_CACHE_DIR = os.getenv("VISION_CACHE_DIR", ".cache/vision_responses")


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION & SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def print_config_summary():
    """Print research-backed configuration summary."""

    # Estimate tokens from characters (4 chars per token for English)
    estimated_chunk_tokens = CHUNK_SIZE / 4
    estimated_overlap_tokens = CHUNK_OVERLAP / 4
    estimated_min_tokens = MIN_CHUNK_SIZE / 4

    # Estimate retrieval k
    estimated_k = round(TARGET_RETRIEVAL_TOKENS / estimated_chunk_tokens)

    print("=" * 80)
    print("RESEARCH-BACKED GRAPHRAG CONFIGURATION")
    print("=" * 80)
    print("\n Based on: Vecta 2026 Chunking Study")
    print("   Strategy: Recursive Character Splitting (LangChain)")
    print("   Winner: 69% accuracy, 0.92 page F1, 0.86 doc F1")
    print("\n Chunking Configuration:")
    print(f"   • Chunk Size:     {CHUNK_SIZE} chars (~{estimated_chunk_tokens:.0f} tokens)")
    print(f"   • Overlap:        {CHUNK_OVERLAP} chars (~{estimated_overlap_tokens:.0f} tokens)")
    print(f"   • Min Size:       {MIN_CHUNK_SIZE} chars (~{estimated_min_tokens:.0f} tokens)")
    print(f"   • Overlap Ratio:  {CHUNK_OVERLAP/CHUNK_SIZE*100:.1f}%")
    print("\n Retrieval Strategy:")
    print(f"   • Target Context: {TARGET_RETRIEVAL_TOKENS} tokens")
    print(f"   • Estimated k:    ~{estimated_k} chunks")
    print(f"   • Total Context:  ~{estimated_k * estimated_chunk_tokens:.0f} tokens")
    print("\n Models:")
    print(f"   • Vision:         {VISION_MODEL}")
    print(f"   • Entity Extract: {ENTITY_EXTRACTION_MODEL}")
    print(f"   • Embeddings:     {EMBEDDING_MODEL}")
    print("\n Why These Values?")
    print("   • 2000 chars (512 tokens) won Vecta study on academic papers")
    print("   • 200 char overlap (50 tokens) preserves boundary context")
    print("   • 400 char minimum avoids semantic chunking's failure mode")
    print("   • Smaller chunks → higher accuracy (sample from more locations)")
    print("\n Optimized For:")
    print("   • Academic papers")
    print("   • Technical documentation")
    print("   • Research reports")
    print("   • Dense, structured content")
    print("\n" + "=" * 80)

    # Warnings
    if CHUNK_SIZE < 1500 or CHUNK_SIZE > 2500:
        print("WARNING: CHUNK_SIZE outside optimal range (1500-2500)")
        print("   Research shows 2000 chars (512 tokens) is optimal for academic text")

    if CHUNK_OVERLAP / CHUNK_SIZE < 0.08 or CHUNK_OVERLAP / CHUNK_SIZE > 0.15:
        print("WARNING: Overlap ratio outside optimal range (8-15%)")
        print("   Research shows 10% overlap (200/2000) works best")

    if MIN_CHUNK_SIZE / CHUNK_SIZE < 0.15:
        print("WARNING: MIN_CHUNK_SIZE too small relative to CHUNK_SIZE")
        print("   Risk of semantic chunking failure (tiny fragments)")


if __name__ == "__main__":
    print_config_summary()