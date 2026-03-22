import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Paths ---
VIDEO_DIR = os.getenv("VIDEO_DIR", "/Volumes/T7 Shield/c4-videos/")
SEGMENTS_DIR = os.getenv("SEGMENTS_DIR", "/Volumes/T7 Shield/c4-segments/")

# --- Model IDs (all via OpenRouter) ---
SCENE_MODEL = os.getenv("SCENE_MODEL", "google/gemini-3-flash-preview")
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "google/gemini-3-flash-preview")
PROSODY_MODEL = os.getenv("PROSODY_MODEL", "google/gemini-3-flash-preview")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "anthropic/claude-sonnet-4.6")

# --- OpenRouter API ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Segmentation ---
SEGMENT_LENGTH = int(os.getenv("SEGMENT_LENGTH", "20"))
SEGMENT_OVERLAP = int(os.getenv("SEGMENT_OVERLAP", "5"))

# --- Retrieval ---
RRF_K = int(os.getenv("RRF_K", "60"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "50"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "10"))

# --- Pinecone ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "video-segments")
EMBEDDING_DIMENSION = 1536  # text-embedding-3-small
