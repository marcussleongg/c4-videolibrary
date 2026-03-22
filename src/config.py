import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (secrets — belong in .env) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Paths ---
VIDEO_DIR = "/Volumes/T7 Shield/c4-videos/"
SEGMENTS_DIR = "/Volumes/T7 Shield/c4-segments/"

# --- OpenRouter API ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Model IDs (all via OpenRouter) ---
SCENE_MODEL = "google/gemini-3-flash-preview"
TRANSCRIPTION_MODEL = "google/gemini-3-flash-preview"
PROSODY_MODEL = "google/gemini-3-flash-preview"
EMBEDDING_MODEL = "openai/text-embedding-3-large"
RERANKER_MODEL = "anthropic/claude-sonnet-4.6"

# --- Segmentation ---
SEGMENT_LENGTH = 20
SEGMENT_OVERLAP = 5

# --- Retrieval ---
RRF_K = 60
TOP_K_RETRIEVAL = 50
TOP_K_RERANK = 10

# --- Pinecone ---
PINECONE_INDEX_NAME = "video-segments"
EMBEDDING_DIMENSION = 3072  # text-embedding-3-large
