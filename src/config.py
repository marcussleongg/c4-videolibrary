import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (secrets — belong in .env) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Paths ---
VIDEO_DIR = "/Volumes/T7 Shield/c4-videos/"
SEGMENTS_DIR = "/Volumes/T7 Shield/c4-segments/"

# --- Model IDs (all via Gemini API) ---
SCENE_MODEL = "gemini-3-flash-preview"
TRANSCRIPTION_MODEL = "gemini-3-flash-preview"
PROSODY_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "gemini-embedding-001"
RERANKER_MODEL = "gemini-3-flash-preview"

# --- Segmentation ---
SEGMENT_LENGTH = 20
SEGMENT_OVERLAP = 5

# --- Retrieval ---
RRF_K = 60
TOP_K_RETRIEVAL = 50
TOP_K_RERANK = 10

# --- Pinecone ---
PINECONE_INDEX_NAME = "video-segments"
EMBEDDING_DIMENSION = 3072  # gemini-embedding-001
