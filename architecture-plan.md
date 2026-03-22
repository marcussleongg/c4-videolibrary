# Video Library Search System — Architecture Plan

## Context
Build a system to make ~38 body-worn camera videos (30min–2hrs each) searchable via natural language. All model inference via OpenRouter (no local model inference). Video files stay local; only text/embeddings are stored in the cloud. Deliverable: Python + Jupyter notebooks.

The system must support **temporal reasoning** — understanding events, actions, and transitions across time, not just static frame-level properties (retrieval of "the moment the person in the red jacket started shouting" not just "frames containing a red jacket").

---

## Core Decisions

1. **Single VLM for ingestion, specialized models for retrieval** — one Gemini Flash call per segment handles scene description, transcription, and prosody. Different models handle embeddings and reranking.
2. **Video clips sent directly to VLM** — Gemini Flash accepts native video input, enabling temporal understanding of actions and events (not just keyframe descriptions)
3. **Text-mediated visual search** — VLM generates rich scene/event descriptions from video, embed as text vectors
4. **Always-on dual retrieval: FTS + vector search** — every query runs both Supabase FTS (keyword matching) and Pinecone vector search (semantic similarity) in parallel. Results fused with Reciprocal Rank Fusion (RRF). No query routing needed — RRF naturally lets the right signal dominate.
5. **Granular hotswappable components** — separate Protocol interfaces for scene description, transcription, and prosody. Default implementation combines all three in one Gemini call (fast path), but each can be independently swapped to a specialized provider
6. **No separate OCR pass** — the VLM scene description notes visible text/signs/plates as part of its output. Dedicated per-frame OCR is unnecessary since we only need to locate footage containing text, not read it precisely

---

## Model Selection (All via OpenRouter)

| Task | Model | Why this model? | Cost |
|------|-------|----------------|------|
| **Scene Description** | Gemini Flash (video input) | Accepts native video. Captures temporal events, actions, people, objects. Also notes visible text/signs/plates as part of description | Combined call: ~$10-15 for 6K segments |
| **Transcription** | Gemini Flash (video input) | Native audio support, 3.1% WER. Swappable: Whisper via Groq | (included in combined call) |
| **Prosody Analysis** | Gemini Flash (video input) | Native audio support for tone/volume detection. Swappable: dedicated audio models | (included in combined call) |
| **Text Embeddings** | OpenAI text-embedding-3-small | Standard, cheap, good quality | ~$0.50 |
| **Reranking** | Claude Sonnet 4 | Strong reasoning to judge relevance of candidates | ~$0.02-0.05/query |

**Hybrid approach**: Scene description, transcription, and prosody each have their own Protocol interface, making them independently swappable. The default `GeminiVideoAnalyzer` combines all three into a **single Gemini Flash call per segment** (fast path — cheaper, fewer API calls). But if evaluation shows a specialized model is better for one task (e.g., Whisper for transcription), we can swap just that provider — the pipeline will make separate calls only for the swapped component.

---

## Cloud Services

| Concern | Service | Why | Free Tier? |
|---------|---------|-----|------------|
| **Model Inference** | OpenRouter | Single API key, access to all models | Pay-per-use |
| **Vector Database** | Pinecone serverless | Semantic similarity search for abstract queries | 2GB free |
| **SQL Database** | Supabase Postgres (FTS) | Structured data, full-text search, metadata filtering. Handles most queries without vector search. | 500MB free |

---

## Architecture

### Hotswappable Provider Design

```python
# --- Granular Protocols (each independently swappable) ---

class SceneDescriber(Protocol):
    def describe(self, video_bytes: bytes) -> SceneDescription: ...
    # Returns: scene_description, time_of_day, events list

class TranscriptionProvider(Protocol):
    def transcribe(self, audio_bytes: bytes) -> TranscriptResult: ...
    # Returns: timestamped transcript text

class ProsodyAnalyzer(Protocol):
    def analyze_prosody(self, audio_bytes: bytes) -> ProsodyResult: ...
    # Returns: has_shouting, max_volume, emotional_tones, background_sounds

class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

# --- Default Fast-Path Implementation ---

class GeminiVideoAnalyzer:
    """Implements SceneDescriber + TranscriptionProvider + ProsodyAnalyzer
    in a SINGLE Gemini Flash call. One API call returns all three outputs.

    If any individual provider is overridden in config, the pipeline
    calls Gemini for the remaining tasks and the specialized provider
    for the swapped task. E.g., if TranscriptionProvider is swapped to
    WhisperTranscriber, the pipeline makes:
      - 1 Gemini call (scene + prosody only)
      - 1 Whisper call (transcription only)
    """
    def describe(self, video_bytes): ...
    def transcribe(self, audio_bytes): ...
    def analyze_prosody(self, audio_bytes): ...

# All providers instantiated via config — swap by changing config, not code
```

### What Each Component Sees

```
Segment (10-60s of body-cam footage)
  │
  └─→ VIDEO CLIP (.mp4 with audio)
        Sent to: Gemini Flash (native video input)
        VLM sees: motion, temporal events, audio, visual scene
        VLM can describe: "officer handcuffs suspect" (action over time)
                           "driver exits vehicle" (temporal event)
                           "voice escalates at 0:18" (audio-visual correlation)
                           "license plate visible on silver sedan" (text presence)
```

### Ingestion Pipeline

```
Video File (30min-2hrs)
  │
  ├─→ [ffmpeg locally] — fixed-length splitting, no model inference
  │     20s segments with 5s overlap (each segment advances 15s)
  │     Overlap gives VLM context at boundaries to avoid mid-sentence cuts
  │     Produces per segment:
  │       - {filename}_seg_NNN.mp4 (video clip with audio, stored on external drive)
  │
  ├─→ Per segment (all via OpenRouter, parallelized):
  │     │
  │     ├─→ [SceneDescriber + TranscriptionProvider + ProsodyAnalyzer]
  │     │
  │     │     DEFAULT (fast path — GeminiVideoAnalyzer):
  │     │       ONE Gemini Flash call with video clip + audio
  │     │       Prompt: "Analyze this body-cam footage segment.
  │     │         EVENTS: What happens, in temporal order, with timestamps
  │     │         PEOPLE: Everyone visible — clothing colors/types, actions, position
  │     │         VEHICLES: Type, color, make/model, notable features
  │     │         OBJECTS: Everything visible, including small/background items
  │     │         ENVIRONMENT: Indoor/outdoor, lighting, time of day, weather
  │     │         AUDIO: Transcribe all speech verbatim with timestamps.
  │     │                Note shouting, raised voices, tone changes.
  │     │                Note background sounds (sirens, engines, radio).
  │     │         Note any visible text, signs, or license plates."
  │     │
  │     │     SWAPPED EXAMPLE (TranscriptionProvider → WhisperTranscriber):
  │     │       Gemini Flash call (scene + prosody only, no transcription)
  │     │       + Whisper call (audio → transcript)
  │     │       = 2 calls instead of 1, but better transcription quality
  │     │
  │     │     Output (regardless of which providers are used): {
  │     │       scene_description: "Officer approaches silver sedan...",
  │     │       transcript: "[00:05] Driver: What did I do? [00:08] Officer: ...",
  │     │       prosody: { has_shouting: false, max_volume: "normal",
  │     │                  emotional_tones: ["calm", "nervous"],
  │     │                  background_sounds: ["engine idling", "radio"] },
  │     │       time_of_day: "night"
  │     │     }
  │     │
  │     ├─→ [Timestamp Conversion]
  │     │     VLM returns timestamps relative to the segment (e.g., [00:18])
  │     │     Convert to absolute timestamps in the original video:
  │     │       absolute_time = segment.start_s + relative_time
  │     │     e.g., segment starts at 1390s → [00:18] becomes [23:28]
  │     │     All timestamps in transcript and events are converted
  │     │     before storing — everything in the database is absolute.
  │     │
  │     ├─→ [EmbeddingProvider → OpenAI]
  │     │     Concatenate: scene_description + transcript
  │     │     → single text embedding vector
  │     │
  │     └─→ [Store in both databases]
  │           → Pinecone: embedding vector + metadata
  │           → Supabase: full segment record with all text fields
```

### Storage Layer — What Goes Where

**Supabase Postgres** (primary store — handles most queries):
```sql
CREATE TABLE segments (
  id              TEXT PRIMARY KEY,    -- "traffic_stop_seg_47"
  file_name       TEXT NOT NULL,       -- "traffic_stop.mp4" (joined with VIDEO_DIR config at query time)
  start_s         INTEGER NOT NULL,    -- 1390
  end_s           INTEGER NOT NULL,    -- 1435
  scene_description TEXT,              -- rich temporal description from VLM (includes visible text/signs/plates)
  transcript      TEXT,                -- verbatim speech with timestamps
  prosody_json    JSONB,               -- { has_shouting, max_volume, tones, sounds }
  time_of_day     TEXT,                -- "day" / "night" / "dusk" / "dawn"
  has_shouting    BOOLEAN,             -- derived from prosody for fast filtering
  -- FTS index:
  search_vector   TSVECTOR             -- auto-generated from scene_desc + transcript
);

CREATE INDEX idx_segments_fts ON segments USING GIN(search_vector);
CREATE INDEX idx_segments_time ON segments(time_of_day);
CREATE INDEX idx_segments_shouting ON segments(has_shouting);
CREATE INDEX idx_segments_file ON segments(file_name);
```

Supabase handles:
- **Full-text search** (FTS) on transcript, scene_description
- **Metadata filtering** (time_of_day, has_shouting, file_name)
- **Combined queries** ("license plates at night" = FTS on scene_description + filter time_of_day)
- **All structured data** — timestamps, file names, prosody JSON
- **Fast for exact/keyword queries** — < 10ms on 6K rows

**Pinecone** (vector search — only for semantic/abstract queries):
```
{
  id: "traffic_stop_seg_47",
  vector: [0.023, -0.118, ...],   // embedding of concatenated text
  metadata: {
    file_name: "traffic_stop.mp4",
    start_s: 1390,
    end_s: 1435,
    has_shouting: false,
    time_of_day: "night"
  }
}
```

Pinecone handles:
- **Semantic similarity** for abstract queries ("tense confrontation", "someone looking nervous")

### Retrieval + Fusion

No query routing — every query always runs both retrieval paths in parallel. RRF naturally lets the right signal dominate (FTS for keyword queries, vector for semantic queries).

```
Natural Language Query
  │
  ├─→ [Parallel Retrieval — always both paths]
  │     │
  │     ├─→ Supabase FTS: keyword/phrase match on search_vector
  │     │     Fast (< 10ms). Catches exact matches.
  │     │     "Miranda rights" → finds segments where transcript contains those words
  │     │
  │     ├─→ Supabase SQL: metadata filters (extracted from query)
  │     │     time_of_day = 'night', has_shouting = true, etc.
  │     │
  │     └─→ Pinecone: embed query → semantic similarity
  │           Slower (~100ms) but catches semantic matches.
  │           "tense confrontation" → finds "driver exits aggressively, officer steps back"
  │
  └─→ [Fusion + Reranking]
        ├─→ Reciprocal Rank Fusion: score = Σ(1/(k+rank_i)) across all result sets
        │     FTS results dominate for keyword queries (precise matches score high)
        │     Vector results dominate for abstract queries (FTS returns nothing)
        │     Metadata filters applied as hard constraints (not scored)
        ├─→ Top-K → Claude Sonnet reranker
        │     Sends scene_description + transcript of top candidates
        │     LLM re-scores relevance to original query
        └─→ Return: [(video_id, start_s, end_s, score, evidence_text)]
```

---

## Project Structure

```
c4-videolibrary/
├── notebooks/
│   └── 00_evaluate_models.ipynb  # Model comparison tests
├── src/
│   ├── __init__.py
│   ├── config.py                 # Model IDs, API keys (env vars), paths, thresholds
│   ├── segmenter.py              # ffmpeg: fixed-length splitting with overlap
│   ├── providers/                # Hotswappable provider interfaces
│   │   ├── __init__.py
│   │   ├── base.py               # Protocol classes: SceneDescriber, TranscriptionProvider, etc.
│   │   ├── video_analyzer.py     # GeminiVideoAnalyzer: combined fast-path (scene+transcript+prosody)
│   │   ├── transcription.py      # Swap-in alternatives (e.g., WhisperTranscriber)
│   │   ├── prosody.py            # Swap-in alternatives (dedicated prosody analyzers)
│   │   └── embedding.py          # OpenAIEmbedder (default). Swappable: Voyage, Cohere
│   ├── ingestion.py              # Orchestrates: segment → analyze → embed → store
│   ├── vector_store.py           # Pinecone client
│   ├── sql_store.py              # Supabase Postgres client (FTS + metadata)
│   ├── retriever.py              # Always-on dual retrieval (FTS + vector) + RRF fusion
│   ├── reranker.py               # Claude Sonnet: rerank top-K
│   └── cli.py                    # CLI entry point (segment, ingest, search commands)
├── data/
│   └── segments/                 # Local segment cache (gitignored)
├── requirements.txt
└── .env                          # OPENROUTER_API_KEY, PINECONE_API_KEY, SUPABASE keys (gitignored)
```

---

## Implementation Phases

### Phase 1 — Foundation (no API keys needed)
1. Project scaffold: `src/` package, config, requirements
2. Provider Protocol interfaces (`base.py`)
3. Segmenter (`segmenter.py`): ffmpeg fixed-length 20s segments with 5s overlap
4. CLI skeleton (`cli.py`): `segment`, `ingest`, `search` commands

### Phase 2 — Ingestion Pipeline (needs OpenRouter key)
5. GeminiVideoAnalyzer (`video_analyzer.py`): scene + transcript + prosody in one call
6. OpenAI embedder (`embedding.py`)
7. Supabase client (`sql_store.py`): schema creation, insert, FTS queries
8. Pinecone client (`vector_store.py`): upsert, similarity search
9. Ingestion orchestrator (`ingestion.py`): segment → analyze → embed → store

### Phase 3 — Search (needs all keys)
10. Retriever (`retriever.py`): parallel FTS + vector search, RRF fusion
11. Reranker (`reranker.py`): Claude Sonnet re-scoring
12. CLI search command: query → results with evidence + video playback reference

### Phase 4 — Web UI (optional)
13. Simple web interface (Flask/Streamlit) wrapping the same `src/` modules

---

## Verification
- Run all 6 example queries from the challenge spec
- For transcript queries: verify matched segment contains expected words
- For visual queries: display keyframes from matched segments in notebook
- For action queries: verify scene description captures temporal events (not just static frames)
- For audio queries: confirm prosody data correctly flags shouting
- For text queries: verify scene description mentions visible text/plates
- For abstract queries: verify vector search finds semantically relevant results
- Latency: queries should return in <3 seconds
