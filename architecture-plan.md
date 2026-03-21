# Video Library Search System — Architecture Plan

## Context
Build a fully cloud-native system to make ~38 body-worn camera videos (30min–2hrs each) searchable via natural language. No local model inference — all processing, storage, and serving runs in the cloud. All model inference via OpenRouter. Deliverable: Python + Jupyter notebooks.

---

## Core Decisions

1. **Separate specialized cloud APIs + unified retrieval layer** — no unified model handles all modalities well enough
2. **Text-mediated visual search** — VLM generates rich scene descriptions, embed as text vectors. No CLIP needed.
3. **Hotswappable components** — each processing step is behind a Protocol interface, easily replaceable
4. **Mixed model strategy via OpenRouter** — use the best model for each task, not one model for everything
5. **OCR as a separate swappable function** — can plug in Google Cloud Vision, Roboflow, or YOLO later

---

## Model Selection (All via OpenRouter)

| Task | Model | Why this model? | Cost |
|------|-------|----------------|------|
| **Transcription** | Gemini Flash | 3.1% WER (better than Whisper 10.3%). One of few models accepting audio input | ~$6 for 50hrs |
| **Scene Description** | Gemini Flash | Bulk task (~6K segments). Cheap + good enough for descriptions | ~$5-10 |
| **OCR** | Claude Sonnet 4 | More precise on small text, license plates, partially obscured characters than Flash | ~$5-8 |
| **Audio Prosody** | Gemini Flash | Requires audio input. Gemini is one of the only models on OpenRouter with native audio support | ~$3-5 |
| **Text Embeddings** | OpenAI text-embedding-3-small | Standard, cheap, good quality. Via OpenRouter or direct | ~$0.50 |
| **Query Routing** | Claude Haiku / GPT-4o-mini | Fast + cheap. Routing is simple classification, doesn't need a large model | ~$0.001/query |
| **Reranking** | Claude Sonnet 4 | Needs strong reasoning to judge relevance of candidates | ~$0.02-0.05/query |

**Rationale for mixing**: Gemini Flash excels at bulk multimodal processing (cheap, accepts audio). Claude Sonnet excels at precision tasks (OCR accuracy, reasoning for reranking). Using the right model per task gives better quality per dollar than one model everywhere.

---

## Cloud Services

| Concern | Service | Why | Free Tier? |
|---------|---------|-----|------------|
| **Model Inference** | OpenRouter | Single API key, access to all models | Pay-per-use |
| **Video Storage** | Cloudflare R2 | No egress fees, S3-compatible | 10GB free |
| **Vector Database** | Pinecone serverless | Managed, metadata filtering | 2GB free |
| **SQL Database** | Supabase Postgres (FTS) | Structured data + full-text search | 500MB free |

---

## Architecture

### Hotswappable Provider Design

```python
class OCRProvider(Protocol):
    def extract_text(self, image_bytes: bytes) -> OCRResult: ...

class TranscriptionProvider(Protocol):
    def transcribe(self, audio_bytes: bytes) -> TranscriptResult: ...

class SceneDescriber(Protocol):
    def describe(self, frames: list[bytes]) -> SceneDescription: ...

class AudioAnalyzer(Protocol):
    def analyze(self, audio_bytes: bytes) -> ProsodyResult: ...

class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

# All providers instantiated via config — swap by changing config, not code
```

### Ingestion Pipeline

```
Video File → Upload to Cloudflare R2
  │
  ├─→ [ffmpeg locally] — not model inference, just file splitting
  │     Adaptive scene-change segmentation (PySceneDetect)
  │     Min 10s, max 60s per segment
  │     Extract audio (.wav) + keyframes (1 per 5s) per segment
  │     Upload extracted files to R2
  │
  ├─→ Per segment (all via OpenRouter, parallelized):
  │     │
  │     ├─→ [TranscriptionProvider → Gemini Flash]
  │     │     Audio → timestamped transcript (3.1% WER)
  │     │
  │     ├─→ [SceneDescriber → Gemini Flash]
  │     │     Keyframes → detailed scene description
  │     │     Prompt: people, objects, actions, clothing,
  │     │     vehicles, environment, lighting, time of day
  │     │
  │     ├─→ [OCRProvider → Claude Sonnet]
  │     │     Keyframes → extract all visible text, license plates, signs
  │     │     Separate call with focused OCR prompt for precision
  │     │     ★ Hotswappable: Google Cloud Vision, Roboflow, etc.
  │     │
  │     ├─→ [AudioAnalyzer → Gemini Flash]
  │     │     Audio → prosody analysis
  │     │     Returns: { has_shouting, max_intensity, tone, description }
  │     │
  │     ├─→ [EmbeddingProvider → OpenAI]
  │     │     Concatenate all text → single embedding vector
  │     │
  │     └─→ [Store]
  │           → Pinecone: embedding + metadata
  │           → Supabase: full segment record + FTS indexes
```

### Indexing Layer

**Pinecone** (vector search):
- One embedding per segment (all modalities merged into text → single vector)
- Metadata: video_id, start_s, end_s, has_shouting, time_of_day
- Supports metadata filtering + vector similarity in one query

**Supabase Postgres** (structured + full-text search):
- `segments` table: all fields, full segment records
- `tsvector` index on: transcript, scene_description, ocr_text, prosody_notes
- Exact keyword search ("Miranda rights") + structured filters

### Retrieval + Fusion

```
Natural Language Query
  │
  ├─→ [Query Analyzer — Claude Haiku via OpenRouter]
  │     Classify query → determine retrieval strategy
  │     Output: { vector_query, text_query, metadata_filters,
  │               modalities: ["visual","audio","text","metadata"] }
  │
  ├─→ [Parallel Retrieval]
  │     ├─→ Pinecone: embed query → semantic similarity
  │     ├─→ Supabase FTS: keyword/phrase match
  │     └─→ Supabase SQL: metadata filters (has_shouting, time_of_day)
  │
  └─→ [Fusion + Reranking]
        ├─→ Reciprocal Rank Fusion: score = Σ(1/(k+rank_i))
        └─→ Top-K → Claude Sonnet reranker (strong reasoning for relevance)
        └─→ Return: [(video_id, start_s, end_s, score, evidence)]
```

---

## Query → Retrieval Path Mapping

| Query | Strategy | How |
|-------|----------|-----|
| "Vehicle pulled over at night" | Vector + metadata | Semantic search on scene desc + time_of_day filter |
| "Someone raises their voice" | Metadata + vector | Filter has_shouting=true + semantic on prosody notes |
| "Person in a red shirt" | Vector | Semantic search — scene descriptions mention clothing |
| "Officer reads Miranda rights" | FTS | Postgres full-text "Miranda" on transcript |
| "License plates visible" | FTS + vector | FTS on ocr_text + semantic on scene_desc |
| "Suspect being handcuffed" | Vector | Semantic search — scene desc covers actions |

---

## Cost Estimate (all 38 videos, ~50hrs footage)

| Component | Cost |
|-----------|------|
| Gemini Flash — transcription (~50hrs audio) | ~$6 |
| Gemini Flash — scene descriptions (~6K segments) | ~$5-10 |
| Claude Sonnet — OCR pass (~6K segments) | ~$5-8 |
| Gemini Flash — audio prosody (~6K segments) | ~$3-5 |
| OpenAI — text embeddings (~6K segments) | ~$0.50 |
| Pinecone serverless (free tier) | $0 |
| Supabase Postgres (free tier) | $0 |
| Cloudflare R2 (10GB free) | $0 |
| **Total indexing** | **~$22-32** |
| Per-query cost (routing + reranking) | ~$0.02-0.05 |

---

## Project Structure

```
video-search/
├── notebooks/
│   ├── 01_segment.ipynb       # ffmpeg + PySceneDetect: split, extract, upload
│   ├── 02_ingest.ipynb        # OpenRouter APIs: transcribe, describe, OCR, analyze
│   ├── 03_index.ipynb         # Push to Pinecone + Supabase
│   └── 04_search.ipynb        # Interactive query interface + demo
├── src/
│   ├── providers/             # Hotswappable provider interfaces
│   │   ├── base.py            # Protocol classes: OCRProvider, TranscriptionProvider, etc.
│   │   ├── transcription.py   # GeminiTranscriber (default). Swappable: Whisper, Deepgram
│   │   ├── scene.py           # GeminiSceneDescriber (default)
│   │   ├── ocr.py             # ClaudeOCR (default). Swappable: Google Vision, Roboflow
│   │   ├── audio.py           # GeminiAudioAnalyzer (default). Swappable: Hume AI
│   │   └── embedding.py       # OpenAIEmbedder (default). Swappable: Voyage, Cohere
│   ├── segmenter.py           # PySceneDetect + ffmpeg
│   ├── storage.py             # R2 upload/download client
│   ├── vector_store.py        # Pinecone client
│   ├── sql_store.py           # Supabase Postgres client
│   ├── query_analyzer.py      # Claude Haiku: classify query → retrieval plan
│   ├── retriever.py           # Multi-index retrieval + RRF
│   ├── reranker.py            # Claude Sonnet: rerank top-K
│   └── config.py              # Model IDs, API keys (env vars), thresholds
├── data/
│   └── segments/              # Local cache (gitignored)
├── requirements.txt
└── .env                       # OPENROUTER_API_KEY, PINECONE_API_KEY, etc. (gitignored)
```

---

## Implementation Order

1. **Setup**: Project scaffold, env vars, provider Protocol interfaces
2. **Segmentation**: ffmpeg + PySceneDetect → split videos, extract audio + keyframes
3. **Upload**: Push segments to Cloudflare R2
4. **Transcription**: Gemini Flash via OpenRouter → timestamped transcripts
5. **Scene descriptions**: Gemini Flash via OpenRouter → rich text descriptions
6. **OCR**: Claude Sonnet via OpenRouter → text/plate extraction (hotswappable)
7. **Audio prosody**: Gemini Flash via OpenRouter → shouting detection, tone
8. **Embeddings**: OpenAI → all text → vectors
9. **Indexing**: Push to Pinecone (vectors) + Supabase (FTS + metadata)
10. **Query pipeline**: Haiku analyzer → parallel retrieval → RRF → Sonnet reranker
11. **Notebook demo**: All 6 example queries with evidence display

---

## Verification
- Run all 6 example queries from the challenge spec
- For transcript queries: verify matched segment contains expected words
- For visual queries: display keyframes from matched segments
- For audio queries: confirm prosody notes mention raised voices
- For OCR queries: verify extracted plate text matches video
- Latency: queries should return in <3 seconds
