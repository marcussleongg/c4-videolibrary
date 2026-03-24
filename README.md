# Video Library Search System

## How to Run

### Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys (see `.env.example`):
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your_supabase_anon_key_here
   ```

_OPTIONAL_
4. Update paths and model settings in `src/config.py` (only needed if ingesting your own videos or playing them back in the frontend):
   - `VIDEO_DIR` — path to directory containing source videos (needed for ingestion and frontend playback)
   - `SEGMENTS_DIR` — path to directory where segments will be stored
   - Model IDs for scene description, transcription, prosody, embedding, and reranking
   - Segmentation parameters (`SEGMENT_LENGTH`, `SEGMENT_OVERLAP`)
   - Retrieval parameters (`RRF_K`, `TOP_K_RETRIEVAL`, `TOP_K_RERANK`)
   - Pinecone index name and embedding dimensions

### Frontend (Gradio)

```bash
python app.py
```

Open `http://localhost:7860` in your browser. This launches a web UI with two tabs:
- **Search** — enter a natural language query, view ranked results, and click to play the matching video segment
- **Ingest** — upload a video file or paste a YouTube URL to segment and ingest it

### CLI

All CLI commands are run via:
```bash
python -m src.cli <command> [options]
```

**`search`** — Search for video segments matching a natural language query
```bash
python -m src.cli search "officer approaches vehicle at night"
python -m src.cli search "use of force" -k 5    # return top 5 results (default: 10)
```

**`segment`** — Split videos into 20s chunks with 5s overlap
```bash
python -m src.cli segment                  # segment all videos in VIDEO_DIR
python -m src.cli segment -f video.mp4     # segment a single video
python -m src.cli segment --force          # re-segment even if segments exist
```

**`ingest`** — Analyze segments (scene + transcript + prosody), embed, and store in Supabase + Pinecone
```bash
python -m src.cli ingest                   # ingest all segmented videos
python -m src.cli ingest -f video.mp4      # ingest a single video (segments first if needed)
python -m src.cli ingest --force           # re-ingest even if already in DB
python -m src.cli ingest -w 2              # use 2 parallel workers (default: 4)
```



## Approach

### Pipeline

```
                INGESTION (one-time)
                ====================

  Raw Videos ──> Segment (20s chunks, 5s overlap)
                        │
                        ▼
              ┌─────────────────────┐
              │   Gemini 3 Flash    │
              │   (medium res)      │
              │                     │
              │  Scene Description  │
              │  Transcript         │
              │  Prosody            │
              └─────────┬───────────┘
                        │
                 ┌──────┴──────┐
                 ▼             ▼
            Supabase       Pinecone
          (structured)     (vectors)
          + FTS index    gemini-embedding-001
                             3072d


                    QUERY (per search)
                    ==================

  User Query ──┬──────────────────────────────┐
               │                              ▼
               │                     Embed Query
               │                  (gemini-embedding-001)
               ▼                              │
            Supabase                     Pinecone
           FTS search                  vector search
                 │                           │
                 └──────┬────────────────────┘
                        ▼
                  RRF Fusion (k=60)
                        │
                        ▼
                ┌───────────────┐
                │  LLM Reranker │
                │ (Gemini 3     │
                │  Flash)       │
                └───────┬───────┘
                        │
                        ▼
                  Ranked Results
```

### Index Once, Query Many
The ingestion pipeline pays the cost of video analysis once per segment. After that, queries only require an embedding lookup, vector/text search, and a reranking call with no video reprocessing. This made most sense for this task as officers would likely have more than one query for a video, and relying on the context window after uploading a video once and asking multiple prompts is unreliable and expensive. The use case is especially prominent if we consider that officers might want to revisit cases and search over the same videos again after several days, where this approach provides persistence. It is also easy for the stored rows/vectors to be deleted from the databases if specific videos are no longer required.

### Hybrid Retrieval with RRF Fusion
Retrieval combines two complementary search strategies: full-text search (FTS) via Supabase for keyword/phrase matches, and vector search via Pinecone for semantic similarity. Results from both are merged using Reciprocal Rank Fusion (RRF), which balances the strengths of each without needing tuned weights. This ensures that both exact matches ("taser deployment") and abstract queries ("use of force") surface relevant segments. (https://www.mongodb.com/resources/basics/reciprocal-rank-fusion)

### Dual-Store Design
Supabase stores structured metadata (timestamps, scene descriptions, transcripts, prosody) and provides full-text search. Pinecone stores dense vector embeddings for semantic search. This is a standard RAG approach, but also because separating these allows each store to do what it's best at — Supabase for SQL queries, filtering, and FTS; Pinecone for fast approximate nearest-neighbor search at scale.

### Model Choice: Gemini 3 Flash (Medium Resolution)
After evaluating Gemini 2.5 Flash and Gemini 3 Flash at both medium and high resolution in the evaluation notebook, Gemini 3 Flash at medium resolution (70 tokens/frame) was selected. It produced the most structured and detailed output while keeping token costs manageable. Gemini 2.5 Flash scene descriptions also tended to be more story-narration esque (though prompt might could have been tuned). High resolution (280 tokens/frame) showed diminishing returns where output was frequently truncated due to the increased input token budget, with no meaningful quality improvement for this use case as the medium resolution seemed to capture sufficient details.

## Considerations in approach

### Frame Sampling VS native video processing
Ideally I would have been able to test/use models that have true native video processing (not frames), through 3D convolution, tokenizers that capture temporal information, etc. but architectures that truly capture space and time don't exist in production LLMs yet (Gemini takes in videos natively but still do frame sampling).

That leaves me with models that use frame sampling (model reasons to reconstruct temporal understanding). Many academic papers on video understanding benchmarks explicitly perform frame sampling, but I did not find work on native video processing (VLMs frame sample themselves). The papers were also older (before models that could natively take in videos were released) and hence did not benchmark models that are available today like Gemini 2.5/3. As such, these benchmarks served as a guideline to the best-performing models, but when considering the scope of this takehome, I decided to take the approach of using models with native video processing rather than frame sampling myself (although models like GPT-4o performed very well in benchmarks, but abstracting from performance of Gemini 1.5, Gemini 2.5/3 should perform as well if not better).

### Testing of different models
I initially wanted to allow for the testing of different models in the notebook, to see if breaking up the reasoning and understanding of video, transcript, and prosody performed better with different model and providers. But issues with OpenRouter's API key meant that I stuck to using Gemini. Similarly, I initially thought that I might want a specialized model to run OCR, but after consideration of the use case, it is likely that an officer would want to verify say for instance a license plate by looking at it, so there is little practical value in providing an OCR functionality that requires more computation of extracting one frame and sending it to a model.

## Architecture

### Chunking by fixed window length with overlap
This is a standard approach as seen in RAG. It is slightly different since text has natural breakpoints (sentences, paragraphs) but we are kind of flying blind for video. I decided on 20 seconds as segment length with 5 seconds overlap to fit length of actions that can occur during incidents (purely educated guess) like a foot chase, a takedown, pit manuever with crash, taser deployment and handcuff, etc. 5 second overlap is 25% of the segment length to keep as much as possible to ensure actions are not split between segments to lose semantic representation.

### Hotswappable Providers
The system is designed so that any provider (video analysis, embeddings, reranking) can be swapped without touching the pipeline logic. This is achieved through three layers:

1. **Protocol interfaces** (`providers/base.py`) — define the contracts (`SceneDescriber`, `TranscriptionProvider`, `ProsodyAnalyzer`, `EmbeddingProvider`) as Python Protocols. The rest of the codebase only depends on these method signatures.
2. **Provider implementations** — each concrete class (e.g. `GeminiVideoAnalyzer`, `GeminiEmbedder`) uses whatever SDK is native to that provider (instead of just using OpenAI format since the Gemini's native SDK allows for more control like resolution).
3. **Factory functions** (`providers/__init__.py`) — the single wiring layer that decides which implementation to return. Swapping a provider means changing one line here.

This means adding a new provider (e.g. an `OpenAIEmbedder`) is just writing a new class that satisfies the Protocol, then pointing the factory at it. No other file changes.

Per-task model configurability is also added. That is instead of only having a singular VLM take in a video, there is the ability to separate the models used for video itself, transcript, and prosody, which are set in `config.py`. But if all 3 point to the same model (like in our use case), the pipeline uses a singular API call to 1 VLM.

### Prosody is About Human Speech
The prosody section focuses exclusively on voice characteristics — volume, pitch, and speaking rate. Background audio (sirens, engines, wind) is handled by the scene description. Notable non-speech audio events (gunshots, explosions) are captured in the transcript with timestamps. This is enforced in the prompts.

### No Hard Metadata Filters at Retrieval
Retrieval does not apply hard constraints on metadata like `time_of_day` or `max_volume`. A user searching for "at night" might mean twilight, or the VLM might have labelled it differently. Instead, metadata is captured in the scene descriptions and embeddings, influencing ranking naturally. The reranker makes the final relevance judgement.

The metadata columns and indexes still exist in Supabase — they're useful for analytics, debugging (e.g. "how many segments labelled as shouting"), and the FTS function still accepts them as optional params if a strict filtering mode is ever needed.

### Resilient Ingestion
Ingestion skips already-processed segments by checking Supabase for existing IDs, so interrupted runs can be resumed. Transient failures (rate limits, timeouts, server errors) retry with exponential backoff. Use `--force` on a fresh run to bypass the existence checks.

### Timestamps are Absolute
The VLM returns timestamps relative to each segment clip. The ingestion pipeline converts these to absolute timestamps in the source video before storing. Everything in the database references the original video timeline.

### Loose Output Parsing
LLM output formatting is unreliable. Section headers from the VLM are parsed with flexible regex that handles `===`, `---`, `##`, `**`, mixed casing, and bare headings. Prosody JSON parsing handles markdown fences, trailing commas, and embedded objects. The system degrades gracefully rather than failing on formatting quirks.

### Prompt Design
- The prosody prompt explicitly tells the model to ignore background noise, since body-cam audio routinely contains wind, radio bleed, and traffic.
- Scene descriptions request "everything visible" rather than "notable objects" to maximise search recall.

### `gemini-embedding-001` at 3072 dimensions
At ~6K segments the cost difference between embedding models is negligible. Using 3072 dimensions (the max for this model) scores higher on retrieval benchmarks, which matters most for the abstract/semantic queries this system is designed for.
