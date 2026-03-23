# Video Library Search System

TODOS:
- run the tests in notebook (probably still just use gemini for all)
- actually ingest the segments
- pray it works
- consider building ui, that allows for ingestion of new videos

## Core Considerations

### Frame Sampling VS native video processing
Ideally I would have been able to test/use models that have true native video processing (not frames), through 3D convolution, tokenizers that capture temporal information, etc. but they don't exist in production LLMs yet.

That leaves me with models that use frame sampling (model reasons to reconstruct temporal understanding). Many academic papers on video understanding benchmarks explicitly perform frame sampling, but I did not find work on native video processing (VLMs frame sample themselves). The papers were also older (before models that could natively take in videos were released) and hence did not benchmark models that are available today like Gemini 2.5/3. As such, these benchmarks served as a guideline to the best-performing models, but when considering the scope of this takehome, I decided to take the approach of using models with native video processing rather than frame sampling myself (although models like GPT-4o performed very well in benchmarks, but abstracting from performance of Gemini 1.5, Gemini 2.5/3 should perform as well if not better).

### Chunking by fixed window length with overlap
This is a standard approach as seen in RAG. It is slightly different since text has natural breakpoints (sentences, paragraphs) but we are kind of flying blind for video. I decided on 20 seconds as segment length with 5 seconds overlap to fit length of actions that can occur during incidents (purely educated guess) like a foot chase, a takedown, pit manuever with crash, taser deployment and handcuff, etc. 5 second overlap is 25% of the segment length to keep as much as possible to ensure actions are not split between segments to lose semantic representation.

### Hotswappable Providers
Each analysis task (scene description, transcription, prosody) has its own Protocol interface. Swapping a provider means changing the class in the factory (`providers/__init__.py`), not rewiring the pipeline. The default fast path combines all three into a single Gemini call; swapping one provider causes only that task to use a separate call.

### Prosody is About Human Speech
The prosody section focuses exclusively on voice characteristics — volume level and emotional tone. Background audio (sirens, engines, wind) is handled by the scene description. Notable non-speech audio events (gunshots, explosions) are captured in the transcript with timestamps. Each section has a single responsibility.

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

### `text-embedding-3-large` over `small`
At ~6K segments the cost difference is under $2. The larger model scores higher on retrieval benchmarks, which matters most for the abstract/semantic queries this system is designed for.
