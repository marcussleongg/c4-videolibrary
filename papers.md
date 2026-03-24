**Prosody:**
Beyond Silent Letters: Amplifying LLMs in Emotion Recognition with Vocal Nuances
https://arxiv.org/pdf/2407.21315
A two-pass where we first analyze the raw audio components then get LLM to reason/determnistically decide on each then passing that as a natural language description of the audio to a VLM to then assess tone. Possible future implementation, but in this takehome I used the 5 audio components in the prompt.

**VLMs:**
VHELM: A Holistic Evaluation of Vision Language Models
https://arxiv.org/pdf/2410.07112
GPT-4o is best across all the metrics, but when we isolate to visual perception, knowledge (VQA) and reasoning (VQA) metrics, models like Gemini 1.5 Flash are somewhat comparable and do better in some metrics. Similarly Flash models are very comparable/do better than Pro models in some of these metrics. If we abstract to newer Gemini models, it can mean Flash is a worthy model to use when considering cost-performance trade-off. Note that this is an image benchmark.

MLVU: Benchmarking Multi-task Long Video Understanding
https://arxiv.org/pdf/2406.04264
GPT-4o performed best out of all models tested (also not alot of newer models/ones we have access to. MLLMs perform much worse with longer videos, justification for chunking down to shorter clips.)

LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding
https://arxiv.org/pdf/2407.15754
Similarly GPT-4o performs best, Gemini 1.5 Flash very comparable to Gemini 1.5 Pro. "Video" understanding via frame sampling.

Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis
https://arxiv.org/pdf/2405.21075
Similar to VHELM, that Gemini 1.5 Flash is comparable to Pro and GPT-4o. Also frame extraction.

Multi-hop, not used in this project but could have been added as an improvement
https://arxiv.org/html/2603.14468v1

**Speech-to-text:**
https://artificialanalysis.ai/speech-to-text
Used to pick models for testing for transcription

**Modern approaches:**
V-JEPA 2 (3d convolution)
https://arxiv.org/pdf/2506.09985

Cosmos (specifically the tokenizer)
https://arxiv.org/html/2501.03575v1#S4

VTok (feature subtraction)
https://arxiv.org/pdf/2602.04202