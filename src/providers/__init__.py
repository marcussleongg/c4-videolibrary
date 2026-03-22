"""Provider factory — reads config to decide which provider handles each task.

Hotswapping: change SCENE_MODEL / TRANSCRIPTION_MODEL / PROSODY_MODEL in config
to use different models. If all three point to the same model (default),
the pipeline can use GeminiVideoAnalyzer.analyze() for a single API call.
If any differ, the pipeline calls each provider independently.
"""
from __future__ import annotations

from src.config import SCENE_MODEL, TRANSCRIPTION_MODEL, PROSODY_MODEL
from src.providers.base import (
    AnalysisResult,
    EmbeddingProvider,
    ProsodyAnalyzer,
    ProsodyResult,
    SceneDescription,
    SceneDescriber,
    TranscriptResult,
    TranscriptionProvider,
)
from src.providers.embedding import OpenAIEmbedder
from src.providers.video_analyzer import GeminiVideoAnalyzer


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured EmbeddingProvider."""
    return OpenAIEmbedder()


def get_scene_describer() -> SceneDescriber:
    """Return the configured SceneDescriber."""
    return GeminiVideoAnalyzer(model=SCENE_MODEL)


def get_transcription_provider() -> TranscriptionProvider:
    """Return the configured TranscriptionProvider."""
    return GeminiVideoAnalyzer(model=TRANSCRIPTION_MODEL)


def get_prosody_analyzer() -> ProsodyAnalyzer:
    """Return the configured ProsodyAnalyzer."""
    return GeminiVideoAnalyzer(model=PROSODY_MODEL)


def can_use_fast_path() -> bool:
    """True if all three tasks use the same model — enables single API call."""
    return SCENE_MODEL == TRANSCRIPTION_MODEL == PROSODY_MODEL


def analyze_segment(video_bytes: bytes) -> AnalysisResult:
    """Analyze a video segment, using the fast path when possible.

    Fast path (all same model): 1 API call → scene + transcript + prosody.
    Slow path (mixed models): up to 3 separate API calls.
    """
    if can_use_fast_path():
        analyzer = GeminiVideoAnalyzer(model=SCENE_MODEL)
        scene, transcript, prosody = analyzer.analyze(video_bytes)
    else:
        scene = get_scene_describer().describe(video_bytes)
        transcript = get_transcription_provider().transcribe(video_bytes)
        prosody = get_prosody_analyzer().analyze_prosody(video_bytes)

    return AnalysisResult(scene=scene, transcript=transcript, prosody=prosody)
