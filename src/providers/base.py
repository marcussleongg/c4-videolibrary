from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SceneDescription:
    scene_description: str
    time_of_day: str  # "day" / "night" / "dusk" / "dawn"
    events: list[str] = field(default_factory=list)


@dataclass
class TranscriptResult:
    transcript: str  # timestamped speech, e.g. "[00:05] Speaker 1: ..."


@dataclass
class ProsodyResult:
    has_shouting: bool
    max_volume: str  # "quiet" / "normal" / "loud" / "shouting"
    emotional_tones: list[str] = field(default_factory=list)
    background_sounds: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Combined output from all three analysis tasks."""
    scene: SceneDescription
    transcript: TranscriptResult
    prosody: ProsodyResult


@runtime_checkable
class SceneDescriber(Protocol):
    def describe(self, video_bytes: bytes) -> SceneDescription: ...


@runtime_checkable
class TranscriptionProvider(Protocol):
    def transcribe(self, video_bytes: bytes) -> TranscriptResult: ...


@runtime_checkable
class ProsodyAnalyzer(Protocol):
    def analyze_prosody(self, video_bytes: bytes) -> ProsodyResult: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
