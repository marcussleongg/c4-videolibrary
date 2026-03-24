from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, SCENE_MODEL
from src.providers.base import (
    ProsodyResult,
    SceneDescription,
    TranscriptResult,
)
from src.providers.prompts import (
    COMBINED_PROMPT,
    PROSODY_ONLY_PROMPT,
    SCENE_ONLY_PROMPT,
    TRANSCRIPT_ONLY_PROMPT,
)


# --- Video helpers ---

def _compress_to_720p(video_bytes: bytes, suffix: str = ".mp4") -> bytes:
    """Compress video to 720p if needed. Works with raw bytes."""
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0", tmp_in_path],
            capture_output=True, text=True,
        )
        height = int(probe.stdout.strip())

        if height <= 720:
            return video_bytes

        with NamedTemporaryFile(suffix=suffix, delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        subprocess.run(
            ["ffmpeg", "-i", tmp_in_path, "-vf", "scale=-2:720",
             "-c:a", "copy", tmp_out_path, "-y", "-loglevel", "error"],
            check=True,
        )
        return Path(tmp_out_path).read_bytes()
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)


def _build_video_content(video_bytes: bytes, prompt: str) -> types.Content:
    """Build Gemini content with inline video bytes at medium resolution (70 tokens/frame)."""
    return types.Content(
        parts=[
            types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"),
                media_resolution=types.PartMediaResolution(
                    level=types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
                ),
            ),
            types.Part(text=prompt),
        ]
    )


# --- Response parsing ---

def _split_scene_time_of_day(scene_text: str) -> tuple[str, str]:
    """Remove TIME_OF_DAY: line from scene text. Returns (cleaned text, time_of_day).

    Falls back to keyword scan if the line is missing.
    """
    out_lines: list[str] = []
    found: str | None = None
    for line in scene_text.splitlines():
        m = re.match(r"^\s*TIME_OF_DAY:\s*(\S+)\s*$", line, re.I)
        if m:
            found = m.group(1).lower()
            continue
        out_lines.append(line)

    cleaned = "\n".join(out_lines).strip()

    if found is None:
        lower = cleaned.lower()
        if "night" in lower:
            found = "night"
        elif "twilight" in lower or "dusk" in lower or "dawn" in lower:
            found = "twilight"
        else:
            found = "day"

    return cleaned, found

# Using regex rather than JSON to avoid issues with LLM JSON formatting.
_SECTION_RE = re.compile(
    r"(?:^|\n)"                       # start of text or newline
    r"[\s*#=\-]*"                     # optional decoration (===, ---, ##, **, whitespace)
    r"(SCENE\s*DESCRIPTION|TRANSCRIPT|PROSODY)"  # section name
    r"[\s*#=\-]*"                     # trailing decoration
    r"(?:\n|$)",                      # end of line
    re.IGNORECASE,
)


def _split_sections(text: str) -> dict[str, str]:
    """Split model output into named sections using flexible header matching."""
    matches = list(_SECTION_RE.finditer(text))
    sections: dict[str, str] = {}
    for idx, m in enumerate(matches):
        name = re.sub(r"\s+", " ", m.group(1)).strip().upper()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


def _parse_combined_response(text: str) -> tuple[SceneDescription, TranscriptResult, ProsodyResult]:
    """Parse the three-section response from a combined Gemini call."""
    sections = _split_sections(text)
    scene_text_raw = sections.get("SCENE DESCRIPTION", "")
    transcript_text = sections.get("TRANSCRIPT", "")
    prosody_text = sections.get("PROSODY", "")

    prosody = _parse_prosody_json(prosody_text)

    scene_body, time_of_day = _split_scene_time_of_day(scene_text_raw)

    scene = SceneDescription(
        scene_description=scene_body,
        time_of_day=time_of_day,
        events=_extract_events(scene_body),
    )
    transcript = TranscriptResult(transcript=transcript_text)

    return scene, transcript, prosody


def _extract_json_object(text: str) -> dict:
    """Parse first balanced {...} from text; handles nested objects."""
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _parse_prosody_json(text: str) -> ProsodyResult:
    """Parse prosody JSON from model output, handling markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = _extract_json_object(text)

    return ProsodyResult(
        max_volume=data.get("max_volume", "normal"),
        emotional_tones=data.get("emotional_tones", []),
    )


def _extract_events(scene_text: str) -> list[str]:
    """Extract event lines from scene text (EVENT: prefix or [MM:SS] timeline bullets)."""
    events: list[str] = []
    for line in scene_text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("EVENT:"):
            events.append(stripped[6:].strip())
        elif re.match(r"^-\s*\[\d{1,2}:\d{2}\]\s+", stripped):
            events.append(re.sub(r"^-\s*", "", stripped).strip())
    return events


def _parse_scene_only(text: str) -> SceneDescription:
    """Parse scene-only response."""
    lines = text.strip().splitlines()
    scene_lines: list[str] = []
    events: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("EVENT:"):
            events.append(stripped[6:].strip())
        else:
            scene_lines.append(line)

    body, time_of_day = _split_scene_time_of_day("\n".join(scene_lines).strip())

    return SceneDescription(
        scene_description=body,
        time_of_day=time_of_day,
        events=events,
    )


# --- Main class ---

class GeminiVideoAnalyzer:
    """Combined SceneDescriber + TranscriptionProvider + ProsodyAnalyzer.

    Default fast path: one Gemini Flash call returns all three outputs.
    Each method can also be called independently when only that task is needed
    (e.g., when the other tasks are handled by swapped-in providers).
    """

    def __init__(self, model: str | None = None, compress: bool = False):
        self._model = model or SCENE_MODEL
        self._compress = compress
        # v1alpha required for media_resolution parameter
        self._client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha"},
        )

    def _call(self, content: types.Content, max_tokens: int = 8192) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=content,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        # warning in the event that the response is truncated
        if response.candidates and response.candidates[0].finish_reason.name == "MAX_TOKENS":
            print(f"  [WARN] Response truncated at {max_tokens} tokens — output may be incomplete")
        return response.text

    def _prepare(self, video_bytes: bytes) -> bytes:
        """Optionally compress video before sending to the API."""
        return _compress_to_720p(video_bytes) if self._compress else video_bytes

    def analyze(self, video_bytes: bytes) -> tuple[SceneDescription, TranscriptResult, ProsodyResult]:
        """Combined call — scene + transcript + prosody in one API call."""
        content = _build_video_content(self._prepare(video_bytes), COMBINED_PROMPT)
        raw = self._call(content)
        return _parse_combined_response(raw)

    # --- Individual Protocol methods (used if other providers are swapped in) ---

    def describe(self, video_bytes: bytes) -> SceneDescription:
        content = _build_video_content(self._prepare(video_bytes), SCENE_ONLY_PROMPT)
        raw = self._call(content)
        return _parse_scene_only(raw)

    def transcribe(self, video_bytes: bytes) -> TranscriptResult:
        content = _build_video_content(self._prepare(video_bytes), TRANSCRIPT_ONLY_PROMPT)
        raw = self._call(content)
        return TranscriptResult(transcript=raw.strip())

    def analyze_prosody(self, video_bytes: bytes) -> ProsodyResult:
        content = _build_video_content(self._prepare(video_bytes), PROSODY_ONLY_PROMPT)
        raw = self._call(content)
        return _parse_prosody_json(raw)
