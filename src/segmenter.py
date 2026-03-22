import subprocess
import json
from pathlib import Path

from src.config import VIDEO_DIR, SEGMENTS_DIR, SEGMENT_LENGTH, SEGMENT_OVERLAP


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path,
        ],
        capture_output=True, text=True,
    )
    try:
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except (json.JSONDecodeError, KeyError):
        raise ValueError(f"Could not read video: {video_path}\nffprobe stderr: {result.stderr}")


def compute_segments(duration: float) -> list[tuple[float, float]]:
    """Compute fixed-length segment boundaries with overlap.

    Each segment is SEGMENT_LENGTH seconds long, advancing by
    (SEGMENT_LENGTH - SEGMENT_OVERLAP) seconds each step.
    The last segment may be shorter.
    """
    step = SEGMENT_LENGTH - SEGMENT_OVERLAP
    segments = []
    cursor = 0.0

    while cursor < duration:
        end = min(cursor + SEGMENT_LENGTH, duration)
        segments.append((cursor, end))
        cursor += step

    return segments


def split_video(
    video_path: str,
    segments: list[tuple[float, float]],
    output_dir: str,
) -> list[dict]:
    """Split video into segments using ffmpeg stream copy.

    Returns list of segment metadata dicts:
        {file_name, segment_path, segment_index, start_s, end_s}
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    results = []

    for i, (start, end) in enumerate(segments):
        segment_name = f"{stem}_seg_{i:03d}.mp4"
        segment_path = output_dir / segment_name

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(video_path),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(segment_path),
            ],
            capture_output=True, text=True,
        )

        results.append({
            "file_name": video_path.name,
            "segment_path": str(segment_path),
            "segment_index": i,
            "start_s": start,
            "end_s": end,
        })

    return results


def is_segmented(video_path: str) -> bool:
    """Check if a video has already been segmented."""
    stem = Path(video_path).stem
    output_dir = Path(SEGMENTS_DIR) / stem
    return output_dir.exists() and any(output_dir.glob("*.mp4"))


def segment_video(video_path: str, force: bool = False) -> list[dict]:
    """Segment a single video into fixed-length overlapping chunks.

    Segments are saved to SEGMENTS_DIR/{video_stem}/.
    Skips if already segmented unless force=True.
    Returns list of segment metadata dicts.
    """
    video_path = str(Path(video_path).resolve())
    stem = Path(video_path).stem
    output_dir = str(Path(SEGMENTS_DIR) / stem)

    if not force and is_segmented(video_path):
        print(f"  Skipping (already segmented). Use --force to re-segment.")
        return []

    duration = get_video_duration(video_path)
    segments = compute_segments(duration)
    return split_video(video_path, segments, output_dir)


def segment_all_videos(force: bool = False) -> list[dict]:
    """Segment all videos in VIDEO_DIR.

    Skips already-segmented videos unless force=True.
    Returns list of all segment metadata dicts across all videos.
    """
    video_dir = Path(VIDEO_DIR)
    all_segments = []

    video_files = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
        and not p.name.startswith("._")
    )

    for video_path in video_files:
        print(f"Segmenting: {video_path.name}")
        segments = segment_video(str(video_path), force=force)
        if segments:
            print(f"  → {len(segments)} segments")
        all_segments.extend(segments)

    print(f"\nTotal: {len(all_segments)} segments from {len(video_files)} videos")
    return all_segments
