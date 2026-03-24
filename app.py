"""Gradio UI — ingest, search, and playback for the video library."""
from __future__ import annotations

import re
import subprocess
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr

from src.config import VIDEO_DIR, SEGMENTS_DIR, SEGMENT_LENGTH, SEGMENT_OVERLAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _resolve_video_path(file_name: str) -> str | None:
    """Try to find the source video locally given a file_name from the DB."""
    p = Path(VIDEO_DIR) / file_name
    if p.exists():
        return str(p)
    # Try without extension variations
    for ext in (".mp4", ".mov", ".avi", ".mkv"):
        candidate = Path(VIDEO_DIR) / (Path(file_name).stem + ext)
        if candidate.exists():
            return str(candidate)
    return None


def _download_youtube(url: str) -> Path:
    """Download a YouTube video to VIDEO_DIR using yt-dlp. Returns the path."""
    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    # Get the title first for a clean filename
    result = subprocess.run(
        ["yt-dlp", "--print", "filename", "-o", "%(title)s.%(ext)s",
         "--merge-output-format", "mp4", url],
        capture_output=True, text=True,
    )
    filename = Path(result.stdout.strip()).name if result.returncode == 0 else None

    output_template = str(Path(VIDEO_DIR) / "%(title)s.%(ext)s")
    proc = subprocess.run(
        ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
         "--merge-output-format", "mp4",
         "-o", output_template, url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{proc.stderr}")

    # Find the downloaded file
    if filename:
        dest = Path(VIDEO_DIR) / filename
        if dest.exists():
            return dest

    # Fallback: most recently modified mp4 in VIDEO_DIR
    mp4s = sorted(Path(VIDEO_DIR).glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if mp4s:
        return mp4s[-1]
    raise RuntimeError("Download succeeded but could not locate output file.")


# ---------------------------------------------------------------------------
# Ingest logic
# ---------------------------------------------------------------------------

def ingest(video_file: str | None, youtube_url: str) -> str:
    """Ingest a local upload or YouTube URL through the full pipeline."""
    from src.segmenter import segment_video
    from src.ingestion import ingest_video

    # Determine video path
    if youtube_url and youtube_url.strip():
        yield "Downloading from YouTube…"
        try:
            video_path = _download_youtube(youtube_url.strip())
        except RuntimeError as e:
            yield f"Error: {e}"
            return
        yield f"Downloaded → {video_path.name}\n"
    elif video_file:
        # Gradio upload — copy to VIDEO_DIR
        src = Path(video_file)
        dest = Path(VIDEO_DIR) / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        video_path = dest
    else:
        yield "Provide a video file or YouTube URL."
        return

    # Segment
    yield f"Segmenting {video_path.name}…\n"
    segment_metas = segment_video(str(video_path), force=False)
    if not segment_metas:
        # Already segmented — reconstruct metas
        seg_dir = Path(SEGMENTS_DIR) / video_path.stem
        step = SEGMENT_LENGTH - SEGMENT_OVERLAP
        segment_metas = []
        for seg_file in sorted(seg_dir.glob("*_seg_*.mp4")):
            match = re.search(r"_seg_(\d+)", seg_file.stem)
            if not match:
                continue
            idx = int(match.group(1))
            segment_metas.append({
                "file_name": video_path.name,
                "segment_path": str(seg_file),
                "segment_index": idx,
                "start_s": idx * step,
                "end_s": idx * step + SEGMENT_LENGTH,
            })
    yield f"  {len(segment_metas)} segments\n"

    # Ingest
    yield "Ingesting (analyze → embed → store)…\n"
    ingested = ingest_video(segment_metas, max_workers=4, force=False)
    skipped = len(segment_metas) - len(ingested)

    summary = f"Done. {len(ingested)} ingested"
    if skipped:
        summary += f", {skipped} skipped (already in DB)"
    yield summary


# ---------------------------------------------------------------------------
# Search logic
# ---------------------------------------------------------------------------

def search(query: str):
    """Run retrieval + reranking and return results for the table."""
    if not query.strip():
        return []

    from src.retriever import retrieve
    from src.reranker import rerank

    candidates = retrieve(query)
    if not candidates:
        return []

    results = rerank(query, candidates)

    table_rows = []
    for i, r in enumerate(results):
        score = r.get("rerank_score", 0)
        start = r.get("start_s", 0)
        end = r.get("end_s", 0)
        table_rows.append([
            i + 1,
            f"{score}/10",
            r.get("file_name", ""),
            f"{_fmt_timestamp(start)} – {_fmt_timestamp(end)}",
            r.get("rerank_reason", ""),
            start,  # hidden: raw start_s for playback
        ])

    return table_rows


# ---------------------------------------------------------------------------
# Playback logic
# ---------------------------------------------------------------------------

def play_segment(table_data, evt: gr.SelectData):
    """When a result row is clicked, load the video at the right timestamp."""
    import pandas as pd
    if table_data is None or (isinstance(table_data, pd.DataFrame) and table_data.empty):
        return gr.update(), ""
    if evt.index[0] >= len(table_data):
        return gr.update(), ""

    row = table_data.iloc[evt.index[0]]
    file_name = row.iloc[2]
    start_s = float(row.iloc[5])

    video_path = _resolve_video_path(file_name)
    if not video_path:
        return None, f"Video not found locally: {file_name}"

    return video_path, f"Starts at {_fmt_timestamp(start_s)} — seek to this timestamp in the player."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Video Library") as app:
    gr.Markdown("# Video Library Search")

    with gr.Tab("Ingest"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload video", sources=["upload"])
            with gr.Column():
                url_input = gr.Textbox(
                    label="Or paste a YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    info="Video will be downloaded locally before ingestion.",
                )
        ingest_btn = gr.Button("Ingest", variant="primary")
        ingest_log = gr.Textbox(label="Progress", lines=8, interactive=False)

        ingest_btn.click(
            fn=ingest,
            inputs=[video_input, url_input],
            outputs=ingest_log,
        )

    with gr.Tab("Search"):
        with gr.Row():
            query_input = gr.Textbox(
                label="Search",
                placeholder="e.g. officer approaches vehicle at night",
                scale=4,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        results_table = gr.Dataframe(
            headers=["#", "Score", "File", "Timestamp", "Reason", "_start"],
            column_widths=["5%", "8%", "20%", "15%", "52%", "0%"],
            interactive=False,
            label="Results",
        )
        player_video = gr.Video(label="Playback", interactive=False)
        player_info = gr.Markdown()

        search_btn.click(
            fn=search,
            inputs=[query_input],
            outputs=[results_table],
        )
        # Submit on Enter
        query_input.submit(
            fn=search,
            inputs=[query_input],
            outputs=[results_table],
        )

        results_table.select(
            fn=play_segment,
            inputs=[results_table],
            outputs=[player_video, player_info],
        )


if __name__ == "__main__":
    # Allow serving files from video directory
    app.launch(
        allowed_paths=[VIDEO_DIR],
        theme=gr.themes.Soft(),
    )
