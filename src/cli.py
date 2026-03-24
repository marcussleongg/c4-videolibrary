import argparse
import sys
from pathlib import Path

from src.config import VIDEO_DIR, SEGMENTS_DIR


def cmd_segment(args):
    """Segment videos into fixed-length overlapping chunks."""
    from src.segmenter import segment_video, segment_all_videos

    if args.file:
        video_path = Path(args.file)
        if not video_path.exists():
            # Try resolving relative to VIDEO_DIR
            video_path = Path(VIDEO_DIR) / args.file
        if not video_path.exists():
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        print(f"Segmenting: {video_path.name}")
        segments = segment_video(str(video_path), force=args.force)
        if segments:
            print(f"  → {len(segments)} segments saved to {SEGMENTS_DIR}/{video_path.stem}/")
    else:
        segment_all_videos(force=args.force)


def cmd_ingest(args):
    """Analyze segments and store results in Supabase + Pinecone."""
    from src.ingestion import ingest_video, ingest_all_videos
    from src.segmenter import segment_video

    workers = args.workers

    if args.file:
        video_path = Path(args.file)
        if not video_path.exists():
            video_path = Path(VIDEO_DIR) / args.file
        if not video_path.exists():
            print(f"Error: file not found: {args.file}")
            sys.exit(1)

        # Segment first if needed, then ingest
        from src.segmenter import segment_video, is_segmented
        if not is_segmented(str(video_path)):
            print(f"Segmenting first: {video_path.name}")
            segment_metas = segment_video(str(video_path))
        else:
            # Reconstruct metas from existing segment files
            seg_dir = Path(SEGMENTS_DIR) / video_path.stem
            import re
            segment_metas = []
            for seg_file in sorted(
                f for f in seg_dir.glob("*_seg_*.mp4")
                if not f.name.startswith("._")  # skip macOS resource forks
            ):
                match = re.search(r"_seg_(\d+)", seg_file.stem)
                if not match:
                    continue
                idx = int(match.group(1))
                from src.config import SEGMENT_LENGTH, SEGMENT_OVERLAP
                step = SEGMENT_LENGTH - SEGMENT_OVERLAP
                segment_metas.append({
                    "file_name": video_path.name,
                    "segment_path": str(seg_file),
                    "segment_index": idx,
                    "start_s": idx * step,
                    "end_s": idx * step + SEGMENT_LENGTH,
                })

        print(f"Ingesting: {video_path.name} ({len(segment_metas)} segments)")
        ingest_video(segment_metas, max_workers=workers, force=args.force)
    else:
        ingest_all_videos(max_workers=workers, force=args.force)


def cmd_search(args):
    """Search for video segments matching a natural language query."""
    from src.retriever import retrieve
    from src.reranker import rerank

    print(f"Searching: {args.query}\n")

    # 1. Retrieve (FTS + vector, fused with RRF)
    candidates = retrieve(args.query)
    if not candidates:
        print("No results found.")
        return

    # 2. Rerank top candidates with Claude
    results = rerank(args.query, candidates, top_k=args.top_k)

    # 3. Display results
    for i, r in enumerate(results, start=1):
        score = r.get("rerank_score", "—")
        reason = r.get("rerank_reason", "")
        start = r.get("start_s", 0)
        end = r.get("end_s", 0)
        start_min, start_sec = int(start // 60), int(start % 60)
        end_min, end_sec = int(end // 60), int(end % 60)

        print(f"  {i}. [{score}/10] {r.get('file_name', '')} "
              f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]")
        if reason:
            print(f"     {reason}")
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="videolibrary",
        description="Video Library Search System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- segment ---
    seg_parser = subparsers.add_parser("segment", help="Segment videos into chunks")
    seg_parser.add_argument(
        "--file", "-f",
        help="Segment a single video file (path or filename in VIDEO_DIR). "
             "If omitted, segments all videos in VIDEO_DIR.",
    )
    seg_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-segment even if segments already exist.",
    )
    seg_parser.set_defaults(func=cmd_segment)

    # --- ingest ---
    ing_parser = subparsers.add_parser("ingest", help="Analyze segments and store results")
    ing_parser.add_argument(
        "--file", "-f",
        help="Ingest segments for a single video file. "
             "If omitted, ingests all segmented videos.",
    )
    ing_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers for API calls (default: 4)",
    )
    ing_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if segments already exist in the database. Bypasses DB checks for already-ingested segments (useful for first run).",
    )
    ing_parser.set_defaults(func=cmd_ingest)

    # --- search ---
    search_parser = subparsers.add_parser("search", help="Search video segments")
    search_parser.add_argument("query", help="Natural language search query")
    search_parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    search_parser.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
