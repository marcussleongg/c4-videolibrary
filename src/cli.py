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
    print("TODO: ingest not yet implemented")


def cmd_search(args):
    """Search for video segments matching a natural language query."""
    print("TODO: search not yet implemented")


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
