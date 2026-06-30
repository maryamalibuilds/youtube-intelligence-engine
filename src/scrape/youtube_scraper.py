"""Scrape comments from YouTube videos via the YouTube Data API v3.

Usage:
    python -m src.scrape.youtube_scraper --video VIDEO_ID --max 500
    python -m src.scrape.youtube_scraper --query "react native" --videos 5 --max 300

Falls back to a tiny built-in sample (no API key needed) so the rest of the
pipeline is runnable offline for development. Set YOUTUBE_API_KEY in .env to
scrape for real. The Moodle sample code maps onto fetch_comments() below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from config import RAW_DIR, YOUTUBE_API_KEY, has_youtube

_SAMPLE = [
    {"video_id": "demo", "comment_id": "c1", "author": "alice",
     "text": "idk why ppl h8 this tutorial, it's gr8 tbh 🔥", "likes": 12},
    {"video_id": "demo", "comment_id": "c2", "author": "bob",
     "text": "this vid sux, the audio is soooo bad omg", "likes": 3},
    {"video_id": "demo", "comment_id": "c3", "author": "carol",
     "text": "y'all gonna luv the part @ 4:20, fo shizzle c u l8r", "likes": 27},
]


def _client():
    from googleapiclient.discovery import build

    return build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def fetch_comments(video_id: str, max_comments: int = 500) -> List[Dict]:
    """Return top-level comments (threaded replies flattened) for a video."""
    if not has_youtube():
        print("[scraper] No YOUTUBE_API_KEY set -> returning offline sample.")
        return [c for c in _SAMPLE]

    yt = _client()
    out: List[Dict] = []
    page_token = None
    while len(out) < max_comments:
        resp = (
            yt.commentThreads()
            .list(part="snippet", videoId=video_id, maxResults=100,
                  pageToken=page_token, textFormat="plainText", order="relevance")
            .execute()
        )
        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "video_id": video_id,
                "comment_id": item["id"],
                "author": s.get("authorDisplayName", ""),
                "text": s.get("textDisplay", ""),
                "likes": s.get("likeCount", 0),
                "published_at": s.get("publishedAt", ""),
            })
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out[:max_comments]


def search_videos(query: str, n: int = 5) -> List[str]:
    if not has_youtube():
        return ["demo"]
    yt = _client()
    resp = yt.search().list(part="id", q=query, type="video", maxResults=n).execute()
    return [i["id"]["videoId"] for i in resp.get("items", [])]


def save(comments: List[Dict], name: str) -> Path:
    path = RAW_DIR / f"{name}.json"
    path.write_text(json.dumps(comments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[scraper] wrote {len(comments)} comments -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="single video id")
    ap.add_argument("--query", help="search query (scrapes top --videos results)")
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--max", type=int, default=500, help="max comments per video")
    ap.add_argument("--out", default="comments")
    args = ap.parse_args()

    all_comments: List[Dict] = []
    if args.video:
        all_comments = fetch_comments(args.video, args.max)
    elif args.query:
        for vid in search_videos(args.query, args.videos):
            all_comments += fetch_comments(vid, args.max)
    else:
        all_comments = fetch_comments("demo", args.max)
    save(all_comments, args.out)


if __name__ == "__main__":
    main()
