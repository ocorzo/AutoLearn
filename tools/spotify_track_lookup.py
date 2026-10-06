#!/usr/bin/env python3
"""Resolve canonical song identities to Spotify catalog tracks.

Credentials are read exclusively from environment variables.  The script never
writes client credentials or access tokens to disk.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def similarity(left: str, right: str) -> float:
    left_normalized = normalize(left)
    right_normalized = normalize(right)
    if left_normalized == right_normalized:
        return 1.0
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_tokens, right_tokens = set(left_normalized.split()), set(right_normalized.split())
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens | right_tokens else 0.0
    containment = 1.0 if left_normalized in right_normalized or right_normalized in left_normalized else 0.0
    return max(sequence, token_score, containment)


def request_json(request: Request, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 429 and attempt < retries - 1:
                time.sleep(float(error.headers.get("Retry-After", "2")))
                continue
            raise
        except URLError:
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def get_access_token(client_id: str, client_secret: str) -> str:
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    request = Request(
        TOKEN_URL,
        data=b"grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    return request_json(request)["access_token"]


def search_tracks(token: str, title: str, artist: str, market: str) -> list[dict]:
    query = urlencode(
        {
            "q": f'track:"{title}" artist:"{artist}"',
            "type": "track",
            "limit": 10,
            "market": market,
        }
    )
    request = Request(
        f"{SEARCH_URL}?{query}",
        headers={"Authorization": f"Bearer {token}"},
    )
    return request_json(request).get("tracks", {}).get("items", [])


def score_candidate(title: str, artist: str, candidate: dict) -> tuple[float, float, float]:
    candidate_artists = "; ".join(item["name"] for item in candidate.get("artists", []))
    title_score = similarity(title, candidate.get("name", ""))
    artist_score = similarity(artist, candidate_artists)
    return 0.7 * title_score + 0.3 * artist_score, title_score, artist_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--market", default="MX")
    parser.add_argument("--delay", type=float, default=0.22)
    args = parser.parse_args()

    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET", file=sys.stderr)
        return 2

    token = get_access_token(client_id, client_secret)
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fieldnames = [
        "local_track_key", "local_track_name", "local_artist_name",
        "canonical_track_name", "canonical_artist_name", "acoustid_score",
        "spotify_track_id", "spotify_uri", "spotify_track_name", "spotify_artist_names",
        "spotify_album_name", "spotify_duration_ms", "match_score", "title_score",
        "artist_score", "resolution_status", "search_error",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            result = {
                "local_track_key": row["local_track_key"],
                "local_track_name": row["local_track_name"],
                "local_artist_name": row["local_artist_name"],
                "canonical_track_name": row["identified_track_name"],
                "canonical_artist_name": row["identified_artist_name"],
                "acoustid_score": row["acoustid_score"],
                "resolution_status": "no_catalog_result",
                "search_error": "",
            }
            try:
                candidates = search_tracks(token, result["canonical_track_name"], result["canonical_artist_name"], args.market)
                if candidates:
                    candidate, scores = max(
                        ((candidate, score_candidate(result["canonical_track_name"], result["canonical_artist_name"], candidate)) for candidate in candidates),
                        key=lambda item: item[1][0],
                    )
                    score, title_score, artist_score = scores
                    result.update(
                        {
                            "spotify_track_id": candidate["id"],
                            "spotify_uri": candidate["uri"],
                            "spotify_track_name": candidate["name"],
                            "spotify_artist_names": "; ".join(item["name"] for item in candidate["artists"]),
                            "spotify_album_name": candidate["album"]["name"],
                            "spotify_duration_ms": candidate["duration_ms"],
                            "match_score": f"{score:.4f}",
                            "title_score": f"{title_score:.4f}",
                            "artist_score": f"{artist_score:.4f}",
                            "resolution_status": "high_confidence" if score >= 0.93 and title_score >= 0.90 else "review",
                        }
                    )
            except (HTTPError, URLError, KeyError, ValueError) as error:
                result["resolution_status"] = "search_error"
                result["search_error"] = f"{type(error).__name__}: {error}"
            writer.writerow(result)
            if index < len(rows):
                time.sleep(args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
