#!/usr/bin/env python3
"""Draft diverse local playlists and embedding-based missing-track recommendations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
from gensim.models import KeyedVectors


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_artist(value: str) -> str:
    return " ".join(value.casefold().split())


VARIANT_TERMS = r"live|remaster(?:ed)?|mix|edit|version|mono|stereo|acoustic|demo|anniversary|deluxe|radio|single|album"


def canonical_song_identity(title: str, artist: str) -> str:
    """Match songs by title/artist while ignoring release-version decorations."""
    base_title = re.sub(
        rf"\s*[\(\[](?=[^\)\]]*(?:{VARIANT_TERMS}))[^\)\]]*[\)\]]",
        "",
        title,
        flags=re.IGNORECASE,
    )
    base_title = re.sub(
        rf"\s*[-–—]\s*.*(?:{VARIANT_TERMS}).*$",
        "",
        base_title,
        flags=re.IGNORECASE,
    )
    def simplify(value: str) -> str:
        ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-z0-9]+", "", ascii_value.casefold())
    return f"{simplify(base_title)}::{simplify(artist)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--communities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-community-size", type=int, default=8)
    parser.add_argument("--maximum-dominant-artist-share", type=float, default=0.50)
    parser.add_argument("--recommendations-per-profile", type=int, default=50)
    parser.add_argument("--minimum-recommendation-occurrences", type=int, default=25)
    parser.add_argument("--early-diversity-window", type=int, default=20)
    parser.add_argument("--minimum-early-artists", type=int, default=6)
    parser.add_argument("--maximum-early-artist-share", type=float, default=0.50)
    parser.add_argument("--artist-repeat-penalty", type=float, default=0.015)
    parser.add_argument("--new-artist-bonus", type=float, default=0.010)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    community_rows = list(csv.DictReader(args.communities.open(encoding="utf-8")))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in community_rows:
        grouped[row["community_id"]].append(row)

    profiles: list[tuple[str, list[dict], str, float]] = []
    excluded: list[dict] = []
    for community_id, members in sorted(grouped.items(), key=lambda item: int(item[0])):
        artist_counts = Counter(member["local_artist_name"] for member in members)
        dominant_artist, dominant_count = artist_counts.most_common(1)[0]
        dominant_share = dominant_count / len(members)
        if len(members) < args.minimum_community_size or dominant_share > args.maximum_dominant_artist_share:
            excluded.append(
                {
                    "source_community_id": community_id,
                    "community_size": len(members),
                    "dominant_artist": dominant_artist,
                    "dominant_artist_share": f"{dominant_share:.4f}",
                    "reason": "too_small" if len(members) < args.minimum_community_size else "artist_centric",
                }
            )
            continue
        profiles.append((community_id, members, dominant_artist, dominant_share))

    kv = KeyedVectors.load(str(args.vectors), mmap="r")
    normalized_vectors = kv.get_normed_vectors()
    vector_index = {key: index for index, key in enumerate(kv.index_to_key)}
    connection = duckdb.connect(str(args.db), read_only=True)
    catalog = {
        key: (track_name, artist_name, occurrences)
        for key, track_name, artist_name, occurrences in connection.execute(
            """
            SELECT track_key, track_name, artist_name, playlist_occurrences
            FROM mpd_catalog
            WHERE playlist_occurrences >= 5
            """
        ).fetchall()
    }
    owned_keys = {
        key
        for (key,) in connection.execute("SELECT DISTINCT mpd_track_key FROM local_track_resolutions").fetchall()
    }
    owned_song_identities = {
        canonical_song_identity(track_name, artist_name)
        for track_name, artist_name in connection.execute(
            """
            SELECT l.track_name, l.artist_name
            FROM local_tracks l
            LEFT JOIN local_track_exclusions e ON e.track_key = l.track_key
            WHERE e.track_key IS NULL
            """
        ).fetchall()
    }
    connection.close()
    owned_indices = [vector_index[key] for key in owned_keys if key in vector_index]

    profile_rows: list[dict] = []
    owned_track_rows: list[dict] = []
    recommendation_rows: list[dict] = []
    for profile_rank, (community_id, members, dominant_artist, dominant_share) in enumerate(profiles, start=1):
        vector_indices = [vector_index[member["mpd_track_key"]] for member in members if member["mpd_track_key"] in vector_index]
        centroid = normalized_vectors[vector_indices].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        internal_scores = normalized_vectors[vector_indices] @ centroid
        profile_id = f"embedding_{profile_rank:02d}"
        profile_rows.append(
            {
                "profile_id": profile_id,
                "source_community_id": community_id,
                "owned_track_count": len(members),
                "dominant_artist": dominant_artist,
                "dominant_artist_share": f"{dominant_share:.4f}",
                "mean_similarity_to_profile": f"{float(internal_scores.mean()):.6f}",
                "status": "draft_diverse_profile",
            }
        )
        for member in members:
            owned_track_rows.append({"profile_id": profile_id, **member})

        scores = normalized_vectors @ centroid
        scores[owned_indices] = -np.inf
        candidate_count = min(len(scores), args.recommendations_per_profile * 200)
        candidate_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        candidates: list[dict] = []
        for index in candidate_indices:
            key = kv.index_to_key[int(index)]
            metadata = catalog.get(key)
            if metadata is None:
                continue
            track_name, artist_name, occurrences = metadata
            if occurrences < args.minimum_recommendation_occurrences:
                continue
            if canonical_song_identity(track_name, artist_name) in owned_song_identities:
                continue
            candidates.append(
                {
                    "index": int(index),
                    "mpd_track_key": key,
                    "track_name": track_name,
                    "artist_name": artist_name,
                    "artist_key": normalize_artist(artist_name),
                    "occurrences": occurrences,
                    "similarity": float(scores[index]),
                }
            )
        artist_counts: Counter[str] = Counter()
        selected_keys: set[str] = set()
        maximum_early_artist_tracks = int(args.early_diversity_window * args.maximum_early_artist_share)
        for recommendation_rank in range(1, args.recommendations_per_profile + 1):
            best: tuple[float, dict] | None = None
            early_position = recommendation_rank <= args.early_diversity_window
            slots_remaining_after = args.early_diversity_window - recommendation_rank
            artists_needed_before_pick = max(0, args.minimum_early_artists - len(artist_counts))
            for candidate in candidates:
                if candidate["mpd_track_key"] in selected_keys:
                    continue
                artist_key = candidate["artist_key"]
                artist_is_new = artist_key not in artist_counts
                if early_position:
                    if artist_counts[artist_key] >= maximum_early_artist_tracks:
                        continue
                    # When the remaining slots are exactly what we need for
                    # diversity, only an unseen artist may fill this slot.
                    if not artist_is_new and artists_needed_before_pick > slots_remaining_after:
                        continue
                adjusted_score = candidate["similarity"] - args.artist_repeat_penalty * np.log1p(artist_counts[artist_key])
                if early_position and artist_is_new:
                    adjusted_score += args.new_artist_bonus
                if best is None or adjusted_score > best[0]:
                    best = (float(adjusted_score), candidate)
            if best is None:
                break
            reranked_score, candidate = best
            artist_key = candidate["artist_key"]
            prior_artist_count = artist_counts[artist_key]
            artist_counts[artist_key] += 1
            selected_keys.add(candidate["mpd_track_key"])
            recommendation_rows.append(
                {
                    "profile_id": profile_id,
                    "recommendation_rank": recommendation_rank,
                    "mpd_track_key": candidate["mpd_track_key"],
                    "track_name": candidate["track_name"],
                    "artist_name": candidate["artist_name"],
                    "embedding_similarity": f"{candidate['similarity']:.6f}",
                    "reranked_score": f"{reranked_score:.6f}",
                    "artist_previous_recommendations": prior_artist_count,
                    "distinct_artists_so_far": len(artist_counts),
                    "mpd_playlist_occurrences": candidate["occurrences"],
                    "score_interpretation": "embedding_affinity_not_calibrated_probability",
                }
            )

    write_csv(args.output_dir / "embedding_playlist_profiles.csv", list(profile_rows[0]), profile_rows)
    write_csv(args.output_dir / "embedding_playlist_owned_tracks.csv", list(owned_track_rows[0]), owned_track_rows)
    write_csv(args.output_dir / "embedding_playlist_missing_recommendations.csv", list(recommendation_rows[0]), recommendation_rows)
    write_csv(args.output_dir / "embedding_excluded_communities.csv", list(excluded[0]), excluded)
    summary = {
        "draft_profiles": len(profiles),
        "owned_tracks_assigned": len(owned_track_rows),
        "excluded_communities": len(excluded),
        "missing_track_recommendations": len(recommendation_rows),
        "minimum_community_size": args.minimum_community_size,
        "maximum_dominant_artist_share": args.maximum_dominant_artist_share,
        "minimum_recommendation_occurrences": args.minimum_recommendation_occurrences,
        "early_diversity_window": args.early_diversity_window,
        "minimum_early_artists": args.minimum_early_artists,
        "maximum_early_artist_share": args.maximum_early_artist_share,
        "artist_repeat_penalty": args.artist_repeat_penalty,
        "note": "Embedding similarity is an affinity score. It requires later calibration before being presented as a probability or certainty.",
    }
    (args.output_dir / "embedding_playlist_drafts_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
