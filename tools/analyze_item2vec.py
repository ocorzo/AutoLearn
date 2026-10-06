#!/usr/bin/env python3
"""Analyze local-song embeddings and build a mutual-neighbor graph.

This produces diagnostic artifacts only. Communities are preliminary clusters,
not final playlists; they must later be named, reviewed and diversified.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
from gensim.models import KeyedVectors


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--graph-neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kv = KeyedVectors.load(str(args.vectors), mmap="r")
    connection = duckdb.connect(str(args.db), read_only=True)
    records = connection.execute(
        """
        SELECT l.track_key AS local_track_key, l.track_name AS local_track_name,
               l.artist_name AS local_artist_name, r.mpd_track_key,
               c.playlist_occurrences
        FROM local_track_resolutions r
        INNER JOIN local_tracks l ON l.track_key = r.local_track_key
        INNER JOIN mpd_catalog c ON c.track_key = r.mpd_track_key
        LEFT JOIN local_track_exclusions e ON e.track_key = l.track_key
        WHERE e.track_key IS NULL AND c.playlist_occurrences >= 5
        ORDER BY l.artist_name, l.track_name
        """
    ).fetchall()
    playlist_memberships = connection.execute(
        """
        SELECT cl.pid, cl.track_key
        FROM candidate_local_tracks cl
        INNER JOIN local_track_resolutions r ON r.local_track_key = cl.track_key
        INNER JOIN mpd_catalog c ON c.track_key = r.mpd_track_key
        WHERE c.playlist_occurrences >= 5
        """
    ).fetchall()
    connection.close()

    local_rows: list[dict] = []
    vectors: list[np.ndarray] = []
    index_by_local_key: dict[str, int] = {}
    for local_key, local_name, local_artist, mpd_key, occurrences in records:
        if mpd_key not in kv:
            continue
        index_by_local_key[local_key] = len(local_rows)
        local_rows.append(
            {
                "local_track_key": local_key,
                "local_track_name": local_name,
                "local_artist_name": local_artist,
                "mpd_track_key": mpd_key,
                "mpd_playlist_occurrences": occurrences,
            }
        )
        vectors.append(np.asarray(kv[mpd_key], dtype=np.float32))

    matrix = np.vstack(vectors)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)
    track_count = len(local_rows)
    neighbor_count = min(args.neighbors, track_count - 1)
    graph_neighbor_count = min(args.graph_neighbors, neighbor_count)

    neighbor_indices: list[list[int]] = []
    neighbor_rows: list[dict] = []
    for index, row in enumerate(local_rows):
        nearest = np.argpartition(similarity[index], -neighbor_count)[-neighbor_count:]
        nearest = nearest[np.argsort(similarity[index, nearest])[::-1]]
        nearest_list = [int(value) for value in nearest]
        neighbor_indices.append(nearest_list)
        for rank, other_index in enumerate(nearest_list, start=1):
            other = local_rows[other_index]
            neighbor_rows.append(
                {
                    "local_track_key": row["local_track_key"],
                    "local_track_name": row["local_track_name"],
                    "local_artist_name": row["local_artist_name"],
                    "neighbor_rank": rank,
                    "neighbor_track_key": other["local_track_key"],
                    "neighbor_track_name": other["local_track_name"],
                    "neighbor_artist_name": other["local_artist_name"],
                    "cosine_similarity": f"{similarity[index, other_index]:.6f}",
                }
            )

    neighbor_sets = [set(indices[:graph_neighbor_count]) for indices in neighbor_indices]
    graph = nx.Graph()
    graph.add_nodes_from(range(track_count))
    edge_rows: list[dict] = []
    for left in range(track_count):
        for right in neighbor_sets[left]:
            if left >= right or left not in neighbor_sets[right]:
                continue
            score = float(similarity[left, right])
            if score <= 0:
                continue
            graph.add_edge(left, right, weight=score)
            edge_rows.append(
                {
                    "left_track_key": local_rows[left]["local_track_key"],
                    "left_track_name": local_rows[left]["local_track_name"],
                    "left_artist_name": local_rows[left]["local_artist_name"],
                    "right_track_key": local_rows[right]["local_track_key"],
                    "right_track_name": local_rows[right]["local_track_name"],
                    "right_artist_name": local_rows[right]["local_artist_name"],
                    "cosine_similarity": f"{score:.6f}",
                }
            )

    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=args.resolution, seed=args.seed
    )
    community_rows: list[dict] = []
    ordered_communities = sorted(communities, key=lambda group: (-len(group), min(group)))
    for community_id, members in enumerate(ordered_communities, start=1):
        for member in sorted(members):
            community_rows.append(
                {
                    "community_id": community_id,
                    "community_size": len(members),
                    **local_rows[member],
                }
            )

    memberships_by_pid: dict[int, list[int]] = defaultdict(list)
    for pid, local_key in playlist_memberships:
        index = index_by_local_key.get(local_key)
        if index is not None:
            memberships_by_pid[pid].append(index)
    validation_ranks: list[int] = []
    validation_count = 0
    for members in memberships_by_pid.values():
        unique_members = sorted(set(members))
        if len(unique_members) < 5:
            continue
        for held_out in unique_members[:3]:
            context = [index for index in unique_members if index != held_out]
            profile = matrix[context].mean(axis=0)
            profile /= np.linalg.norm(profile)
            scores = matrix @ profile
            scores[context] = -np.inf
            rank = int(np.count_nonzero(scores > scores[held_out]) + 1)
            validation_ranks.append(rank)
            validation_count += 1

    upper_triangle = similarity[np.triu_indices(track_count, k=1)]
    report = {
        "local_tracks_with_embeddings": track_count,
        "embedding_dimensions": int(matrix.shape[1]),
        "nearest_neighbors_per_track": neighbor_count,
        "mutual_knn_neighbors_per_track": graph_neighbor_count,
        "mutual_graph_edges": graph.number_of_edges(),
        "isolated_tracks": len(list(nx.isolates(graph))),
        "louvain_communities": len(ordered_communities),
        "community_sizes": sorted((len(group) for group in ordered_communities), reverse=True),
        "pairwise_cosine_quantiles": {
            str(quantile): round(float(np.quantile(upper_triangle, quantile)), 6)
            for quantile in (0.50, 0.75, 0.90, 0.95, 0.99)
        },
        "mutual_edge_cosine_quantiles": {
            str(quantile): round(float(np.quantile([float(row["cosine_similarity"]) for row in edge_rows], quantile)), 6)
            for quantile in (0.05, 0.25, 0.50, 0.75, 0.95)
        },
        "leave_one_out": {
            "test_cases": validation_count,
            "median_rank_among_local_tracks": int(np.median(validation_ranks)) if validation_ranks else None,
            "hit_at_10": round(float(np.mean(np.array(validation_ranks) <= 10)), 4) if validation_ranks else None,
            "hit_at_25": round(float(np.mean(np.array(validation_ranks) <= 25)), 4) if validation_ranks else None,
            "hit_at_50": round(float(np.mean(np.array(validation_ranks) <= 50)), 4) if validation_ranks else None,
        },
        "validation_note": "Leave-one-out within candidate playlists; a cohesion diagnostic, not an independent held-out MPD test.",
    }
    write_csv(args.output_dir / "local_embedding_tracks.csv", list(local_rows[0]), local_rows)
    write_csv(args.output_dir / "local_embedding_neighbors.csv", list(neighbor_rows[0]), neighbor_rows)
    write_csv(args.output_dir / "local_embedding_mutual_edges.csv", list(edge_rows[0]), edge_rows)
    write_csv(args.output_dir / "local_embedding_communities.csv", list(community_rows[0]), community_rows)
    (args.output_dir / "local_embedding_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
