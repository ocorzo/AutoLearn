#!/usr/bin/env python3
"""Train playlist-context song embeddings from an ordered DuckDB corpus table.

The table must have ``pid`` and ``track_key`` columns, physically ordered by
``pid``. Each playlist is treated as a sentence; its tracks are reshuffled on
each corpus pass so a finite word2vec window samples different playlist
co-occurrences across epochs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import duckdb
from gensim.models import Word2Vec


class PlaylistCorpus:
    def __init__(self, db_path: Path, table_name: str, seed: int) -> None:
        self.db_path = db_path
        self.table_name = table_name
        self.seed = seed
        self.passes = 0

    def __iter__(self):
        pass_number = self.passes
        self.passes += 1
        connection = duckdb.connect(str(self.db_path), read_only=True)
        try:
            cursor = connection.execute(f"SELECT pid, track_key FROM {self.table_name}")
            current_pid = None
            playlist: list[str] = []
            while rows := cursor.fetchmany(100_000):
                for pid, track_key in rows:
                    if current_pid is None:
                        current_pid = pid
                    if pid != current_pid:
                        if len(playlist) >= 2:
                            random.Random(self.seed + pass_number * 1_000_003 + current_pid).shuffle(playlist)
                            yield playlist
                        playlist = []
                        current_pid = pid
                    playlist.append(track_key)
            if len(playlist) >= 2 and current_pid is not None:
                random.Random(self.seed + pass_number * 1_000_003 + current_pid).shuffle(playlist)
                yield playlist
        finally:
            connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--table", default="embedding_corpus_tracks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(args.db), read_only=True)
    playlist_count, token_count = connection.execute(
        f"""
        SELECT count(*), sum(track_count)
        FROM (
            SELECT pid, count(*) AS track_count
            FROM {args.table}
            GROUP BY pid
            HAVING count(*) >= 2
        )
        """
    ).fetchone()
    connection.close()
    logging.info("Corpus: %s playlists, %s track occurrences", playlist_count, token_count)

    corpus = PlaylistCorpus(args.db, args.table, args.seed)
    started = time.time()
    model = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=1,
        sg=1,
        negative=args.negative,
        sample=1e-4,
        workers=args.workers,
        seed=args.seed,
        sorted_vocab=1,
    )
    model.build_vocab(corpus_iterable=corpus, progress_per=100_000)
    logging.info("Vocabulary: %s tracks", len(model.wv))
    model.train(corpus_iterable=corpus, total_examples=playlist_count, epochs=args.epochs)

    vectors_path = args.output_dir / "mpd_item2vec_64d.kv"
    model_path = args.output_dir / "mpd_item2vec_64d.model"
    metadata_path = args.output_dir / "mpd_item2vec_64d.metadata.json"
    model.wv.save(str(vectors_path))
    model.save(str(model_path))
    metadata_path.write_text(
        json.dumps(
            {
                "model": "skip-gram negative-sampling item2vec",
                "vector_size": args.vector_size,
                "window": args.window,
                "negative": args.negative,
                "epochs": args.epochs,
                "seed": args.seed,
                "corpus_table": args.table,
                "playlist_count": playlist_count,
                "track_occurrences": token_count,
                "vocabulary_tracks": len(model.wv),
                "elapsed_seconds": round(time.time() - started, 1),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logging.info("Finished in %.1f seconds", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
