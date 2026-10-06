#!/usr/bin/env python3
"""Extrae canciones únicas de un informe de auditoría musical.

El álbum se ignora deliberadamente: una canción se identifica solamente por
su título y artista normalizados. La salida JSON Lines conserva el texto
original para mostrarlo al usuario y una llave estable para las uniones.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def normalize(value: str) -> str:
    """Normaliza mayúsculas, acentos, espacios y puntuación no significativa."""
    decomposed = unicodedata.normalize("NFKD", value.casefold().replace("&", " and "))
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return "".join(char for char in without_accents if char.isalnum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path, help="Informe JSON generado por music_metadata.py")
    parser.add_argument("output", type=Path, help="Salida JSON Lines")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"La salida ya existe: {args.output}")

    inventory = json.loads(args.audit.read_text(encoding="utf-8"))
    tracks: dict[str, dict[str, str]] = {}
    skipped = 0
    for item in inventory["files"]:
        metadata = item.get("metadata") or {}
        title = (metadata.get("title") or "").strip()
        artist = (metadata.get("artist") or "").strip()
        if not title or not artist:
            skipped += 1
            continue
        track_key = f"{normalize(title)}::{normalize(artist)}"
        tracks.setdefault(track_key, {"track_key": track_key, "track_name": title, "artist_name": artist})

    with args.output.open("x", encoding="utf-8") as handle:
        for track in sorted(tracks.values(), key=lambda item: item["track_key"]):
            handle.write(json.dumps(track, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"{len(tracks):,} canciones únicas exportadas; {skipped:,} archivos sin título o artista.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
