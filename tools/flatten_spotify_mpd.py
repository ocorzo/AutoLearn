#!/usr/bin/env python3
"""Aplana el Spotify Million Playlist Dataset a JSON o JSON Lines.

La salida sólo conserva: pid, nombre de playlist, canción y artista.
El formato recomendado es JSON Lines (un objeto por línea), porque permite
procesar decenas de millones de canciones sin cargar todo el resultado en RAM.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import TextIO


def open_text(path: Path, mode: str) -> TextIO:
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Carpeta Data del MPD")
    parser.add_argument("output", type=Path, help="Salida .jsonl/.json, opcionalmente terminada en .gz")
    parser.add_argument(
        "--format",
        choices=("jsonl", "array"),
        default="jsonl",
        help="jsonl es el predeterminado y el más práctico para este tamaño",
    )
    parser.add_argument("--limit-files", type=int, help="Procesa sólo N slices; útil para una prueba")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error(f"No es una carpeta: {args.data_dir}")
    if args.output.exists():
        parser.error(f"La salida ya existe: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"No existe la carpeta de salida: {args.output.parent}")

    sources = sorted(args.data_dir.glob("mpd.slice.*.json"))
    if args.limit_files is not None:
        sources = sources[: args.limit_files]
    if not sources:
        parser.error("No se encontraron archivos mpd.slice.*.json")

    written = 0
    pids: set[int] = set()
    # Escribimos en el mismo volumen y sólo publicamos el archivo final cuando
    # todos los slices hayan terminado. Así una interrupción no parece un
    # export completo.
    suffix = ".partial.gz" if args.output.suffix == ".gz" else ".partial"
    fd, temp_name = tempfile.mkstemp(prefix=f".{args.output.stem}.", suffix=suffix, dir=args.output.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        with open_text(temporary, "wt") as output:
            if args.format == "array":
                output.write("[\n")
            first = True
            for source in sources:
                with source.open(encoding="utf-8") as handle:
                    for playlist in json.load(handle)["playlists"]:
                        pid = playlist["pid"]
                        if pid in pids:
                            raise ValueError(f"PID duplicado en el origen: {pid}")
                        pids.add(pid)
                        base = {"pid": pid, "playlist_name": playlist["name"]}
                        for track in playlist["tracks"]:
                            record = {
                                **base,
                                "track_name": track["track_name"],
                                "artist_name": track["artist_name"],
                            }
                            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                            if args.format == "array":
                                if not first:
                                    output.write(",\n")
                                output.write(encoded)
                                first = False
                            else:
                                output.write(encoded + "\n")
                            written += 1
                print(f"{source.name}: {len(pids):,} playlists, {written:,} canciones", flush=True)
            if args.format == "array":
                output.write("\n]\n")
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Terminado: {len(pids):,} playlists y {written:,} registros en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
