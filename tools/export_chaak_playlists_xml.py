#!/usr/bin/env python3
"""Export all Chaak playlists as a Music/iTunes playlist XML file.

The generated XML references tracks already known to the local Music library
export.  It is intended for Music > File > Library > Import Playlist.
"""

from __future__ import annotations

import argparse
import csv
import json
import plistlib
import re
import unicodedata
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFTS = ROOT / "embedding-playlist-drafts"


def normalized(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", plain.casefold())


def identity(title: str, artist: str) -> tuple[str, str]:
    return normalized(title), normalized(artist)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owned-csv", type=Path, default=DEFAULT_DRAFTS / "embedding_playlist_owned_tracks.csv")
    parser.add_argument("--names-csv", type=Path, default=DEFAULT_DRAFTS / "playlist_names.csv")
    parser.add_argument("--audit", type=Path, default=ROOT / "music-audit-final.json")
    parser.add_argument(
        "--library-xml",
        type=Path,
        required=True,
        help="Exportación XML de la biblioteca de Music/iTunes que se usará para resolver Track IDs",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DRAFTS / "Chaak Playlists.xml")
    parser.add_argument("--missing-report", type=Path, default=DEFAULT_DRAFTS / "xml_tracks_not_in_library_export.csv")
    parser.add_argument("--prefix", default="Chaak — ")
    args = parser.parse_args()

    names = {row["profile_id"]: row["playlist_name"] for row in csv.DictReader(args.names_csv.open(encoding="utf-8"))}
    songs_by_profile: dict[str, list[tuple[str, str]]] = {}
    for row in csv.DictReader(args.owned_csv.open(encoding="utf-8")):
        songs_by_profile.setdefault(row["profile_id"], []).append((row["local_track_name"], row["local_artist_name"]))

    audit = json.load(args.audit.open(encoding="utf-8"))["files"]
    path_by_song = {
        identity(record["metadata"]["title"], record["metadata"]["artist"]): record["path"]
        for record in audit
        if record.get("status") == "ok" and record.get("metadata", {}).get("title") and record.get("metadata", {}).get("artist")
    }
    library = plistlib.load(args.library_xml.open("rb"))
    track_id_by_path = {
        urllib.parse.unquote(track.get("Location", "").removeprefix("file://")): int(track_id)
        for track_id, track in library["Tracks"].items()
        if track.get("Location")
    }

    exported_ids: set[int] = set()
    playlists = []
    missing_rows = []
    for offset, profile_id in enumerate(sorted(songs_by_profile), start=1):
        track_ids = []
        for title, artist in songs_by_profile[profile_id]:
            path = path_by_song.get(identity(title, artist))
            track_id = track_id_by_path.get(path or "")
            if track_id is None:
                missing_rows.append({"profile_id": profile_id, "playlist_name": names[profile_id], "track_name": title, "artist_name": artist, "path": path or ""})
                continue
            track_ids.append(track_id)
            exported_ids.add(track_id)
        playlists.append(
            {
                "Name": args.prefix + names[profile_id],
                "Playlist ID": 9_000_000 + offset,
                "Playlist Persistent ID": f"C0DE{offset:012X}",
                "Playlist Items": [{"Track ID": track_id} for track_id in track_ids],
            }
        )

    export = {
        "Major Version": 1,
        "Minor Version": 1,
        "Application Version": library.get("Application Version", "1.0"),
        "Library Persistent ID": library.get("Library Persistent ID", ""),
        "Tracks": {str(track_id): library["Tracks"][str(track_id)] for track_id in sorted(exported_ids)},
        "Playlists": playlists,
    }
    with args.output.open("wb") as handle:
        plistlib.dump(export, handle, fmt=plistlib.FMT_XML, sort_keys=False)
    with args.missing_report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["profile_id", "playlist_name", "track_name", "artist_name", "path"])
        writer.writeheader()
        writer.writerows(missing_rows)
    print(f"Wrote {args.output}: {len(playlists)} playlists, {len(exported_ids)} tracks; {len(missing_rows)} tracks need a later direct add.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
