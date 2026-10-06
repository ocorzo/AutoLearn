#!/usr/bin/env python3
"""Create the Chaak playlists in macOS Music without copying audio files.

Default mode is a dry run.  Use --apply only after reviewing its report.
The script locates tracks by their exact local file location in Music's library,
so it neither re-imports nor duplicates audio files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFTS = ROOT / "embedding-playlist-drafts"
DEFAULT_AUDIT = ROOT / "music-audit-final.json"


def normalized(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", plain.casefold())


def identity(title: str, artist: str) -> tuple[str, str]:
    return normalized(title), normalized(artist)


def applescript_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def music_script(playlist_name: str, songs: list[tuple[str, str]], replace: bool) -> str:
    name = applescript_string(playlist_name)
    if replace:
        playlist_setup = f"""
        if exists user playlist {name} then delete user playlist {name}
        set targetPlaylist to make new user playlist with properties {{name:{name}}}
        """
    else:
        playlist_setup = f"""
        if exists user playlist {name} then
            error "La playlist ya existe: {playlist_name}"
        end if
        set targetPlaylist to make new user playlist with properties {{name:{name}}}
        """
    lookup_lines = []
    for title, artist in songs:
        lookup_lines.append(
            f"""
    try
        set end of sourceTracks to (some file track of library playlist 1 whose name is {applescript_string(title)} and artist is {applescript_string(artist)})
    on error errorMessage number errorNumber
        log "SKIPPED {title} — {artist}: " & errorMessage
    end try"""
        )
    return f"""
tell application "Music"
    {playlist_setup}
    set sourceTracks to {{}}
    {''.join(lookup_lines)}
    repeat with sourceTrack in sourceTracks
        duplicate (contents of sourceTrack) to targetPlaylist
    end repeat
end tell
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owned-csv", type=Path, default=DEFAULT_DRAFTS / "embedding_playlist_owned_tracks.csv")
    parser.add_argument("--names-csv", type=Path, default=DEFAULT_DRAFTS / "playlist_names.csv")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--prefix", default="Chaak — ")
    parser.add_argument("--apply", action="store_true", help="Create the playlists in Music; otherwise only print the plan.")
    parser.add_argument("--replace", action="store_true", help="Delete and recreate a same-named playlist (requires --apply).")
    args = parser.parse_args()
    if args.replace and not args.apply:
        parser.error("--replace requires --apply")

    names = {row["profile_id"]: row["playlist_name"] for row in csv.DictReader(args.names_csv.open(encoding="utf-8"))}
    songs_by_profile: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in csv.DictReader(args.owned_csv.open(encoding="utf-8")):
        songs_by_profile[row["profile_id"]].append((row["local_track_name"], row["local_artist_name"]))

    audit = json.load(args.audit.open(encoding="utf-8"))
    path_by_song: dict[tuple[str, str], str] = {}
    for record in audit["files"]:
        metadata = record.get("metadata") or {}
        title, artist = metadata.get("title"), metadata.get("artist")
        if record.get("status") != "ok" or not title or not artist:
            continue
        path_by_song.setdefault(identity(title, artist), record["path"])

    plan: list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]] = []
    for profile_id in sorted(songs_by_profile):
        if profile_id not in names:
            raise ValueError(f"Missing name for {profile_id}")
        resolved_songs, missing = [], []
        for title, artist in songs_by_profile[profile_id]:
            path = path_by_song.get(identity(title, artist))
            if path:
                resolved_songs.append((title, artist))
            else:
                missing.append((title, artist))
        plan.append((args.prefix + names[profile_id], resolved_songs, missing))

    total_tracks = sum(len(songs) for _, songs, _ in plan)
    total_missing = sum(len(missing) for _, _, missing in plan)
    print(f"{len(plan)} playlists; {total_tracks} local tracks ready; {total_missing} unresolved paths.")
    for name, songs, missing in plan:
        suffix = f"; {len(missing)} unresolved" if missing else ""
        print(f"- {name}: {len(songs)} tracks{suffix}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to create these playlists in Music.")
        return 0

    for name, songs, missing in plan:
        result = subprocess.run(
            ["osascript", "-"], input=music_script(name, songs, args.replace), text=True, capture_output=True
        )
        if result.returncode:
            print(f"FAILED: {name}\n{result.stderr.strip()}", file=sys.stderr)
            return result.returncode
        if missing:
            print(f"Created {name}; skipped {len(missing)} tracks without a local path.")
        else:
            print(f"Created {name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
