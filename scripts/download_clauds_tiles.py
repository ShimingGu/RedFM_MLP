#!/usr/bin/env python3
"""Download CLAUDS public release tile images from CANFAR VOSpace.

By default this downloads science image FITS files from:
  vault:/clauds/desprez/PublicRelease/tilesv5

Companion weight maps and catalogues are skipped unless requested.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import urllib.parse
import urllib.request


VOSPACE_DIR = "vault:/clauds/desprez/PublicRelease/tilesv5"
PUBLIC_FILE_BASE = "https://www.canfar.net/storage/vault/file"
PUBLIC_PATH = "/clauds/desprez/PublicRelease/tilesv5"


def list_vospace_names() -> list[str]:
    result = subprocess.run(
        ["vls", VOSPACE_DIR],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def is_selected(name: str, include_weights: bool, include_catalogs: bool) -> bool:
    if name.endswith(".cat"):
        return include_catalogs
    if name.endswith(".weight.fits"):
        return include_weights
    return name.endswith(".fits")


def public_url(name: str) -> str:
    quoted_path = urllib.parse.quote(f"{PUBLIC_PATH}/{name}", safe="/")
    return f"{PUBLIC_FILE_BASE}{quoted_path}"


def download_one(name: str, output_dir: pathlib.Path, overwrite: bool) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / name
    if destination.exists() and not overwrite:
        print(f"exists: {destination}")
        return destination

    tmp = destination.with_suffix(destination.suffix + ".part")
    url = public_url(name)
    print(f"download: {name}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/clauds/images/tilesv5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-after", default=None)
    parser.add_argument("--include-weights", action="store_true")
    parser.add_argument("--include-catalogs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    names = [
        name
        for name in list_vospace_names()
        if is_selected(name, args.include_weights, args.include_catalogs)
    ]
    names.sort()

    if args.start_after:
        names = [name for name in names if name > args.start_after]
    if args.limit is not None:
        names = names[: args.limit]

    if args.list_only:
        for name in names:
            print(name)
        return 0

    output_dir = pathlib.Path(args.output_dir)
    for name in names:
        try:
            download_one(name, output_dir, args.overwrite)
        except Exception as exc:
            print(f"failed: {name}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
