#!/usr/bin/env python3
"""Restore ignored physics artifacts from their checksum-manifested project archive.

This never copies data into Git or overwrites existing regular files. A fresh
clone needs access to the external archive; code, reports and figures do not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path, help='Defaults to configs/physics_artifact_storage.json')
    parser.add_argument('--storage-root', type=Path, help='Override the manifest archive location')
    parser.add_argument('--verify', action='store_true', help='Verify every archived file SHA-256 as well as its size')
    parser.add_argument('--check-only', action='store_true', help='Check existing links without creating any')
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve(strict=True)
    manifest_path = args.manifest or workspace / 'configs/physics_artifact_storage.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported storage manifest version')
    storage = (args.storage_root or Path(manifest['storage_root'])).expanduser().resolve(strict=True)
    pending = []
    seen = set()
    # Preflight the entire archive and workspace before mutating any path.
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] not in {'reports', 'fast-check-runs', 'fast-check-plots'}:
            raise ValueError(f'Unsafe artifact path: {relative}')
        if str(relative) in seen:
            raise ValueError(f'Duplicate artifact path: {relative}')
        seen.add(str(relative))
        target = storage / relative
        if not target.is_file() or target.stat().st_size != record['bytes']:
            raise ValueError(f'Missing archive file or size mismatch: {target}')
        if args.verify and sha256(target) != record['sha256']:
            raise ValueError(f'Archive checksum mismatch: {target}')
        link = workspace / relative
        if not link.parent.resolve().is_relative_to(workspace):
            raise ValueError(f'Artifact parent escapes the workspace: {link}')
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise FileExistsError(f'Existing link points elsewhere: {link}')
        elif link.exists():
            raise FileExistsError(f'Preserve existing data before linking; refusing to overwrite: {link}')
        elif args.check_only:
            raise FileNotFoundError(f'Local artifact link is missing: {link}')
        else:
            pending.append((link, target))
    if len(seen) != manifest['file_count']:
        raise ValueError('Manifest file count does not match its records')
    for link, target in pending:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    print(json.dumps({'artifacts': len(seen), 'created_links': len(pending),
                      'storage_root': str(storage), 'sha256_verified': args.verify,
                      'check_only': args.check_only}, indent=2))


if __name__ == '__main__':
    main()
