#!/usr/bin/env python3
"""Restore this checkout's filter-aware artifact link after a fresh clone."""
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_STORAGE = Path("/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    storage = args.storage_root.expanduser().resolve(strict=True)
    for level in range(3):
        folder = storage / f"1023/hsc-warmstart/level{level}"
        for name in ("latest.pt", "summary.json", "manifest.json", "selected/photoz_head.pt",
                     "selected/adapter/adapter_config.json", "selected/adapter/adapter_model.safetensors"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
    link = args.workspace.expanduser().resolve(strict=True) / "filter-aware-runs"
    if link.is_symlink():
        if link.resolve() != storage:
            raise FileExistsError(f"Existing link points elsewhere: {link}")
    elif link.exists():
        raise FileExistsError(f"Preserve existing artifacts before linking; refusing to replace {link}")
    else:
        link.symlink_to(storage, target_is_directory=True)
    print(f"{link} -> {storage}")


if __name__ == "__main__":
    main()
