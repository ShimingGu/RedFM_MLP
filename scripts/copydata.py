#!/usr/bin/env python3
"""Copy CLAUDS data from source directory to local project data directory."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import pathlib
import shutil
import sys
import time


def copy_or_link_file(
    src_file: pathlib.Path,
    dst_file: pathlib.Path,
    symlink: bool,
    overwrite: bool,
    dry_run: bool,
) -> tuple[pathlib.Path, str]:
    """Copy or symlink a single file from src_file to dst_file."""
    action = "Link" if symlink else "Copy"
    
    if dry_run:
        return src_file, f"[DRY-RUN] Would {action.lower()} to {dst_file}"

    # Create parent directory
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    if dst_file.exists():
        if not overwrite:
            # Check if sizes match to skip
            try:
                src_stat = src_file.stat()
                dst_stat = dst_file.stat()
                if src_stat.st_size == dst_stat.st_size:
                    return src_file, f"Skipped (already exists with same size): {dst_file.name}"
            except Exception:
                pass
            return src_file, f"Skipped (already exists): {dst_file.name}"
        
        # If overwrite, delete existing (important for symlinks or files we want to replace)
        if dst_file.is_symlink() or dst_file.is_file():
            dst_file.unlink()
        elif dst_file.is_dir():
            shutil.rmtree(dst_file)

    try:
        if symlink:
            dst_file.symlink_to(src_file)
        else:
            # shutil.copy2 preserves metadata (mtime, etc)
            shutil.copy2(src_file, dst_file)
        return src_file, f"{action}ed: {dst_file.name}"
    except Exception as e:
        return src_file, f"Error: Failed to {action.lower()} {src_file.name} to {dst_file}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy or link Cosmic Imprint of Time CLAUDS data to the local project directory."
    )
    
    # Determine default destination directory relative to this script
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_dst = repo_root / "data" / "clauds"
    
    parser.add_argument(
        "--src",
        type=str,
        default="/arc/projects/ots/Cosmic_Imprint_of_Time/clauds",
        help="Source directory containing clauds data (default: %(default)s)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=str(default_dst),
        help="Destination directory (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--symlink",
        action="store_true",
        help="Create symbolic links instead of copying the files (recommended for speed and disk space)",
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files at destination",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for copying (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print actions that would be performed without modifying the filesystem",
    )
    
    args = parser.parse_args()
    
    src_dir = pathlib.Path(args.src)
    dst_dir = pathlib.Path(args.dst)
    
    if not src_dir.exists():
        print(f"Error: Source directory '{src_dir}' does not exist.", file=sys.stderr)
        return 1
    
    print(f"Scanning files in {src_dir}...")
    all_files: list[pathlib.Path] = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            all_files.append(pathlib.Path(root) / file)
            
    total_files = len(all_files)
    if total_files == 0:
        print("No files found in source directory.")
        return 0
        
    print(f"Found {total_files} files to process.")
    print(f"Destination: {dst_dir}")
    if args.symlink:
        print("Mode: Symlinking (symbolic links)")
    else:
        print("Mode: Copying (physical copy)")
        
    if args.dry_run:
        print("Running in DRY-RUN mode.")
        
    # Create target pairs
    tasks: list[tuple[pathlib.Path, pathlib.Path]] = []
    for src_file in all_files:
        rel_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel_path
        tasks.append((src_file, dst_file))
        
    # Run the copies
    completed = 0
    errors = 0
    skips = 0
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for I/O bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                copy_or_link_file,
                src_file,
                dst_file,
                args.symlink,
                args.overwrite,
                args.dry_run,
            ): (src_file, dst_file)
            for src_file, dst_file in tasks
        }
        
        for future in concurrent.futures.as_completed(futures):
            src_file, dst_file = futures[future]
            try:
                _, msg = future.result()
                completed += 1
                if "Error" in msg:
                    errors += 1
                    print(msg, file=sys.stderr)
                elif "Skipped" in msg:
                    skips += 1
                else:
                    # Print normal progress (throttle output to avoid overwhelming logs)
                    if completed % 100 == 0 or completed == total_files:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        print(f"Progress: {completed}/{total_files} files processed ({completed/total_files*100:.1f}%) - {rate:.1f} files/sec...")
            except Exception as exc:
                errors += 1
                print(f"Exception copying {src_file.name}: {exc}", file=sys.stderr)
                
    elapsed = time.time() - start_time
    print("\n--- Summary ---")
    print(f"Total files: {total_files}")
    print(f"Processed:   {completed}")
    print(f"Skipped:     {skips}")
    print(f"Errors:      {errors}")
    print(f"Time taken:  {elapsed:.2f} seconds")
    
    return 1 if errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
