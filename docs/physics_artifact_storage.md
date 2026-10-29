# Physics / Cross-Matching storage

The 2026-10-29 migration moved 3,271 generated files (6,322,948,182 bytes,
5.89 GiB) to:

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/RedFM_MLP_artifacts/physics_branch2_20261029
```

The archive mirrors repository-relative paths. Every original and destination
file passed SHA-256 verification before the local copy was atomically replaced
by a symlink. All experiment input paths and file contents remain available in
this workspace. No new training or scientific re-selection was performed.

## Files retained in Git

Code, experiment prompts, Markdown reports, summary metrics, plots, small
metadata and verification manifests remain versioned. Numeric arrays, model
checkpoints, FITS catalogues, serialized `.jsonl` inputs, CSV files larger than
1 MiB and JSON files larger than 2 MiB in the experimental output directories
were archived. The exact exceptions and file hashes are recorded in
[configs/physics_artifact_storage.json](../configs/physics_artifact_storage.json).
The migration receipt is
[logs/physics_artifact_storage_20261029.json](../logs/physics_artifact_storage_20261029.json).

## Restore after cloning

Code and reports can be inspected without access to the project archive. To
run experiments using saved data or to open the large per-object tables, mount
or obtain access to the archive and run from the checkout:

```bash
python scripts/link_physics_storage.py --verify
```

For an alternative archive location containing the same relative paths:

```bash
python scripts/link_physics_storage.py --storage-root /path/to/archive --verify
```

For a read-only integrity check:

```bash
python scripts/link_physics_storage.py --check-only --verify
```

The script preflights the manifest, archive and destination paths before creating
links. It refuses to overwrite a regular file or redirect an existing link.
The `.gitignore` rules cover both generated files and restored links. Do not
force-add these artifacts. Existing filter-aware storage uses its separate
`scripts/link_filter_aware_storage.py` command and is unchanged.

Historical experiment manifests retain their original hashes and absolute
provenance paths; the archive move changes neither their contents nor the
scientific results. Reproduction on another machine also requires the original
external catalogues/model checkpoints and an appropriate environment; symlink
restoration alone does not install those dependencies.

## Lightweight branch history

The original oversized local snapshot was commit
`1e73f5004c4092473cfbb2fdc695016e195a0930` (`branch 2 initialisation`). It
contained the large data blobs. The 3,271 migrated files are preserved in the
external archive; the remaining physics code and reports are preserved in the
lightweight branch. Local `main` was reset to `origin/main`, removing that
oversized commit from its active history. Its local reflog may retain the old
commit temporarily; no aggressive history cleanup was performed.

`Cross-Matching` uses the original common base `d5da3d9` as the parent of its
lightweight physics snapshot, excluding the oversized commit from its ancestry.
Remote `main` has separately advanced with Paper I work; this branch keeps the
physics/Paper II work separate. The remote `main` branch is not rewritten.

The local Git object store may still occupy several GiB until its old unreachable
objects are eventually cleaned up, but pushing `Cross-Matching` does not send the
old local-only blobs. Future clones need archive access to restore large artifacts.

The publication command is:

```bash
git push -u origin Cross-Matching
```
