# Filter-aware checkpoint storage

The complete generated artifact tree is stored at:

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/
```

This includes the completed A/B/C selected adapters and PDF heads, restart
checkpoints and optimizer/RNG state, original snapshots, source/query cohorts,
manifests, logs, pilot results and the common-flux evaluation.

The checkout's `filter-aware-runs` is a directory symlink to that location.
Existing experiment commands and report paths continue to work, and atomic
checkpoint writes through that directory go to the persistent storage.
The generated tree and the local Pixi environment are excluded from Git.
Compact report copies and figures are under `reports/filter-aware/`.

After a fresh clone in this environment, restore the artifact link before using
the experiment commands:

```bash
python scripts/link_filter_aware_storage.py
```

The helper verifies the essential A/B/C files and refuses to replace an existing
artifact directory or a link to a different location. The archived manifests
retain their original paths and hashes; recreating the link at the original
workspace path also preserves those absolute path references.

Every regular file was verified by SHA-256 during relocation. The inventory is:

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_storage_manifest.json
```

Earlier checkpoint blobs remain in existing Git history. Removing the generated
tree from the current index keeps future commits small without rewriting that
history.
