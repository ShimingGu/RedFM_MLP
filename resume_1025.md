# Recover the two lost sessions

`codex resume` cannot find these sessions in this environment: their IDs and user
prompt history survived, but their original transcripts/thread records did not.
The IDs alone cannot recreate a resumable conversation.

## Start replacement sessions with the recovered context

Run one command per terminal. These open **new conversations**, load the recovered
history and run mapping, and begin with a current progress check.

First session — Summarise current progress:

```bash
codex -C /scratch/RedFM_MLP "Read logs/recovered_session_1_1025.md and the recovery inventory it references. Continue from that recovered context. First verify current run status and report progress."
```

Second session — Change GPU device / filter-aware experiments:

```bash
codex -C /scratch/RedFM_MLP "Read logs/recovered_session_2_1025.md and the recovery inventory it references. Continue from that recovered context. First verify current run status and report progress."
```

## Resume the original conversations if their storage is restored

These commands require the original transcript/thread storage, or execution in
an environment where that storage still exists:

```bash
codex resume 01a157ce-9bd6-7872-b0b8-ccefd35855ef
```

```bash
codex resume 01a15ddd-7108-7c21-b878-fda7069abc8c
```

The replacement context contains 90 recovered user prompts and links to surviving
logs/checkpoints. It does not include the missing assistant/tool messages. No
Codex databases or transcript files were fabricated or modified.
