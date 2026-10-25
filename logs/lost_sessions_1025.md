# Lost Codex sessions and surviving runs — 2026-10-25

Inspected at 2026-10-25T09:37:37.297460+00:00. All timestamps below are UTC.

## Identified sessions

| Session | ID | First prompt | Last prompt | Surviving prompts |
|---|---|---|---|---|
| Summarise current progress | `01a157ce-9bd6-7872-b0b8-ccefd35855ef` | 2026-10-20T05:35:33+00:00 | 2026-10-25T05:50:09+00:00 | 52 |
| Change GPU device | `01a15ddd-7108-7c21-b878-fda7069abc8c` | 2026-10-21T09:50:22+00:00 | 2026-10-25T05:50:13+00:00 | 38 |

Evidence: [Codex prompt history](/arc/home/gsm/.codex/history.jsonl) (session 1 starts at line 670; session 2 starts at line 694) and [session index](/arc/home/gsm/.codex/session_index.jsonl). All 90 matching prompts, with original line numbers, are preserved in [the evidence JSON](/scratch/RedFM_MLP/logs/lost_sessions_1025_evidence.json). Guardian-review entries are separate automatic-review threads, not the two user sessions.

The full older rollout transcripts were not found. The `sessions` symlink currently resolves to `/scratch/.cache-gsm/data/_codex/sessions`, which contained only the current October 25 conversation and its guardian thread. The older home state database and the relocated active state database do not contain either target ID. Session identification and the mapping below use surviving user history, launch notes and experiment artifacts; this does not restore the conversations in the Codex UI. The storage relocation is observable, but the cause of transcript loss is unproven.

## Session-to-work mapping

- **Summarise current progress:** requested the DESI-Legacy/GalaxiesML-Spec catalogues and four 4B survey commands; managed the first three 4B runs; requested the HSC→Legacy/CLAUDS and CLAUDS→HSC/Legacy fast checks; subsequently managed 0.8B CLAUDS QLoRA/DoRA and DESI-Legacy QLoRA assignments.
- **Change GPU device:** requested a separate allocation for the 4B GalaxiesML-Spec DoRA run; prepared the 0.8B matrix and allocations for its GalaxiesML-Spec QLoRA/DoRA runs; ran/analysed GLM capacity, sensitivity and toy-pretraining checks; executed the filter-aware pilot and A→B→C warm-start experiments.
- These associations describe requested and managed work. The sessions monitored overlapping jobs; surviving prompt logs do not prove exclusive process ownership.

## Qwen runs

“Completed” requires a surviving adapter `result.pt`. Partial progress comes from the final training-log update and a nonempty periodic checkpoint. Epoch-final work may be newer than the last periodic checkpoint. No training process matching these jobs was visible in this environment. A status file saying `training` does not establish a live process.

| Run | Saved state | Training log | Latest periodic checkpoint |
|---|---|---|---|
| qwen-desi-ls-qlora-specz-e10 | Partial: 30,900/37,410 updates | [log](/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/multigpu_logs/adapter.log) | [checkpoint-update-0030900.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/desi-ls-qlora-specz-e10/checkpoint-update-0030900.pt) |
| qwen-desi-ls-dora-specz-e10 | Partial: 29,900/37,410 updates | [log](/arc/home/gsm/aion_output/figures/qwen-desi-ls-dora-specz-e10/multigpu_logs/adapter.log) | [checkpoint-update-0029900.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/desi-ls-dora-specz-e10/checkpoint-update-0029900.pt) |
| qwen-galaxiesml-spec-qlora-specz-e10 | Completed; final result saved | [log](/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10/multigpu_logs/adapter.log) | [checkpoint-update-0014000.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/galaxiesml-spec-qlora-specz-e10/checkpoint-update-0014000.pt) |
| qwen-galaxiesml-spec-dora-specz-e10 | Completed; final result saved | [log](/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-dora-specz-e10/multigpu_logs/adapter.log) | [checkpoint-update-0014000.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/galaxiesml-spec-dora-specz-e10/checkpoint-update-0014000.pt) |
| qwen3.5-0.8b-galaxiesml-spec-qlora-specz-e10 | Completed; final result saved | [log](/arc/home/gsm/aion_output/figures/qwen3.5-0.8b-galaxiesml-spec-qlora-specz-e10/multigpu_logs/adapter.log) | [checkpoint-update-0014000.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/qwen3.5-0.8b-galaxiesml-spec-qlora-specz-e10/checkpoint-update-0014000.pt) |
| qwen3.5-0.8b-galaxiesml-spec-dora-specz-e10 | Completed; final result saved | [log](/arc/home/gsm/aion_output/figures/qwen3.5-0.8b-galaxiesml-spec-dora-specz-e10/multigpu_logs/adapter.log) | [checkpoint-update-0014000.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/qwen3.5-0.8b-galaxiesml-spec-dora-specz-e10/checkpoint-update-0014000.pt) |
| qwen3.5-0.8b-desi-ls-qlora-specz-e10 | Partial: 15,100/37,410 updates | [log](/arc/home/gsm/aion_output/figures/qwen3.5-0.8b-desi-ls-qlora-specz-e10/multigpu_logs/adapter.log) | [checkpoint-update-0015100.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/qwen3.5-0.8b-desi-ls-qlora-specz-e10/checkpoint-update-0015100.pt) |
| qwen3.5-0.8b-clauds-qlora-e10 | Partial: 21,600/37,500 updates | [log](/arc/home/gsm/aion_output/figures/qwen3.5-0.8b-clauds-qlora-e10/multigpu_logs/qlora.log) | [checkpoint-update-0021600.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/qlora_checkpoints/qwen3.5-0.8b-clauds-qlora-e10/checkpoint-update-0021600.pt) |
| qwen3.5-0.8b-clauds-dora-e10 | Partial: 15,100/37,500 updates | [log](/arc/home/gsm/aion_output/figures/qwen3.5-0.8b-clauds-dora-e10/dora.log) | [checkpoint-update-0015100.pt](/arc/projects/ots/Cosmic_Imprint_of_Time/dora_checkpoints/qwen3.5-0.8b-clauds-dora-e10/checkpoint-update-0015100.pt) |

Full output paths, final-result paths, log tails, checkpoint sizes and per-run session associations are in the evidence JSON. Frozen-head control results are separate from adapter completion. The 0.8B DESI-Legacy DoRA command remains in the launch notes, but no corresponding run output/checkpoint directory was found at the expected locations.

## Filter-aware warm-start sequence (second session)

| Arm | Last persisted state | Checkpoint / report |
|---|---|---|
| A / Level 0 | Complete: 2,806/2,806; selected update 2,500; CE 2.440432, NMAD 0.026268 | [latest.pt](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-warmstart/level0/latest.pt), [report](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-warmstart/level0/report.md) |
| B / Level 1 | Partial: 980/2,806; best update 750; CE 2.648250, NMAD 0.034537 | [latest.pt](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-warmstart/level1/latest.pt), [report](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-warmstart/level1/report.md) |
| C / Level 2 | Queued; no training checkpoint | [report](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-warmstart/level2/report.md) |

The pipeline manifest still says B is running and C is waiting. Recorded PIDs 483538 (supervisor), 483565 (A), and 568424 (B) are historical metadata, not verified live jobs. A finished at 2026-10-24 00:50:55; B’s report was last updated at 2026-10-24 06:01:51. Later user progress requests through October 25 survive, but their assistant replies were not recovered. [Pipeline log](/scratch/RedFM_MLP/filter-aware-runs/1023/hsc-warmstart/pipeline.log).

## Other recovered work

- [HSC QLoRA transfer sanity checks](/scratch/RedFM_MLP/fast-check-runs/1023/hsc-qlora-to-legacy-clauds/summary.json).
- [CLAUDS QLoRA transfer sanity checks](/scratch/RedFM_MLP/fast-check-runs/1023/clauds-qlora-to-hsc-legacy/summary.json).
- [Completed small filter-aware pilot](/scratch/RedFM_MLP/SFT-label_results_1023.md).
- [GLM initial capacity checks](/scratch/RedFM_MLP/glm52_test_1021.md).
- [GLM GalaxiesML-Spec capacity checks](/scratch/RedFM_MLP/glm52_test_1021spec.md).
- [GLM numerical sensitivity report](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/report.md).
- [GLM toy-pretrained versus random report](/arc/home/gsm/aion_output/figures/glm52-pretraining-transfer-galaxiesml-spec-1021/report.md).

GLM capacity results survive under `/arc/home/gsm/aion_output/figures/glm52-capacity-{1021,galaxiesml-spec-1021}/{overfit-n512,matched-numeric-mlp}/result.pt`.

This investigation did not resume jobs or change Codex indexes/databases. Checkpoints were checked for presence and nonzero size, not loaded or validated for restart compatibility.
