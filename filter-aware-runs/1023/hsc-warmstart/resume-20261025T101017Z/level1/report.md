# Experiment B — warm-start Level-1 band-aware QLoRA

Status: **training**. Updated 2026-10-24T06:01:51.472354+00:00.

The run starts from the original completed HSC adapter and trained PDF head, independently of A's continued weights. Restoration of both components is verified tensor by tensor when GPU execution starts. The exact same 22,444 training and 5,683 validation objects, native values/errors/masks, five-decimal formatting and 300-bin 0–4.5 PDF grid are retained. Only the serializer changes to explicit survey and physical-band identities. No wavelengths or widths are added.

Training matches A: at most two epochs / 2,806 updates, no head-only warmup, adapter LR 3e-6, head LR 1e-5, effective batch 16, seed 1023, constant learning rates, new AdamW moments, same rank/modules/weight decay/clipping. Source-validation CE selects among update zero and validations every 250 updates plus epoch ends. Test objects are excluded. Target labels enter inference metrics only.

Actual tokenizer audit: 28,127 source texts, maximum 174 / 512 tokens, zero truncations. Exact strings and token examples are saved. Target token audits are in the manifest.

## Serializer-only change and source adaptation

The two zero-update rows use identical original trained weights and the full identical source-validation cohort. Their difference isolates the input-format change.

| Model / evaluation | CE | NMAD | Outliers | Bias | CRPS | 68% coverage |
|---|---:|---:|---:|---:|---:|---:|
| Original weights / Level 0 | 2.453027 | 0.026762 | 1.073% | 0.000700 | 0.027404 | 73.025% |
| Original weights / Level 1 | 6.668456 | 0.153247 | 47.369% | -0.157844 | 0.220149 | 18.916% |
| Selected Level 1 / update 750 | 2.648250 | 0.034537 | 1.355% | -0.002730 | 0.033085 | 71.494% |
| Latest Level-1 validation | 2.648250 | 0.034537 | 1.355% | -0.002730 | 0.033085 | 71.494% |
| Matched Level 0 A / update 2500 | 2.440432 | 0.026268 | 1.073% | 0.001061 | 0.027190 | 73.500% |

Serializer-only delta CE: +4.215429; delta NMAD: +0.126485. Changes are measured rather than presumed to be damage. A separate 32-object canonical Level-0 replay verifies restoration before this diagnostic.

Completed 980/2,806 optimizer updates. Best candidate: update 750; elapsed training/periodic-validation time 5.10 hours.

Fixed-weight target evaluation remains pending until source checkpoint selection finishes.

## Additional evaluation GPUs

Training uses `MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25`. Validation and selected-model transfer inference are split over `MIG-bef01423-2506-5edb-a289-80c60b1f0297,MIG-412ff209-a29c-5bc8-9aa2-fd1a41474aec` using two independent frozen replicas, with synchronized adapter/head weights and the original batch-of-eight grouping and object order. Training, effective batch 16, optimizer, RNG, source selection and independent original-checkpoint initialization are unchanged. A → B → C executes sequentially. This accelerates evaluation, not optimizer updates. `evaluation_acceleration.json` records per-evaluation timings and both workers' peak allocated memory; the original training memory figure covers the training GPU only. `execution_backend.json` records code and devices.
