# Experiment C — warm-start Level-2 filter physics

Status: **training**. Updated 2026-10-26T00:43:25.051371+00:00.

C is an independent branch from the original archived HSC Qwen3.5-4B-Base rank-8 QLoRA adapter and matching trained 256-hidden-unit PDF head. Both components are verified tensor by tensor when GPU execution begins. It uses the same 22,444 training and 5,683 validation objects, labels, numerical formatting, missingness and 300-bin 0–4.5 grid as A/B. Removing only `pivot_nm` and `width_nm` from every source and target string reproduces B byte for byte.

Protocol matches A/B: at most two additional epochs / 2,806 updates; no head-only warmup; adapter LR 3e-6; head LR 1e-5; effective batch 16; seed 1023; unchanged trainable modules, new AdamW moments, constant LRs, clipping and weight decay. Source-validation CE selects among update zero and validations every 250 updates plus epoch ends. No held-out test evaluation or target-label fitting/calibration/selection.

Actual tokenizer audit: source maximum 272 tokens; Legacy 276; CLAUDS 462. All inputs fit the common 512-token limit without truncation.

## Filter metadata and approximations

Pinned response profiles and original-file hashes are verified. Pivot wavelength is sqrt(∫Rλ dλ / ∫R/λ dλ); rectangular equivalent bandwidth is ∫R dλ / max(R), not FWHM. Both are in nm, serialized to three decimal places. R is the supplied relative photon-counting system response; no extra wavelength factor, SED weights, learned filter embeddings, full transmission curves or additional descriptors enter the prompts. The integrals are reproduced from the vendored profiles in `registry_audit.json`.

Source HSC grizy uses nominal HSC 2017 profiles. Exposure mixtures of r/r2 and i/i2 are unavailable in the feature cache, so r and i remain representative approximations; no per-object mixture is invented. Legacy RELEASE 9010/9012 uses representative DECam DR1 grz responses, 9011 uses BASS gr and MzLS z, and all use WISE W1/W2. CLAUDS uses MegaCam u/u_star (MP9302/MP9301), HSC grizy and VIRCAM YJHKs. These nominal curves are not object-specific throughputs. Other registry approximations are preserved in `registry_audit.json` and `filter_registry_snapshot.json`.

## Source serializer change and recovery

| Model / evaluation | CE | NMAD | Outliers | Bias | CRPS | 68% coverage |
|---|---:|---:|---:|---:|---:|---:|
| Original archived weights / Level 0 | 2.453027 | 0.026762 | 1.073% | 0.000700 | 0.027404 | 73.025% |
| Original archived weights / Level 2 | 6.795675 | 0.152845 | 47.545% | -0.158619 | 0.221691 | 17.720% |
| Original archived weights / Level 1 | 6.668456 | 0.153247 | 47.369% | -0.157844 | 0.220149 | 18.916% |
| Selected Level 0 A / update 2500 | 2.440432 | 0.026268 | 1.073% | 0.001061 | 0.027190 | 73.500% |
| Selected Level 1 B / update 2500 | 2.507556 | 0.028672 | 1.056% | -0.000350 | 0.029106 | 72.462% |
| Selected Level 2 C / update 750 | 2.705865 | 0.037351 | 1.548% | -0.005627 | 0.035589 | 69.488% |
| Latest Level-2 validation | 2.705865 | 0.037351 | 1.548% | -0.005627 | 0.035589 | 69.488% |

The zero-update rows use identical original weights and objects; their differences measure serializer-only changes. Recovery through SFT is measured separately. A 32-object canonical Level-0 replay checks restoration before the full 5,683-object Level-2 diagnostic.

Training: 900/2,806 updates; selected update 750; training/periodic-validation wall time 4.957 hours; peak allocated GPU memory 5.452 GiB.

Target inference remains pending until source-only checkpoint selection finishes.

## Interpretation limits

The critical comparison is C versus B, since B already supplies explicit band identities. Source validation also selects checkpoints, so these are model-selection results. Retaining the zero-update candidate guarantees non-increasing CE only relative to Level 2, not the original Level 0; latest validation and last/selected parameter changes are saved. Single-seed point estimates on fixed 256-object target samples do not establish statistical significance.

Every HSC training galaxy uses essentially the same fixed filter set: wavelength and bandwidth do not vary independently across source objects. Successful source adaptation therefore does not demonstrate learning a physically transferable response dependence. Any transfer improvement is preliminary; multiple training filter systems and shuffled or incorrect metadata controls would be needed. This experiment does not demonstrate universal filter-aware generalisation.

## Additional evaluation GPUs

Training uses `MIG-1bba1f26-8b2b-5b97-9bb1-c886ab9bb8b7`. Validation and selected-model transfer inference are split over `MIG-0a2e21a6-8961-56b1-9eb4-997ae914386e,MIG-6c85a6c1-35aa-5ff1-ae36-a468e3e230b6` using two independent frozen replicas, with synchronized adapter/head weights and the original batch-of-eight grouping and object order. Training, effective batch 16, optimizer, RNG, source selection and independent original-checkpoint initialization are unchanged. A → B → C executes sequentially. This accelerates evaluation, not optimizer updates. `evaluation_acceleration.json` records per-evaluation timings and both workers' peak allocated memory; the original training memory figure covers the training GPU only. `execution_backend.json` records code and devices.
