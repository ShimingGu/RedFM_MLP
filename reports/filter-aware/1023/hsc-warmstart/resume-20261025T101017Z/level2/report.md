# Experiment C — warm-start Level-2 filter physics

Status: **queued**. Updated 2026-10-23T09:41:50.542768+00:00.

C is an independent branch from the original archived HSC Qwen3.5-4B-Base rank-8 QLoRA adapter and matching trained 256-hidden-unit PDF head. Both components are verified tensor by tensor when GPU execution begins. It uses the same 22,444 training and 5,683 validation objects, labels, numerical formatting, missingness and 300-bin 0–4.5 grid as A/B. Removing only `pivot_nm` and `width_nm` from every source and target string reproduces B byte for byte.

Protocol matches A/B: at most two additional epochs / 2,806 updates; no head-only warmup; adapter LR 3e-6; head LR 1e-5; effective batch 16; seed 1023; unchanged trainable modules, new AdamW moments, constant LRs, clipping and weight decay. Source-validation CE selects among update zero and validations every 250 updates plus epoch ends. No held-out test evaluation or target-label fitting/calibration/selection.

Actual tokenizer audit: source maximum 272 tokens; Legacy 276; CLAUDS 462. All inputs fit the common 512-token limit without truncation.

## Filter metadata and approximations

Pinned response profiles and original-file hashes are verified. Pivot wavelength is sqrt(∫Rλ dλ / ∫R/λ dλ); rectangular equivalent bandwidth is ∫R dλ / max(R), not FWHM. Both are in nm, serialized to three decimal places. R is the supplied relative photon-counting system response; no extra wavelength factor, SED weights, learned filter embeddings, full transmission curves or additional descriptors enter the prompts. The integrals are reproduced from the vendored profiles in `registry_audit.json`.

Source HSC grizy uses nominal HSC 2017 profiles. Exposure mixtures of r/r2 and i/i2 are unavailable in the feature cache, so r and i remain representative approximations; no per-object mixture is invented. Legacy RELEASE 9010/9012 uses representative DECam DR1 grz responses, 9011 uses BASS gr and MzLS z, and all use WISE W1/W2. CLAUDS uses MegaCam u/u_star (MP9302/MP9301), HSC grizy and VIRCAM YJHKs. These nominal curves are not object-specific throughputs. Other registry approximations are preserved in `registry_audit.json` and `filter_registry_snapshot.json`.

Queued: Managed sequential pipeline: waiting for all earlier arms to complete successfully. GPU `MIG-a9b1fb4a-5c28-5b03-9686-f831c6facf25`. Existing jobs retain their reservations; C waits without a CUDA context.

Zero-update source diagnostics and adaptation results are pending; no performance penalty or benefit is presumed.

Target inference remains pending until source-only checkpoint selection finishes.

## Interpretation limits

The critical comparison is C versus B, since B already supplies explicit band identities. Source validation also selects checkpoints, so these are model-selection results. Retaining the zero-update candidate guarantees non-increasing CE only relative to Level 2, not the original Level 0; latest validation and last/selected parameter changes are saved. Single-seed point estimates on fixed 256-object target samples do not establish statistical significance.

Every HSC training galaxy uses essentially the same fixed filter set: wavelength and bandwidth do not vary independently across source objects. Successful source adaptation therefore does not demonstrate learning a physically transferable response dependence. Any transfer improvement is preliminary; multiple training filter systems and shuffled or incorrect metadata controls would be needed. This experiment does not demonstrate universal filter-aware generalisation.
