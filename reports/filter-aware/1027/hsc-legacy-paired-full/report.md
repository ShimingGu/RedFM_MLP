# Frozen same-galaxy HSC → Legacy benchmark

Five completed HSC-trained models were evaluated with both HSC and original Tractor observations of the **same galaxies**. No completed Legacy-trained adapter/head pair exists, so the reverse transfer cells are unavailable. No training, calibration or target-based checkpoint selection was performed.

On the independent selected P cohort, Legacy observations increase CE for **5/5 models**. For original QLoRA, NMAD changes from 0.0269 to 0.0663, and the catastrophic-outlier fraction from 1.03% to 7.12%. The comparisons below distinguish absolute predictive performance from the same-model observational transfer penalty.

## Cohort and independence

The complete raw audit found 134,533 HDF galaxies. The eligible paired pool contains **61,696 P**, **4,257 V**, and **16,600 T** objects. This execution evaluates **61,696 P** and **256 V** objects. P excludes training and validation exposure for every included model using TARGETID, HSC imaging identity and one-arcsecond spherical matches against HSC and DESI targeting coordinates. T is never scored. V is explicitly exploratory.

Selection: Ascending SHA256(1027:hsc_object_id); no z or predictions in sampling; n=0 means entire category. If a finite size limit was used, the numerical results describe that deterministic first-pass sample, **not every object in the eligible pool**. Whole-pool audit counts are provided separately. Every model/domain cell uses exactly the same selected IDs and labels.

Original HSC test objects were previously evaluated in completed training reports; now permitted for this predeclared frozen-model matched observation comparison, with no tuning or checkpoint selection. Not a never-inspected external sample.

P is independent of training and validation of the five completed HSC models only. Original held-out HSC test objects are allowed and explicitly flagged. Legacy incomplete models excluded; mutual HSC/Legacy-source independence not established.

### Sequential cuts

| Cut | Before | Removed | Remaining |
|---|---:|---:|---:|
| Exactly one original Tractor record per exact TARGETID | 134533 | 0 | 134533 |
| Unique primary DESI Iron spectrum passing original uniform galaxy quality and 0<=z<=4.5 | 134533 | 22531 | 112002 |
| Original Legacy survey restriction: main bright/dark (intersection of source protocols) | 112002 | 27277 | 84725 |
| Valid native HDF redshift and error | 84725 | 0 | 84725 |
| HDF/Iron label agreement to predeclared 1e-6 relative tolerance | 84725 | 2140 | 82585 |
| Original HSC minimum three valid bands | 82585 | 0 | 82585 |
| Original Legacy minimum three valid bands, with positive IVAR and NOBS | 82585 | 0 | 82585 |
| Verified DR9 filter provenance RELEASE in 9010/9011/9012 | 82585 | 0 | 82585 |
| Exact-ID pair positional verification within fixed one arcsecond | 82585 | 0 | 82585 |
| Unique HSC imaging identity across raw HDF rows | 82585 | 32 | 82553 |
| Unique Legacy imaging key among remaining pairs | 82553 | 0 | 82553 |
| Reject both members of ambiguous sub-arcsecond paired sky neighbours | 82553 | 0 | 82553 |

### Measurements and spectroscopy

HSC PDR2 cmodel AB magnitudes converted with original Jacobian; original DR9 Tractor signed fluxes, positive IVAR/NOBS; each uses its own EBV; no extra extinction correction. HDF fibermap targeting measurements were not assumed equivalent to the original Tractor VAC; the HDF also lacks NOBS. Native signed Legacy fluxes and all genuine missingness are retained. HSC i/y never occupy Legacy W1/W2 fields.

Spectroscopy uses the intersection of the archived source protocols: primary GALAXY/TGT, ZWARN=0, COADD_FIBERSTATUS=0, DELTACHI2>40, finite nonnegative ZERR, 0≤z≤4.5, main bright/dark. Label agreement was fixed at 1e−6×(1+|z_Iron|) before predictions. 41,965/84,725 checked labels are numerically identical; 82,585 pass the tighter agreement rule. All cells retain the same native HDF spectroscopic label. The HDF reduction tag is absent: numerical agreement does **not** establish identical DESI reductions.

The HDF lacks ZWARN, DELTACHI2, SPECTYPE, ZCAT_PRIMARY, SURVEY and PROGRAM. Its available OBJTYPE agrees for all 82,553 eligible pairs; its COADD_FIBERSTATUS differs for 400 pairs despite their DR1 Iron status passing the original source quality cut. These differences are preserved and reported, rather than claiming identical quality metadata. Selection uses the same Iron quality definition as the original source pipeline; no new HDF fiber-status cut was introduced. All 28,127 reconstructed HSC train/validation feature rows match the archived source arrays exactly. See `supplementary_audit.json` for quality flags, NOBS, MASKBITS and saved primary-provenance hashes.

## Frozen checkpoint inventory

| Model | Selected state | Source train / val / test | Native representation |
|---|---|---|---|
| original-qlora | epoch 9 (zero-based); minimum HSC val CE | 22444 / 5683 / 83853 | Level 0 |
| original-dora | epoch 9 (zero-based); minimum HSC val CE | 22444 / 5683 / 83853 | Level 0 |
| A | update 2500; minimum HSC val CE | 22444 / 5683 / 83853 | Level 0 |
| B | update 2500; minimum HSC val CE | 22444 / 5683 / 83853 | Level 1 |
| C | update 2806; minimum HSC val CE | 22444 / 5683 / 83853 | Level 2 |

All five source cohort membership sets agree; exact original QLoRA/DoRA validation/test IDs, matching preparation cache keys, and identical persisted A/B/C source arrays were verified. Exact ID lists are saved per model. Adapter/head paths, SHA256 hashes, configurations, selection rules and provenance are in `manifest.json`. Every adapter tensor was checked against its saved file, every source text replay checked against archived warm-start text, and a 32-object source-validation logit replay passed before paired inference. Checkpoint hashes were unchanged after inference.

## Matched transfer matrix: independent P

The point estimate is the median PDF-bin centre. CE is negative log probability **mass** in the spec-z bin. All grids were verified exactly equal: 300 equal-width bins over [0,4.5]. NMAD and median bias use the existing lower-median convention. CRPS and binned PIT use the existing evaluator; 68% coverage uses the bin-centre 16th–84th interval. These are frozen-checkpoint descriptive results.

| Model | n | HSC CE | Legacy CE | HSC NMAD | Legacy NMAD | HSC outliers | Legacy outliers |
|---|---:|---:|---:|---:|---:|---:|---:|
| original-qlora | 61696 | 2.4739 | 3.5830 | 0.0269 | 0.0663 | 1.03% | 7.12% |
| original-dora | 61696 | 2.4709 | 3.9302 | 0.0268 | 0.0838 | 1.04% | 11.79% |
| A | 61696 | 2.4622 | 3.5731 | 0.0265 | 0.0679 | 1.06% | 7.57% |
| B | 61696 | 2.5245 | 3.1430 | 0.0285 | 0.0483 | 1.15% | 2.00% |
| C | 61696 | 2.5672 | 3.0065 | 0.0304 | 0.0475 | 1.25% | 2.06% |
| Legacy QLoRA | — | unavailable | unavailable | — | — | — | — |
| Legacy DoRA | — | unavailable | unavailable | — | — | — | — |

### Paired transfer penalties (Legacy minus HSC)

| Model | ΔCE [95% CI] | ΔNMAD | Δoutlier fraction | Mean Δabs residual | Newly outlier / recovered |
|---|---|---:|---:|---:|---:|
| original-qlora | 1.1091 [1.0955, 1.1231] | 0.0394 | +6.09% | 0.0349 | 4033 / 278 |
| original-dora | 1.4593 [1.4441, 1.4743] | 0.0570 | +10.74% | 0.0467 | 6865 / 238 |
| A | 1.1109 [1.0974, 1.1246] | 0.0413 | +6.51% | 0.0361 | 4308 / 294 |
| B | 0.6186 [0.6069, 0.6300] | 0.0198 | +0.84% | 0.0137 | 836 / 316 |
| C | 0.4394 [0.4307, 0.4483] | 0.0171 | +0.81% | 0.0127 | 860 / 361 |

1000 paired galaxy bootstrap draws, percentile 95%; conditional on these checkpoints; excludes training-seed uncertainty. All source, target and delta metric intervals, biases, CRPS, coverage, PIT, outlier transition counts and per-galaxy residual changes are in `results.json` and `*_paired_differences.csv`. The same paired bootstrap indices are used in every within-model contrast and all four cells of a cross-model penalty contrast.

## A/B/C and adaptation comparisons

| Penalty contrast | ΔΔCE [95% CI] | ΔΔNMAD [95% CI] |
|---|---|---|
| B_minus_A | -0.4923 [-0.5043, -0.4787] | -0.0216 [-0.0223, -0.0209] |
| C_minus_B | -0.1792 [-0.1894, -0.1691] | -0.0026 [-0.0032, -0.0021] |
| original-dora_minus_original-qlora | 0.3502 [0.3392, 0.3612] | 0.0176 [0.0167, 0.0185] |

Positive values indicate the second model has a larger observational transfer penalty. Absolute HSC and Legacy scores must also be considered: a smaller penalty can arise from worse in-domain predictions. C had worse archived HSC validation performance than B. These matched comparisons cannot establish that wavelength metadata were interpreted physically.

## Redshift dependence

Bins were fixed before inference from the spectroscopic distribution: {'initial_edges': [0.0, 0.2, 0.4, 0.6, 1.0, 2.0, 4.5], 'initial_counts': [23141, 16094, 14551, 7510, 400, 0], 'edges': [0.0, 0.2, 0.4, 0.6, 1.0, 4.5], 'counts': [23141, 16094, 14551, 7510, 400], 'minimum': 40, 'merge_rule': 'Accumulate adjacent bins left-to-right until >=40; merge final small tail backward', 'standardization': 'Equal weight for each nonempty merged spec-z bin; macro-average within-bin metrics, including NMAD (not pooled weighted NMAD)'}. All bins retain the same objects in both input conditions. Standardized results equally weight the merged nonempty bins, rather than changing the galaxy selection. Macro-averaged within-bin NMAD is explicitly not a pooled NMAD.

| Model | Spec-z range | n | HSC NMAD | Legacy NMAD | HSC outlier | Legacy outlier | ΔCE [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| original-qlora | 0–0.2 | 23141 | 0.0283 | 0.0695 | 0.48% | 10.87% | 1.1123 [1.0871, 1.1354] |
| original-qlora | 0.2–0.4 | 16094 | 0.0303 | 0.0755 | 0.89% | 7.58% | 1.4852 [1.4535, 1.5156] |
| original-qlora | 0.4–0.6 | 14551 | 0.0192 | 0.0406 | 0.96% | 0.19% | 0.7772 [0.7537, 0.8006] |
| original-qlora | 0.6–1 | 7510 | 0.0270 | 0.0453 | 1.12% | 4.26% | 0.9030 [0.8705, 0.9333] |
| original-qlora | 1–4.5 | 400 | 0.0765 | 0.0923 | 39.75% | 77.25% | 1.7316 [1.5016, 1.9828] |
| original-dora | 0–0.2 | 23141 | 0.0282 | 0.0468 | 0.48% | 5.65% | 0.9266 [0.9078, 0.9455] |
| original-dora | 0.2–0.4 | 16094 | 0.0304 | 0.1611 | 0.89% | 33.17% | 2.1966 [2.1655, 2.2248] |
| original-dora | 0.4–0.6 | 14551 | 0.0191 | 0.0459 | 1.02% | 2.48% | 2.1932 [2.1565, 2.2270] |
| original-dora | 0.6–1 | 7510 | 0.0269 | 0.0406 | 1.03% | 0.73% | 0.1618 [0.1388, 0.1811] |
| original-dora | 1–4.5 | 400 | 0.0775 | 0.0830 | 40.75% | 52.50% | 0.2713 [0.0807, 0.4527] |
| A | 0–0.2 | 23141 | 0.0277 | 0.0711 | 0.52% | 12.18% | 1.1274 [1.1034, 1.1510] |
| A | 0.2–0.4 | 16094 | 0.0299 | 0.0713 | 0.93% | 7.88% | 1.5224 [1.4934, 1.5513] |
| A | 0.4–0.6 | 14551 | 0.0187 | 0.0403 | 1.01% | 0.24% | 0.7951 [0.7716, 0.8184] |
| A | 0.6–1 | 7510 | 0.0271 | 0.0464 | 1.03% | 3.37% | 0.7817 [0.7514, 0.8102] |
| A | 1–4.5 | 400 | 0.0743 | 0.0913 | 40.00% | 73.25% | 1.2749 [1.0596, 1.5071] |
| B | 0–0.2 | 23141 | 0.0292 | 0.0391 | 0.41% | 1.20% | 0.4233 [0.4054, 0.4405] |
| B | 0.2–0.4 | 16094 | 0.0323 | 0.0499 | 1.02% | 1.06% | 0.4027 [0.3843, 0.4210] |
| B | 0.4–0.6 | 14551 | 0.0208 | 0.0321 | 1.11% | 0.39% | 0.4273 [0.4068, 0.4467] |
| B | 0.6–1 | 7510 | 0.0302 | 0.0384 | 1.69% | 5.38% | 1.9794 [1.9376, 2.0196] |
| B | 1–4.5 | 400 | 0.0744 | 0.0860 | 41.00% | 80.50% | 2.0140 [1.7943, 2.2285] |
| C | 0–0.2 | 23141 | 0.0287 | 0.0398 | 0.34% | 0.92% | 0.3095 [0.2954, 0.3232] |
| C | 0.2–0.4 | 16094 | 0.0319 | 0.0720 | 1.07% | 2.11% | 0.7551 [0.7356, 0.7729] |
| C | 0.4–0.6 | 14551 | 0.0261 | 0.0342 | 1.68% | 0.53% | 0.1775 [0.1608, 0.1931] |
| C | 0.6–1 | 7510 | 0.0317 | 0.0413 | 1.56% | 4.69% | 0.6416 [0.6200, 0.6650] |
| C | 1–4.5 | 400 | 0.0746 | 0.1275 | 40.00% | 72.50% | 0.9790 [0.8058, 1.1487] |

| Model | Equal-bin HSC CE | Equal-bin Legacy CE | Equal-bin ΔCE [95% CI] |
|---|---:|---:|---|
| original-qlora | 3.2929 | 4.4948 | 1.2019 [1.1543, 1.2518] |
| original-dora | 3.2915 | 4.4414 | 1.1499 [1.1082, 1.1906] |
| A | 3.2973 | 4.3976 | 1.1003 [1.0541, 1.1465] |
| B | 3.3299 | 4.3792 | 1.0493 [1.0054, 1.0944] |
| C | 3.3424 | 3.9149 | 0.5725 [0.5367, 0.6105] |

Per-bin bias, median bias, CRPS, coverage and all transfer deltas are saved in `results.json`. This is redshift-stratified transfer, not strict high-z extrapolation: the source training includes these redshifts.

## Photometric support

Both HSC and Legacy must have valid common g/r/z and all 11 transformed coordinates within HSC training marginal 0.5–99.5 percentiles. HSC source training photometry only; no target labels. The transform uses source-only median g/r/z flux errors as asinh softening scales, log errors, asinh SNRs, and two asinh colours. It accepts signed target fluxes and requires genuine common-band errors. No target redshift enters the estimator.

| Model | P subset | n | ΔCE | Mean Δabs residual | Legacy outlier fraction |
|---|---|---:|---:|---:|---:|
| original-qlora | full | 61696 | 1.1091 | 0.0349 | 7.12% |
| original-qlora | support | 33935 | 1.3762 | 0.0415 | 8.83% |
| original-qlora | outside_support | 27761 | 0.7826 | 0.0268 | 5.03% |
| original-dora | full | 61696 | 1.4593 | 0.0467 | 11.79% |
| original-dora | support | 33935 | 1.7060 | 0.0555 | 15.61% |
| original-dora | outside_support | 27761 | 1.1577 | 0.0361 | 7.11% |
| A | full | 61696 | 1.1109 | 0.0361 | 7.57% |
| A | support | 33935 | 1.3790 | 0.0437 | 9.78% |
| A | outside_support | 27761 | 0.7832 | 0.0269 | 4.86% |
| B | full | 61696 | 0.6186 | 0.0137 | 2.00% |
| B | support | 33935 | 0.6662 | 0.0144 | 1.59% |
| B | outside_support | 27761 | 0.5604 | 0.0130 | 2.49% |
| C | full | 61696 | 0.4394 | 0.0127 | 2.06% |
| C | support | 33935 | 0.4468 | 0.0126 | 1.77% |
| C | outside_support | 27761 | 0.4303 | 0.0127 | 2.41% |

The full selected P sample remains the main reference. `photometric_distributions.json` provides source-training, full-eligible-pool and evaluated-P flux, error, signed SNR, positive-flux AB magnitude, nominal 5σ depth, g−r/r−z colour, missingness-pattern and EBV distributions. Colours/magnitudes are undefined for nonpositive fluxes; those measurements remain valid for inference. The support subset is a diagnostic fixed independently of target labels; it is a conservative marginal-box rule and does not establish equality of filter transmission, depth, calibration, or measurement pipelines. A paired observation comparison controls the evaluated physical galaxies and spec-z distribution, but does not causally isolate bandpass effects.

## Validation-exposed exploratory cohort

| Model | n | HSC CE | Legacy CE | ΔCE |
|---|---:|---:|---:|---:|
| original-qlora | 256 | 2.4896 | 3.5642 | 1.0746 |
| original-dora | 256 | 2.4929 | 3.8670 | 1.3742 |
| A | 256 | 2.4728 | 3.5449 | 1.0721 |
| B | 256 | 2.5289 | 3.1627 | 0.6338 |
| C | 256 | 2.5685 | 2.9343 | 0.3658 |

V is not independent evidence because its HSC labels were used in checkpoint selection. Its full stratified/support results are separately tagged in `results.json`.

## Figures

![Matched predictions](figures/matched_predictions.png)

![Redshift distributions and PIT](figures/redshift_and_pit.png)

![Transfer by redshift and support](figures/transfer_by_redshift_and_support.png)

![Common optical photometry](figures/common_optical_photometry.png)

## Reproduction and remaining bidirectional gap

Persistent outputs: `/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full`. The compact report is mirrored under `reports/filter-aware/1027/hsc-legacy-paired-full/`. `audit.json` contains raw fingerprints, filtering counts, numerical label comparisons, per-band HDF/Tractor comparison, RELEASE counts and all per-model membership counts. `eligible_pairs.npz` preserves the full audited pool; `cohort.npz` preserves exact evaluated IDs, both measurement matrices, masks, labels and exposure. `inputs/` stores every exact serialized record. `evaluation_policy.json` was written before predictions and its hash was verified by every worker. `manifest.json` contains exact checkpoints, hashes and compatible PDF grids. Prediction chunks are atomic and resumable.

```bash
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/audit_hsc_legacy_pairing.py --output-dir /arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full/raw-audit
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/hsc_legacy_paired_benchmark.py prepare --output-dir /arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full --n-primary 0 --n-validation 256
# For each of original-qlora original-dora A B C, select one actually idle MIG slice:
CUDA_VISIBLE_DEVICES=<idle-MIG-UUID> PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/hsc_legacy_paired_benchmark.py run --output-dir /arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full --model <model>
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/audit_hsc_legacy_measurements.py --output-dir /arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/hsc_legacy_paired_benchmark.py report --output-dir /arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1027/hsc-legacy-paired-full
```

Do not rerun prepare over an existing manifest. Frozen inference reuses verified completed chunks. The original Legacy runs currently have no completed adapter/head; their removed sampled-cohort cache prevents reconstructing exact source exposure from preparation metadata alone. To complete a mutually independent bidirectional benchmark: (1) recover or deterministically reconstruct the original Legacy sampled cohort and train/val/test IDs using the archived preparation code/configuration and raw input order, verifying the saved cache key and split counts; (2) obtain genuinely completed compatible Legacy-trained adapter/head states selected solely on source validation (or separately authorize any future training); (3) repeat identifier/group/position exposure OR across both source cohorts, freeze the mutually clean P intersection and the evaluation policy, then evaluate all four cells on exactly those IDs. If that intersection is too small, reserve a genuinely new matched catalogue outside both source exposure sets before any new training. No training was launched here.
