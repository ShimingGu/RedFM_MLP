# Legacy checkpoint audit and matched-training package — 2026-10-28

**Audit and preparation are complete. No full Qwen training or full-cohort reverse inference was launched.** Both original Legacy models are recovered and replay exactly. The preferred shared split is valid, and the new four-model design is ready for a separately authorised one-seed pilot. Three-seed configurations are preserved.

## Final decision answers

1. **Completed Legacy checkpoints:** both selected adapter/head pairs exist in their actual original output directories, listed in Section A. Both load and reproduce 64 original source-validation objects exactly. Their selected epoch is the fixed final epoch in these particular runs; the rolling restart files are earlier states.
2. **Mutual independence:** **61,310** of the existing 61,696 P galaxies survive all seven historical models’ training/validation exposure. The 386 exclusions comprise 317 training-exposed and 69 validation-exposed galaxies; two training exposures require imaging/position matching beyond exact TARGETID.
3. **Matched split:** the proposed **16,600 train / 4,257 validation / 61,696 evaluation** assignment passes identifier, both imaging-identity and cross-representation one-arcsecond positional checks with **zero** cross-split candidates. No reassignment is necessary.
4. **Initial states and inputs:** all branches use the same verified pre-adaptation Qwen3.5-4B-Base and a common Level-1 template. HSC and Legacy branches have exactly identical trainable initial states for each method/seed; all four branches share the same initial head for a seed. QLoRA/DoRA have different trainable structures, so full cross-method state/parameter equivalence is not claimed.
5. **Minimum pilot:** four fresh models, seed 42, ten fixed epochs (three head-only plus seven joint), 10,380 optimizer updates and 166,000 training presentations each. Historical native-input timing proxies plus measured Level-1 validation forwards give **182.2 MIG-GPU hours** for training/validation and **11.9** more for selected-checkpoint paired evaluation.
6. **Three-seed confirmation:** seeds 42/43/44 use identical memberships and matched seed assignments. Total training/validation proxy **546.5 MIG-GPU hours**, an additional **364.3** beyond the pilot; selected-state evaluation adds **35.8** over all three seeds.
7. **Launch readiness:** checkpoint discovery, exposure recovery, split integrity, serialization and fresh-source initialization are resolved. Code, configurations, tests and commands are prepared. Full training still needs the separate go-ahead explicitly required by `task_1028.md`. Fresh Level-1 convergence, full backward-pass memory and exact training throughput remain unmeasured; these are pilot questions, not silently claimed verified properties. Expansion must depend on source-validation viability and compute budget before locked-P scoring, never which transfer result is favourable.

## A — Completed Legacy-source checkpoint audit

| Model | Checkpoint found | Adapter/head verified | Source IDs recovered | Validation replay | Ready for matched inference |
|---|---|---|---|---|---|
| Legacy QLoRA | Yes | Exact tensor/file checks | Exact deterministic reconstruction | 64 objects, exact logits/PDFs | Yes, on P_mutual |
| Legacy DoRA | Yes | Exact tensor/file checks | Exact deterministic reconstruction | 64 objects, exact logits/PDFs | Yes, on P_mutual |

Actual selected-checkpoint directories:

```text
/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/qlora
/arc/home/gsm/aion_output/figures/qwen-desi-ls-dora-specz-e10/dora
```

Each contains `adapter/adapter_model.safetensors`, `adapter/adapter_config.json`, and `photoz_head.pt`; the corresponding `result.pt` contains the original validation/test IDs, predictions and ten-epoch history. Exact SHA256 identities are in `legacy_checkpoint_audit/verified_inventory.json` and the package manifest. QLoRA exported at 2026-10-27 07:14 UTC; DoRA at 12:44 UTC, after the earlier initial discovery pass. Original benchmark/counterfactual manifests remain unmodified historical snapshots.

Both source runs use 59,855 / 15,053 / 225,080 train/validation/test galaxies and the same cache key. The completed adapters are selected by minimum own-source validation CE, attained at zero-based epoch 9 in both runs. Therefore the exported selected weights equal the final 37,410-update budget weights for these runs. There is no separately exported final adapter. Do not confuse these with the last rolling update-37,400 restart checkpoint, or its stored best state from epoch 8.

QLoRA’s selected validation CE is 2.981402; DoRA’s is 2.973850. No new full validation or test inference was needed for this audit. Limited replay uses the first 64 archived source-validation records in original order and their original native serializer. Pooling, 512-token limit, NF4 double quantization, bf16 compute, missingness and the shared 300-bin [0,4.5] probability-mass grid reproduce exactly.

### Provenance and restart recovery

Original cache key: `86208709213f26c0bf44aa8c19278ba2e6a55a4203f615b09667ae0a70af3030`. The archived catalogue-loader code SHA256 is `dbc6418296f262fd8dfc035b8729bcf42927d7543310c27e0f6d2ef963bfd001`, matching Git revision `f53a211c65efa2f1737e3675bb42a359f21a02af`. Current raw-file sizes, mtimes and exact sorted file ordering reproduce the original preparation cache identity. The lost scratch cache was not treated as proof that training failed.

The recovery replays the original NumPy PCG64 seed-42 priority stream over 12,047,393 quality-eligible Tractor rows, keeps the lowest 300,000 priorities, removes the original 12 duplicate imaging groups, and applies the original SHA256(seed:imaging-group) split. Only bookkeeping fields—RA/DEC and exact raw file/row references—were added to sampled records; the RNG, feature arithmetic, sampling, deduplication, ordering and split algorithm are unchanged. A CPU test independently verifies bookkeeping does not change sampler output.

Both models’ entire ordered 15,053-object validation and 225,080-object test TARGETID sequences and spec-z arrays match the reconstruction exactly. Thus training identity is recovered from verified historical logic and independent archived evidence, rather than invented with a fresh split. `recovered_cohort/cohort.npz` also preserves original features, groups, coordinates, source file indices and source row indices. The original source-validation predictions then provide a separate feature/serializer/checkpoint replay check.

Recorded restarts are `['resuming QLoRA from /arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/desi-ls-qlora-specz-e10/checkpoint-update-0030900.pt at update 30,900/37,410']` and `['resuming DoRA from /arc/projects/ots/Cosmic_Imprint_of_Time/qwen_survey_checkpoints/desi-ls-dora-specz-e10/checkpoint-update-0029900.pt at update 29,900/37,410']`. They restore the same run’s trainable weights, optimizer, scheduler and RNG under the original run contract; no HSC-adapted initialization is used in the original Legacy training code. Full source-history records and distinct restart-state tensor hashes are preserved in the inventory.

### Exposure of the existing paired benchmark

| Exposure to either identical Legacy source cohort | Existing P objects |
|---|---:|
| Training: exact TARGETID | 315 |
| Training: union of TARGETID, imaging identity and positions | 317 |
| Source validation/checkpoint selection | 69 |
| Surviving P_mutual | 61310 |

Every historical HSC QLoRA/DoRA/A/B/C membership flag is preserved alongside separate Legacy QLoRA/DoRA train/validation/test flags. Exact TARGETIDs, Legacy release/brick/object keys, HSC imaging IDs where cross-identified, and spherical matches against both observations at one arcsecond are combined. Held-out test exposure is recorded separately; previous test evaluation is not described as a never-inspected blind sample.

Eight runnable original-model configurations cover HSC/Legacy source × QLoRA/DoRA × HSC/Legacy observations on exactly P_mutual. None was executed. Historical HSC rows can additionally be reused by exact ID from their archived predictions after checking model/input identity. These old models have unequal original source cohorts, serializers and budgets; their results are secondary diagnostics, not the matched-training causal comparison.

## B — Matched-data feasibility

The full **82,553**-object paired pool was checked again against raw-metadata extracts. HSC features/validity match exactly; Legacy features reconstruct exactly from original Tractor signed flux, IVAR, NOBS and its own EBV. Imaging identities and native HDF labels match exactly. Every row passes the original shared DR1 Iron spectral-quality definition and the fixed native/Iron label agreement rule. The maximum HSC/Legacy separation is 0.9967 arcsec. No image or spectral pixels were read.

The shared quality definition is primary GALAXY/TGT, ZWARN=0, COADD_FIBERSTATUS=0, DELTACHI2>40, finite nonnegative ZERR, 0≤z≤4.5, main bright/dark, at least three usable bands per system, known DR9 release, and the existing 1e−6×(1+|z_Iron|) label-consistency cut. HDF-specific fiber-status discrepancies remain documented under the established Iron policy; no new HDF quality cut or membership selection was introduced.

| New-model role | Historical category | Objects | z-bin populations: 0–.2 / .2–.4 / .4–.6 / .6–1 / 1–4.5 |
|---|---|---:|---|
| train | T | 16600 | 6199 / 4290 / 3925 / 2078 / 108 |
| validation | V | 4257 | 1621 / 1111 / 1013 / 492 / 20 |
| evaluation | P | 61696 | 23141 / 16094 / 14551 / 7510 / 400 |

No distinct paired objects share an identifier/imaging identity or fall within one arcsecond in HSC/HSC, Legacy/Legacy or cross-representation positions. This stronger geometry audit found zero candidates, including across split boundaries. The proposed assignment is therefore usable without moving or dropping any P object. Historical T/V are just fixed allocations for fresh-base models; old adapted weights are not inherited.

The new-model evaluation retains all 61,696 P objects, even though 386 are excluded from historical-model P_mutual: those historical Legacy exposures do not contaminate a new model trained from the original base exclusively on T. The fixed P set is a **locked, previously analysed paired benchmark**, not a prospective blind test. No additional eligible uninspected pool is established within the current audited files under unchanged cuts. A new externally obtained, identifier/position-disjoint paired catalogue reserved before training would provide a separate prospective replication; excluded or ambiguous rows are not silently promoted.

Split membership manifests preserve TARGETID, both imaging identities, original row and shared spec-z. Full redshift and source-specific flux, error, signed SNR, g−r/r−z colour, apparent AB magnitude, nominal 5σ depth, missingness-pattern and EBV distributions are in `paired_split/photometric_distributions.json`. Nonpositive fluxes remain valid measurements; magnitudes/colours are not invented for them.

## C — Matched-training design

| Training observations | QLoRA | DoRA |
|---|---|---|
| HSC grizy | H-Q | H-D |
| Original Tractor grzW1W2 | L-Q | L-D |

All four start from the same SHA256-verified pre-adaptation `/arc/home/gsm/hf_models/Qwen3.5-4B-Base`. No source-trained adapter is accepted by the fresh model factory. A new CPU-initialized head is saved once per seed and explicitly restored into every branch. It is never an old HSC or Legacy trained head. The three seeds are 42, 43 and 44; all twelve model constructions were checked with forward passes only, using source-training records and zero optimizer updates.

Within each method/seed the two source branches have identical full relevant trainable-state hashes. Their common head has 799,020 parameters; QLoRA adds 1,572,864 adapter parameters and DoRA adds 1,675,264. The additional **102,400** DoRA magnitude parameters are explicitly counted, so equal rank is not claimed to be equal parameter budget.

Common architecture/settings: last valid-token pooling; 2560→256→256→300 PDF head with LayerNorm/GELU and 0.1 head dropout; 300 mass bins over [0,4.5]; NF4 double quantization with bf16 compute; rank 8, alpha 16, LoRA dropout 0.05, q_proj/k_proj/v_proj/o_proj targets. The pre-existing model builder provides gradient checkpointing and proven QLoRA/DoRA support; the new loop adds explicit common heads, exact partial-window averaging, matched object order, two rotating resume slots and separate selected/final exports.

### Shared Level-1 numerical contract

Common existing Level-1 template, identical field structure; real survey, cmodel/Tractor, filter/instrument identifiers and EBV provenance differ. Header, five bands in fixed increasing nominal wavelength order for each true system, each with flux/sigma, then EBV. Flux/error precision is five decimals; missing is `NA`. HSC catalogued cmodel AB magnitudes use the original nanomaggy/Jacobian conversion, while Legacy uses the original signed Tractor flux and positive IVAR/NOBS rule. Both retain genuine EBV provenance without an extra extinction correction. Real HSC, DECam, BASS, MzLS and WISE identities remain explicit. **No pivot wavelength or width is added.**

| Domain | Records audited | Token median | Token p95 | Token max | Truncated |
|---|---:|---:|---:|---:|---:|
| hsc | 82553 | 163 | 168 | 174 | 0 |
| legacy | 82553 | 165 | 169 | 175 | 0 |

### Budget, scheduling, selection and seeds

The minimum pilot is four models at seed 42, each with 16,600 training examples, ten fixed epochs, microbatch 1 and accumulation 16. That gives 1,038 updates per epoch, **10,380 total updates**, **166,000 examples**, and the same eight-example final window each epoch. Three head-only epochs give 3,114 updates, followed by seven joint epochs/7,266 updates. Final-window loss is divided by its actual eight examples; no half-strength nominal-sixteen update is introduced.

AdamW uses head LR 2e−4 and weight decay .01, adapter LR 1e−5 and weight decay zero. Head and adapter gradients are clipped separately at 1.0 and .1. Both use linear decay with 3% warmup over their active budgets; the adapter LR is zero during the head-only phase. Source validation is evaluated at every epoch end, with minimum source-validation CE selecting the exported checkpoint. Every branch still trains to the fixed budget. A final-budget state and its validation predictions are saved **before** restoring the selected state; both exports and resumable optimizer/RNG state are retained. No P dataset is passed to the training loop.

Ten epochs is an evidence-backed **pilot maximum**, not a claim of fresh-Level-1 convergence: the historical native-source curves continue improving at late epochs, and warm-start Level-1 convergence does not validate fresh-base training. Any common-budget revision or expansion to all three seeds must be decided from source-validation implementation/convergence diagnostics and resource availability, before locked-P scoring. All conditions must receive the same revised budget. There is no target-side LR or checkpoint tuning.

Seed streams are explicit: head=seed, adapter=seed+10000, presentation=seed+20000+epoch, dropout=seed+30000. Actual presentation permutations are saved and hashed for all ten epochs and three seeds. Same-method source initialization is checked exactly; stochastic trajectories across QLoRA/DoRA or unequal token lengths are not claimed identical. Report every seed separately, then matched-seed means and dispersion. A one-seed pilot does not supply training-seed uncertainty.

### Measured compute and storage feasibility

| Method | Historical head seconds/update | Historical joint seconds/update | Projected train+validation MIG-GPU hours/model |
|---|---:|---:|---:|
| QLoRA | 3.327 | 20.178 | 44.59 |
| DoRA | 3.696 | 20.922 | 46.50 |

Training rates come from adjacent 100-update historical Legacy checkpoint mtimes, excluding epoch boundaries; all raw intervals, medians/IQR and interruption outliers are retained. These are actual native-input timing proxies including checkpoint I/O, **not measured fresh-Level-1 backward throughput**. Validation and inference use the newly measured Level-1 batch-eight forward timings from all twelve initialization smoke constructions. No unmeasured FLOP scaling or compression factor is presented as fact. All hours below are **MIG-slice GPU hours**, not full-H100-equivalent hours.

One seed: **182.17** training/validation hours plus **11.93** selected-state paired-inference hours = **194.10**. Three seeds: **582.31** including selected-state inference; the two extra seeds require **364.34** additional training/validation hours. Evaluating fixed-final states as well adds a second paired-inference pass (23.86 total inference hours per seed for both states). No such full inference has run.

With the three currently free slices and indivisible model jobs, the proposed one-seed queue has a training/validation makespan proxy of **89.2 hours**, rather than incorrectly dividing four jobs’ GPU hours by three. Twelve jobs balance into four waves, approximately **182.2 hours** before inference. The other four slices remain assigned to existing CLAUDS work. Queue waits, failures, new Level-1 gradient speed and memory differences can change these projections.

Using measured historical adapter/head and full-resume sizes, projected four-model run storage is 0.51 GiB, or 1.53 GiB for twelve runs, excluding shared base/data. Selected-state raw logits+PDFs require approximately 1.10 GiB per seed; selected plus final doubles this. Actual compressed size is not assumed. Initial forward-only peak memory is recorded; full gradient/optimizer memory was not profiled because new Qwen training was not authorised. The existing same-architecture training runs provide compatibility evidence, not a new memory guarantee.

## D — Matched transfer evaluation plan

For each method, seed and checkpoint selection, evaluate the exact same P objects in all four cells:

| Source model | Test HSC | Test Legacy |
|---|---|---|
| HSC-trained | CE(H,H) | CE(H,L) |
| Legacy-trained | CE(L,H) | CE(L,L) |

Report absolute CE/CRPS/NMAD/outlier fraction, mean and median normalized bias, 68% coverage and PIT in every cell. The point estimate is the PDF median-bin centre and delta=(z_pred−z_spec)/(1+z_spec), with outliers |delta|>0.15. Every grid/probability-mass convention is verified before CE comparison. Selected-checkpoint results are primary; fixed-budget exports are separately labelled trajectory controls. Source-validation numbers are never substituted for P-cell metrics.

| Quantity | Definition |
|---|---|
| H→L same-model penalty | CE(H,L) − CE(H,H) |
| L→H same-model penalty | CE(L,H) − CE(L,L) |
| H→L target-native gap | CE(H,L) − CE(L,L) |
| L→H target-native gap | CE(L,H) − CE(H,H) |
| Method robustness | P_DoRA − P_QLoRA, plus absolute target CE differences |
| Source×observation interaction | [CE(H,L)−CE(H,H)] − [CE(L,L)−CE(L,H)] |

Negative target-native gaps are permitted. A smaller penalty may reflect worse native performance, so absolute cells and target-native gaps remain essential. The interaction is descriptive and equals the sum of the two directional same-model penalties under these sign conventions. It is not a causal estimate of filter transmission alone.

Fixed redshift edges: `[0.0, 0.2, 0.4, 0.6, 1.0, 4.5]`. Report every cell, penalty, target-native gap, method comparison and calibration diagnostic by bin, plus native-population and equal-bin macro summaries. Macro NMAD is explicitly a mean of within-bin NMADs, not a pooled weighted NMAD. The P tail has 400 objects above z=1 and none above z=2; this does not demonstrate strict redshift extrapolation.

Use the same saved 1,000 paired-galaxy bootstrap draws across all models, methods, domains and seeds. Compute percentile 95% intervals directly on penalties, target-native gaps, method differences and interactions. Pair object-level absolute normalized residual changes and newly catastrophic/rescued outliers by physical galaxy. Keep conditional galaxy-bootstrap uncertainty distinct from across-seed means/dispersion; galaxies are not substitutes for independent training seeds.

The matched design holds physical populations, labels, sample sizes, memberships, initialization and major optimization exposure fixed. It does not isolate photometric extraction, depth/noise, calibration, extinction history or the different information in grizy versus grzW1W2. The description is **photometric observational-system transfer**, with no additional filter-metadata interventions.

## E — Run-ready execution package

Persistent package: `/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design`. Compact report/artifact copies live under `reports/filter-aware/1028/matched-training-design/`. Package manifest SHA256: `bc7ac99fda7629f1235fb1a4c547e699fa9039984e82a474ebe654d418183661`.

| Location | Contents |
|---|---|
| legacy_checkpoint_audit/ | Exact selected/restart paths and hashes, original histories, verified reconstructed IDs/features/raw-row references, source-validation replay proofs |
| original_model_exposure/ | All seven models’ exposure flags; 61,310-object P_mutual; eight original-model inference configurations |
| paired_split/ | Immutable full cohort and T/V/P manifests, coordinate/measurement/quality audits, distributions |
| representation_audit/ | Exact Level-1 input strings, byte hashes, common contract and token/truncation audit |
| model_initialisation/ | Verified base identity, three common random heads, canonical adapter/head states and six source-equality proofs, ten-epoch presentation arrays |
| training_configs/ | Twelve final configurations plus explicitly labelled preparation drafts |
| evaluation_policy/ | Fixed matrix/metrics/contrasts/bins/seed policy, saved bootstrap indices, selected/final frozen-inference configurations |
| feasibility/ | Raw historical timing intervals, measured initialization forward timings, transparent cost/storage projections |
| launch_commands.txt | Exact prepared training and original-model inference commands, not executed |

CPU implementation checks passed: bookkeeping-invariant historical sampling; common initial heads and exact epoch permutations; correct eight-example partial-window averaging; RA-wrap and cross-representation leakage detection; exact interrupted/resumed toy-optimizer equivalence with all updates/examples retained. The latter optimizes only a tiny synthetic CPU network to validate the state machine, not Qwen. GPU checks consist of the two 64-object original source-validation replays and twelve fresh forward-only model constructions; **zero new Qwen optimizer updates** occurred.

After a separate training go-ahead, the one-seed pilot command is:

```bash
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/launch_matched_training_1028.py --package /arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design --seeds 42
```

The full predeclared three-seed command, if separately approved after source-validation viability/resource review, is:

```bash
PIXI_CACHE_DIR=/scratch/.tmp-gsm/pixi-cache pixi run --locked python notebooks/launch_matched_training_1028.py --package /arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1028/matched-training-design --seeds 42 43 44
```

The launcher reserves the three specified free slices, verifies package/config/code identities, skips completed runs and resumes only matching run-specific checkpoints. It never launches P inference automatically. Original-model reverse inference has separate configurations/commands and remains unexecuted. Future new-model selected/final inference configurations require their newly trained exports and verify checkpoint SHA256 identities before scoring.

Completion checks for every future run: exact 10,380 updates/166,000 examples, own-source validation history, selected and fixed-final exports, saved validation PDFs, consistent grid, source-specific trained checkpoint hashes, and restartable optimizer/scheduler/CPU/CUDA RNG state. First source-validation replay must match the exported state before any locked-P matrix is scored.

No existing checkpoint, filter registry, old benchmark result or counterfactual manifest was overwritten. Full original-model inference and four-model/full three-seed training still require the separate authorization specified in the task.
