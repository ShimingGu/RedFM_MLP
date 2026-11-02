# RedFM Paper I — High-z H1 Pilot, Head Controls, and Concurrent-Experiment Isolation

## Objective

Implement and execute the first controlled higher-redshift extrapolation pilot for RedFM Paper I.

This task belongs exclusively to the **redshift-generalisation research branch**.

Other experiment branches are simultaneously running against the same research repository and HPC infrastructure. Preventing interference with those branches is a hard requirement, not merely a GPU availability preference.

The experiment has two main scientific questions:

1. Can a Qwen foundation model post-trained only below a fixed spectroscopic-redshift cutoff extrapolate beyond that cutoff?
2. How much of any apparent extrapolation failure is attributable to the representation learned by the foundation model versus the supervised prediction head and loss?

The first pilot uses HSC photometry, Qwen3.5-4B-Base, QLoRA, seed 42 and a fixed redshift cutoff of 0.6.

**Authorisation:** audit and revise the high-z experiment, implement its controls, and launch the six pilot training arms after the explicitly defined correctness and isolation gates pass. No broader model or hyperparameter sweep is authorised.

---

# Phase 0 — Protect concurrent experiments

## 0.1 Concurrent workloads

The following research tasks are independently running or prepared:

| Branch | Status | Scope |
|---|---|---|
| Main matched transfer | Seeds 43/44 running | 12 models, 24 final evaluation cells |
| Morphology M1 | Seed 42 training running | Four new models, 16 P0/M1 evaluation cells |
| Morphology M2 | Data audit complete | Training not authorised |
| High-z H1 | This task | Six initial HSC QLoRA pilot arms |

The high-z task must not change, restart, delay by deliberate preemption, terminate, or otherwise interfere with the first three branches.

In particular, **do not touch existing main matched-transfer or morphology training workers, queues, configurations, output files, saved checkpoints, locks, or frozen evaluation policies.**

## 0.2 Isolated source-code workspace

Before changing code:

1. Locate the repository root and its exact current commit.
2. Inspect active branches, worktrees and running training entrypoints.
3. Create a **dedicated Git worktree and high-z experiment branch** from a recorded, compatible base commit.
4. Make all high-z code changes inside that worktree.
5. Do not modify tracked files in the worktree used by other Codex agents.
6. Do not switch the active branch in the shared working directory.
7. Do not perform `git pull`, `reset`, `checkout`, `rebase`, dependency updates or cleanup operations in another branch's workspace.

Prefer high-z-specific modules and entrypoints over edits to shared training code.

If a shared module must be changed, implement the required change in the isolated worktree and test it there. Do not propagate it into the running branches.

Record the base Git SHA, new branch name, worktree path and exact modified files.

## 0.3 Isolated experiment artifacts

Existing high-z feasibility package:

`/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-feasibility`

Read its frozen design files and reproduce their hashes. Treat this package as read-only.

Create a separate persistent execution root:

`/arc/projects/ots/Cosmic_Imprint_of_Time/filter_aware_checkpoints/1102/highz-hsc-pilot-zcut0p6`

All newly generated:

- training checkpoints,
- optimizer/RNG states,
- metrics,
- source-validation predictions,
- pilot P predictions,
- logs,
- manifests,
- plots,
- evaluation configurations,

must be written exclusively beneath this new root.

Use distinct checkpoint names and process locks for each high-z training arm.

Read-only access to existing pretrained model files and frozen catalogue assets is allowed.

Do not duplicate or overwrite shared Qwen base weights.

## 0.4 GPU and process isolation

The user expects additional GPU resources to be available, but this must be verified at execution time.

Before launching:

- Enumerate actual GPU/MIG assignments and existing job reservations.
- Verify the resource requirements of the matched-transfer and morphology workers.
- Reserve only independent GPU/MIG slices.
- Use per-device reservation locks and high-z-specific process locks.
- Do not assume low instantaneous utilisation means a GPU is unreserved.
- Use existing cluster job-allocation conventions where applicable.
- Avoid changing global CUDA, scheduler or environment configuration used by other workers.
- Avoid unnecessary whole-catalogue reloads or competing high-volume scratch writes.
- Use an isolated temporary/cache namespace where practical.
- Preserve the ability to resume every training arm independently.

Do not modify GPU allocations of other running jobs.

If no genuinely safe compute resources are available, leave this task queued under its own isolated reservation mechanism without changing other experiments.

Other experiments always retain their existing reservations.

---

# Phase 1 — Audit the existing high-z protocol

## 1.1 Existing data and split

The high-z feasibility report established the following paired HSC cohort:

- Train T: 16,600 galaxies.
- Validation V: 4,257 galaxies.
- Locked evaluation P: 61,696 galaxies.

At the selected cutoff \(z_{\rm cut}=0.6\):

- Low-z training candidates: 14,414.
- Low-z validation candidates: 3,745.
- Higher-z P galaxies: 7,910.
- Lower-z P galaxies: 53,786.

The earlier protocol used:

- Exactly 10,489 distinct training galaxies per arm.
- Ten epochs.
- 6,560 optimizer updates.
- 104,890 training presentations.
- Full 300-bin redshift support over [0,4.5].
- Source-only checkpoint selection.
- Common Level-1 HSC representation.

Retain the already frozen cutoff, cohort and primary sample-size policy unless the audit identifies a correctness defect.

A larger training sample might eventually be scientifically useful, but do not introduce one in this first controlled pilot.

Verify the actual training membership and sample composition independently from the saved manifests.

## 1.2 Three training conditions

Use the existing predeclared conditions:

**A. Low-z only**

Exactly 10,489 distinct training galaxies with:

\[
z_{\rm spec}<0.6.
\]

No training example above the cutoff.

**B. Natural tail**

Exactly 10,489 distinct galaxies drawn from the original T population using the frozen hash policy, retaining the natural high-z scarcity.

**C. Tail enriched**

Exactly 10,489 distinct galaxies, including all available original-T galaxies above the cutoff, with the remaining slots filled using low-z galaxies without replacement.

Do not generate synthetic galaxies or duplicate high-z training examples.

Report the actual below-cutoff and above-cutoff counts for every condition.

Use the exact same galaxy membership between the two prediction-head variants for each condition.

Low-z-only is the strict supervised-support extrapolation experiment.

Natural-tail and tail-enriched are supervised high-z-exposure controls; do not call their high-z test performance strict extrapolation.

---

# Phase 2 — Prediction-head audit and control

## 2.1 Audit the original classifier

The existing RedFM head predicts a 300-bin probability mass vector over redshift [0,4.5].

The training loss is categorical cross-entropy.

For a bin \(j\) without positive training labels:

\[
\frac{\partial L}{\partial s_j}=p_j.
\]

This induces pressure to suppress bins above the supervised training-redshift cutoff.

This observation motivates a control, but it does not mathematically prove that a softmax classifier cannot extrapolate.

Inspect:

- Exact head implementation.
- Probability-mass and redshift-grid conventions.
- CE and CRPS implementations.
- Target redshift binning.
- Softmax normalisation.
- Initial logits for untrained bins.
- Prediction-head parameter initialisation.
- Numerical treatment of very low probability bins.

Do not change the archived classifier or existing training contracts.

## 2.2 Implement a continuous probabilistic head

Implement one controlled alternative:

**Three-component truncated Gaussian mixture head.**

Let:

\[
p(z\mid x)
=
\frac{\sum_{k=1}^{3}\pi_k(x)\,
\mathcal N(z;\mu_k(x),\sigma_k(x)^2)}
{\int_0^{4.5}\sum_{k=1}^{3}\pi_k(x)\,
\mathcal N(t;\mu_k(x),\sigma_k(x)^2)\,dt}
\]

for \(0\le z\le4.5\).

The prediction head must use the same backbone representation and shared hidden-layer architecture as the original RedFM head wherever possible.

Parameterise mixture weights, means and positive scales with stable differentiable transformations.

Predeclare a physically and numerically defensible minimum scale. Do not optimise this parameter using high-z P results.

For each existing redshift bin \([a_j,b_j)\), calculate probability mass by integrating the mixture:

\[
P_j(x)=\int_{a_j}^{b_j}p(z\mid x)\,dz.
\]

Use stable CDF differences and appropriate floating-point precision.

The vector of bin probabilities must sum to one within numerical tolerance.

Use the same binned CE objective and the same existing RedFM evaluator for both prediction heads.

The continuous head must not:

- Clip predictions at the training cutoff.
- Condition on the true redshift at inference.
- Change the observational input representation.
- Read test labels for calibration.
- Introduce a different definition of CE.

Document its trainable parameter count relative to the 300-bin head.

Do not claim identical head capacity if the counts differ.

This head is an architectural control, not an automatically superior estimator.

## 2.3 Numerical tests

Before training, require tests demonstrating:

1. Every predicted probability is finite and nonnegative.
2. Total probability mass is normalised.
3. Bin-integrated probability agrees with a trusted numerical integration reference.
4. Analytic or autodifferentiated gradients agree with finite-difference checks at representative inputs.
5. No NaNs or infinities appear for narrow or broad mixture components.
6. Both heads use the same redshift bin edges and label conventions.
7. CE/CRPS and quantile calculations correctly consume either head's binned PDF.
8. Model save/reload reproduces output PDFs.
9. The new head has nonzero gradients and parameter updates during training.
10. No operation accesses or mutates another experiment branch's code or artifacts.

Use small synthetic inputs and permitted T/V records only for these tests.

Do not inspect P predictions.

---

# Phase 3 — Freeze checkpoint-selection and evaluation policy

## 3.1 Primary checkpoint definition

Use the **fixed ten-epoch final checkpoint** as the primary scientific comparison for all six pilot arms.

This avoids selecting models under a low-z-only validation criterion that might systematically penalise conditions with high-z supervised exposure.

Every arm must complete the same fixed training budget.

Do not extend some arms merely because they converge more slowly.

## 3.2 Secondary source-validation checkpoints

Retain the source-validation-selected checkpoint as a secondary diagnostic.

For all six arms, use the same original V galaxies with \(z<0.6\), selected independently of test predictions.

Record whether the best-validation and fixed-final checkpoints differ.

A separately defined full-range validation-selection policy for the tail-exposed controls can be proposed for future experiments, but is not part of the primary six-arm comparison.

No high-z P labels may influence checkpoint selection.

## 3.3 Locked evaluation

Freeze the six-arm evaluation before starting training.

Primary tests:

- P galaxies with \(z_{\rm spec}\ge0.6\): 7,910.
- P galaxies with \(z_{\rm spec}<0.6\): 53,786.

Both groups belong to the same original locked P cohort and are disjoint from T/V.

This P cohort has been used by other RedFM studies. Describe it as a previously analysed, locked benchmark, not a new prospective blind test.

Do not read prediction outputs from the matched-transfer or morphology branches to choose high-z training conditions.

High-z testing may take place only after all six training arms and their fixed checkpoint identities are complete and frozen.

---

# Phase 4 — Execute the H1 pilot

## 4.1 Six authorised arms

| Arm | Training sample | Head |
|---|---|---|
| H1-A | Low-z only | 300-bin softmax |
| H1-B | Natural tail | 300-bin softmax |
| H1-C | Tail enriched | 300-bin softmax |
| H1-D | Low-z only | Truncated Gaussian mixture |
| H1-E | Natural tail | Truncated Gaussian mixture |
| H1-F | Tail enriched | Truncated Gaussian mixture |

Model:

`Qwen3.5-4B-Base`

Method:

`QLoRA`

Seed:

`42`

Observations:

`HSC grizy, common Level-1 representation`

The two head architectures must not change the backbone adapter settings.

Use the existing validated training infrastructure from the isolated high-z worktree.

Where compatible, retain the matched-transfer pilot's three head-only epochs and seven joint-training epochs.

Match:

- 10,489 training examples.
- Ten fixed epochs.
- 6,560 optimiser updates.
- 104,890 presentations.
- Effective batch size.
- Backbone initialisation.
- QLoRA rank and target modules.
- Head hidden-layer structure.
- Optimiser and scheduling protocol.
- Evaluation cadence.
- Checkpoint export format.
- Data ordering seed policy.

If the continuous head requires an implementation difference, document it rather than pretending the two architectures are identical.

Use shared initial head-trunk weights for the two architectures when shapes permit; independently initialise the architecture-specific output layers with fixed seeds.

## 4.2 Training safety

Before launching an arm:

- Verify the current branch and code hash.
- Verify input manifest identities.
- Verify no P objects are loaded into the worker.
- Verify checkpoint/output directory is unique.
- Verify resource isolation.
- Perform a short, source-only first-backward smoke test.
- Check allocated GPU memory and throughput.

The smoke test must not consume the official training sequence or alter the canonical initial state.

Then launch or queue the full training arm.

If an arm crashes, resume only from its verified own checkpoint.

Do not modify shared code or another experiment's checkpoints to repair the problem.

If the continuous head fails a correctness test, do not launch its three arms. Document the blocker and retain the valid classifier experiment.

## 4.3 Training diagnostics

At every epoch, record:

- Training CE.
- Low-z source-validation CE.
- Source NMAD and bias.
- Outlier fraction.
- CRPS and coverage.
- Parameter and gradient norms.
- PDF entropy and scale diagnostics.
- Best-selected versus fixed-final checkpoint.
- Optimiser steps and examples seen.
- GPU memory and wall time.

The training conditions intentionally have different redshift compositions, so do not interpret their raw training-loss differences as equivalent predictive difficulty.

---

# Phase 5 — Controlled pilot evaluation

Only after the six training runs finish and the exact final checkpoints are frozen, evaluate all six on the existing locked P cohort.

If some authorised arms are blocked by implementation failure, complete only scientifically valid comparisons and explain the reduced matrix. Do not silently replace them.

## 5.1 Primary metrics

For both below-cutoff and above-cutoff test strata, report:

- CE.
- CRPS.
- NMAD.
- Catastrophic outlier fraction.
- Mean and median normalised bias.
- 68% credible interval coverage.
- PIT.
- PDF probability mass above \(z=0.6\).
- Median predicted redshift.
- Fraction of high-z galaxies predicted below \(z=0.6\).

Use the existing definition:

\[
\delta z=\frac{z_{\rm phot}-z_{\rm spec}}{1+z_{\rm spec}}.
\]

Do not infer successful extrapolation from NMAD alone.

## 5.2 Primary scientific contrasts

Compare:

**A versus D:** strict extrapolation with discrete versus continuous head.

**A versus B/C:** effect of high-z supervision under the same discrete head.

**D versus E/F:** effect of high-z supervision under the continuous head.

**B versus C and E versus F:** effect of natural versus enriched tail exposure.

Report absolute performance for every arm, not only relative differences.

A continuous head may have broader PDFs and therefore better high-z CE without making useful high-z point predictions. Evaluate both calibration and point accuracy.

## 5.3 Baselines and failure modes

Include training-only, non-test-fitted baselines:

- A fixed probability distribution derived from the training-label prior.
- A predeclared boundary-centred predictive distribution.

Report whether low-z-only models:

- Predict almost all high-z galaxies below the cutoff.
- Concentrate near the cutoff.
- Assign small but nonzero probability mass to high-z.
- Produce informative, input-dependent predictions above the cutoff.
- Obtain better probabilistic scores merely by becoming more diffuse.

Do not call an excessively broad PDF successful astrophysical redshift recovery solely because it has lower CE.

## 5.4 Redshift and selection stratification

Use fixed redshift intervals above and below the cutoff.

Include the available \(0.6\)–\(0.8\), \(0.8\)–\(1.0\), \(1.0\)–\(1.5\) and \(1.5\)–\(2.0\) populations, with explicit object counts.

Report higher-redshift bins as exploratory when their sample counts are too small for stable inference.

Diagnose magnitude/SNR differences between training and test redshift strata.

A strict redshift-support holdout still produces a joint shift in galaxy properties and selection. It does not isolate redshift while holding all photometric observables fixed.

Use shared paired-galaxy bootstrap indices for comparisons between the six arms.

One seed does not establish training-seed robustness.

---

# Phase 6 — Reporting and next gate

Create the following persistent deliverables:

```text
highz-hsc-pilot-zcut0p6/
    isolation_audit/
    protocol/
    head_tests/
    splits/
    training/
        H1-A/
        H1-B/
        H1-C/
        H1-D/
        H1-E/
        H1-F/
    evaluation/
    figures/
    report.md
    manifest.json
    status.json
```

The report must contain:

### A. Isolation audit

- Exact Git worktree and commit.
- Confirmation that matched-transfer and morphology workspaces were not modified.
- GPU/MIG resource ownership.
- Unique output and checkpoint paths.
- Shared assets accessed read-only.
- Any blocking concurrency issues.

### B. Prediction-head investigation

- Mathematical and implementation differences.
- Head parameter counts.
- Numerical tests and gradient checks.
- Interpretation of classification-bin suppression.
- Limitations of the continuous alternative.

### C. Training results

- Six-arm completion status.
- Exact training cohort compositions.
- Fixed-budget equivalence.
- Complete source-validation histories.
- Best versus final checkpoints.
- Actual GPU resource consumption.

### D. Extrapolation results

- Full six-arm comparison.
- Below-cutoff and above-cutoff performance.
- Predicted-redshift collapse diagnostics.
- PDF mass above the cutoff.
- Point and probabilistic metrics.
- Redshift-stratified results.
- Paired bootstrap contrasts.
- Interpretation of differences between supervised extrapolation and supervised high-z exposure.

### E. Next scientific decision

Answer:

1. Does the softmax head show evidence of suppressed unobserved-redshift bins?
2. Does the continuous head meaningfully change the strict-extrapolation result?
3. How much do a few high-z labels change performance?
4. Are the differences attributable to redshift recovery, PDF broadening, or both?
5. Are these results useful enough to justify expansion to Legacy, DoRA, additional seeds or other foundation models?
6. What is needed before the separately prepared Legacy \(z_{\rm cut}=1.5\) experiment?

Do not automatically launch Legacy, DoRA, new seeds, MLP/TabICL, the \(z_{\rm cut}=1.5\) experiment or any \(z>2\) experiment.

Those require a separate scientific review.

**Execute the isolated HSC pilot after its correctness gates pass. Preserve the running matched-transfer and morphology branches completely. Do not stop at producing another design-only proposal.**