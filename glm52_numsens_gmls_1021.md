# GLM numerical sensitivity: fixed validation subset

Objects: 32; seed: 42; target: DESI spectroscopic redshift.

Inference only. No model was trained and no held-out test object was evaluated.

## Protocol

Each grizy magnitude is perturbed individually in both directions. Its native magnitude error is held fixed, so converted flux and flux error scale by the same factor. EBV, missingness, saved imputation/scaling, text formatting, and model weights remain fixed. Nonfinite bands and perturbations outside the native valid magnitude range are skipped.

GLM uses the saved corrected-implementation **512-object training-fit checkpoint**, evaluated here on validation objects. This is not a full-cohort GLM generalization run. MLP and optional Qwen comparisons are sensitivity controls with different training budgets.

Token sequences are compared before/after perturbation. Hidden changes use the last non-padding token at every hidden-state output. MoE routing compares actual selected expert sets at the last token and at sequence-aligned unchanged tokens; expert order is ignored. Router logits are also compared. These observations establish associations, not causal mediation.

## Prediction comparisons

Flag threshold: |delta median z|/(1+spec-z) > 0.02; a descriptive threshold, not a significance test.

| Model | Step (mag) | Pairs | Median abs delta z | P95 abs delta z | Maximum | Flagged | Median PDF TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLM QLoRA | 0.0001 | 320 | 0 | 0.25575 | 0.495 | 28.44% | 0.17526 |
| GLM QLoRA | 0.001 | 320 | 0 | 0.3 | 0.495 | 37.81% | 0.23657 |
| GLM QLoRA | 0.01 | 320 | 0 | 0.303 | 0.51 | 34.69% | 0.21678 |
| GLM QLoRA | 0.05 | 320 | 0 | 0.2715 | 0.51 | 38.44% | 0.2529 |
| GLM QLoRA | 0.1 | 320 | 0 | 0.27 | 0.495 | 35.00% | 0.23853 |
| Numeric MLP | 0.0001 | 320 | 0 | 0 | 0 | 0.00% | 0.00013411 |
| Numeric MLP | 0.001 | 320 | 0 | 0 | 0.015 | 0.00% | 0.0013394 |
| Numeric MLP | 0.01 | 320 | 0 | 0.015 | 0.015 | 0.00% | 0.013271 |
| Numeric MLP | 0.05 | 320 | 0 | 0.03 | 0.045 | 5.31% | 0.066852 |
| Numeric MLP | 0.1 | 320 | 0.015 | 0.045 | 0.135 | 22.50% | 0.13253 |
| Qwen control | 0.0001 | 320 | 0.0075 | 0.03 | 0.09 | 6.25% | 0.043382 |
| Qwen control | 0.001 | 320 | 0 | 0.03 | 0.105 | 8.44% | 0.047923 |
| Qwen control | 0.01 | 320 | 0.015 | 0.03075 | 0.135 | 12.19% | 0.056311 |
| Qwen control | 0.05 | 320 | 0.015 | 0.045 | 0.12 | 11.25% | 0.055274 |
| Qwen control | 0.1 | 320 | 0.015 | 0.045 | 0.165 | 15.31% | 0.055324 |

## Tokenisation, representations, and routing

Token sequences changed in 1600/1600 perturbations; last-token MoE expert sets changed in 1024/1600.

| GLM condition | Pairs | P95 abs delta z | Median hidden relative L2 | Median PDF TV |
|---|---:|---:|---:|---:|
| Tokens changed, routing same | 576 | 0.195 | 0.2034 | 0.15111 |
| Tokens changed, routing changed | 1024 | 0.38925 | 0.52043 | 0.29075 |

## Interpretation and limits

558/1600 GLM perturbations exceeded the stated response threshold. Large finite-step responses occur in this subset; inspect the matched controls and individual cases before attributing them to numerical boundaries.

The maximum repeated-identical-input GLM median-z difference was 0; maximum PDF TV was 0.

A changed text token is an input boundary; hidden-state movement measures its propagated representation effect. Expert-set changes indicate routing boundaries, but occur downstream of the same changed input. Correlation with prediction changes cannot by itself distinguish routing causation from representation changes. No routing-freezing counterfactual is performed.

PDF medians are quantized to the saved redshift-bin centers. A median-bin jump alone is not evidence of a discontinuous model function: compare continuous PDF TV and mean-redshift changes in diagnostics.csv. The sweep detects finite differences; it does not prove mathematical discontinuities or explain generalization causally.

Results are limited to a fixed validation subset, these perturbation sizes, and available checkpoints. Changes relative to catalogue magnitude errors are recorded as delta_mag_in_sigma; no universal physical threshold is assumed.

Optional Qwen: Saved frozen-Qwen + trained PDF-head control: /arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10/frozen/aion_baseline.pt

## Plots and artifacts

![Prediction response](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/prediction_sensitivity.png)

![Token, representation, and routing associations](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/token_representation_routing.png)

![Hidden layer response](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/hidden_layers.png)

![Largest GLM response](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/largest_response.png)

Per-pair measurements: [diagnostics.csv](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/diagnostics.csv). Largest GLM cases and token sequences: [largest_cases.json](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/largest_cases.json). Fixed subset/provenance: [manifest.json](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/manifest.json). Aggregate values: [summary.json](/arc/home/gsm/aion_output/figures/glm52-numerical-sensitivity-galaxiesml-spec-1021/summary.json).
