# Generic timm morphology target

## Goal

Test whether a general-purpose image representation can recognize galaxy
morphology without using an astronomy-pretrained vision backbone.

The primary comparison should be:

```text
AION astronomical representation + Galaxy10 head
generic natural-image representation + Galaxy10 head
```

The generic backbone has not been pretrained specifically on galaxies. Only
the small downstream classifier sees Galaxy10 morphology labels.

## Recommended first model

Use the generic DINOv3 ConvNeXt-Tiny model already configured in
`aion_magnitude/timm_morphology.py`:

```text
hf-hub:timm/convnext_tiny.dinov3_lvd1689m
```

`timm` supports pretrained classifier-free feature extraction and replacing
the classifier for a new task.

For the initial experiment:

1. Keep the timm backbone frozen.
2. Train a small 10-class Galaxy10 head, comparable to the AION head.
3. Use the identical Galaxy10-AION train/test split.
4. Temperature-calibrate the output logits.
5. Collapse the ten class probabilities with the existing
   `collapse_galaxy10_morphology_probabilities()` function.

The resulting quantities retain the current AION semantics:

- `p_spiral`: sum of Galaxy10 classes 5–9.
- `p_bar`: probability of the barred-spiral class.
- `p_elliptical_type`: sum of Galaxy10 classes 2–4.

Store the comparison outputs separately:

- `p_spiral_timm`
- `p_bar_timm`
- `p_elliptical_type_timm`

Do not overwrite the AION columns.

## Image input

The timm model should inspect the real 96×96 CLAUDS `u/uS` cutout. Avoid
constructing replicated grizy proxy images for timm, since that could
encourage the network to use colour or magnitude proxies rather than visible
shape.

For Galaxy10 training, create a deterministic grayscale structural image from
its four bands, with the same robust normalization used for the CLAUDS
cutouts. Rotation and reflection augmentation are appropriate because the
morphology label should be orientation-invariant.

## Experimental stages

### Primary comparison

Use a frozen generic backbone and train only the morphology head. This tests
whether a general natural-image representation already contains useful
features for spiral arms, bars, and smooth elliptical structure.

### Optional later comparison

Unfreeze the final ConvNeXt stage and fine-tune it with a small learning rate.
This should be reported as a separate astronomy-adapted condition, not as the
generic-backbone result.

### Zero-astronomy baseline

A zero-shot CLIP experiment could use prompts such as "a barred spiral
galaxy", but its prompt scores would not be well-calibrated morphology
probabilities. It is therefore less directly comparable than frozen timm plus
the shared Galaxy10 head.

## Evaluation

Compare AION and timm on the same held-out objects using:

- Galaxy10 ten-class accuracy.
- Trait-specific AUROC.
- Brier score and negative log-likelihood.
- Calibration/reliability curves.
- Per-class recall, especially for barred and spiral subclasses.
- Rotation consistency or rotation-averaged inference.

The outputs should be interpreted as probabilities of morphology *visible in
the supplied 96×96 image*, not necessarily intrinsic morphology. Faint,
small, or high-redshift galaxies may not contain enough resolved information
for a confident bar or spiral classification.

## Implementation boundary

`aion_magnitude/timm_morphology.py` already provides:

- the generic pretrained timm encoder;
- single-channel cutout preprocessing;
- frozen embedding extraction and caching.

The remaining work is:

1. Reuse the Galaxy10 benchmark loader and split.
2. Add grayscale Galaxy10 preprocessing.
3. Train and calibrate the timm 10-class head.
4. Apply the shared probability-collapse function.
5. Add resumable CLAUDS catalogue inference with model-specific output
   columns.

