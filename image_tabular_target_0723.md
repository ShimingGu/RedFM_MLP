# Compact image representation for tabular models — 2026-07-23

The recommended correction to the current TabICL image experiment is:

1. Decode each AION token ID into its FSQ factor vector.
2. Spatially pool those factors using global moments, coarse 4×4 pooling, or PCA fitted only on the training rows.
3. Append approximately 16–64 continuous image features to the 11 magnitude features.
4. Include a shuffled-image control to verify that any improvement genuinely comes from the matched galaxy morphology.

The goal is to avoid treating discrete token IDs as ordered continuous values and to prevent hundreds or thousands of image columns from overwhelming the stronger photometric signal.

A frozen timm or AION embedding is a sensible alternative, but a compact decoded-FSQ representation is the cleanest correction to the original experiment.

Note that expanding every 24×24 token position into five unpooled FSQ factors produces 2,880 image columns. This fixes the false ordering of raw token IDs, but it does not implement the recommended compact 16–64-feature representation or solve the feature-imbalance problem.
