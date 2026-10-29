# Engineered photometric kNN integrity audit

GMS raw source is HSC grizy observed flux/error and MW EBV, joined on persistent target ID; the Stage 4 numeric path produces scaled flux/error/SNR, mask, quality/availability, brightness and EBV/availability. It contains no input target redshift, ZPHOT, supervised redshift estimate or template-fit output. Support-only mean/STD with floor and clipping, k=10, inverse-distance weights and the fixed redshift KDE exactly reproduce the saved Stage 4 PDFs, maximum discrepancy 0. The query-label and query-row perturbation tests pass. Source fields and selection metadata may carry observational/population information; absence of identified leakage is not a proof of its impossibility.

There are no cross-role shared IDs/groups, no support/query match within 2 arcsec (minimum 30.03), and no identified BS/BC historical train or validation exposure/ambiguity. See [asset_and_identity_audit.json](asset_and_identity_audit.json) and [knn_reproduction_checks.json](knn_reproduction_checks.json).

Original 2.54%; constant median 34.44%; 60 fixed support-label permutations average 37.98%. Normalized shape 4.49%; brightness 7.10%; their union 3.45%; quality/selection-only 17.64%. See [knn_metrics.csv](knn_metrics.csv), [permutation_null_metrics.csv](permutation_null_metrics.csv) and the [full assessment](scientific_assessment.md).
