# BS checkpoint and native replay

BS base, adapter and original head were loaded frozen using the Stage 4 provenance. Native strings were reconstructed from source HDF feature rows matched by ID and compared to Stage 4 photometry; 2304 clean objects fit within the historical 512-token limit (maximum 237, zero truncated).

The historical 32-validation-example replay and the additional 32-query-example replay both pass with zero maximum logit and PDF discrepancy. Every clean object belongs to historical test, and all 1,024 matched query redshifts are identical to archived truth. Native query predictions are the archived outputs with an audited exact subset replay; the native cache itself was newly computed.

Native BS outliers 0.98%; canonical BS historical head 36.82%. All output uses the 0–4.5, 300-bin grid. Full metrics: [native_vs_canonical_metrics.csv](native_vs_canonical_metrics.csv). Detailed manifest: [native_extraction.json](native_extraction.json).
