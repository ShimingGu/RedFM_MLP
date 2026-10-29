Execute Stage 2A (./phys_guide_1027ca) first, then Stage 2B (./phys_guide_1027cb) sequentially.

Stage 2A must complete its physical input checks, filter integration tests, and baseline evaluation before Stage 2B begins.

If Stage 2A identifies an unresolved photometric-unit, extinction, response-curve, or data-leakage error that invalidates the proposed Stage 2B training, stop and report the blocker.

Otherwise, proceed automatically to Stage 2B without requesting additional approval.

Use verified Stage 2A physical infrastructure, but keep all model-training results and scientific reports separate.

Stage 2B must begin with correctness tests and a small N0/N1/N2 pilot. Expand only if the gates in its prompt pass.

Never use CLAUDS target redshift labels for training, selection, calibration, or iterative tuning.

Do not train Qwen, initiate Stage 2C, overwrite historical experiments, or interrupt existing GPU jobs.

Deliver two independent reports and one short comparison explaining whether explicit physics, numerical architecture, or neither provides credible CLAUDS transfer.