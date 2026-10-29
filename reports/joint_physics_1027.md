# Joint scientific assessment — Stage 0.5 followed by Stage 1

Both experiment stages completed sequentially. Reports:
[Stage 0.5](stage05/1027/report.md) and [Stage 1](stage1/1027/report.md).

**Current real-data failure mechanisms.** On all 5,683 source-validation objects,
native outliers are 1.07%,
median-EBV outliers 1.13%,
NA-EBV outliers 23.60%, and moving
true EBV to the front gives 31.97%.
This pattern is much more consistent with substantial syntax/terminal-state
fragility than a need for each galaxy's exact EBV value, while not uniquely
separating token identity, position and semantics. Dimming at fixed S/N gives
92.63% outliers and median
z=0.9075; uncertainty-reporting changes alone
are much milder in screening. Actual noisy-faint cases remain separate.
The direction under true EBV is predominantly toward ~0.91, not the archived
CLAUDS ~0.08–0.13 mode. Thus source interventions reproduce severe concentration
but do not uniquely explain the real-domain collapse or its full direction.
These are source-condition sensitivities, not additive causal shares of CLAUDS
outliers. Remaining extinction/calibration, extraction-pipeline and population
differences, and the target's ZPHOT reference quality, remain unresolved.

**Pretrained knowledge and cross-filter tasks.** Stage 1 quantifies constrained
reasoning and the matched frozen-representation advantages without assuming them.
On strictly new filters, pretrained/random/numerical/physical outliers are
70.31%/79.69%/
68.75%/37.50%.
These controlled outcomes do not establish astrophysical pretraining as the
specific cause of any benefit, or success on real CLAUDS. Last-pool gains over
random features are modest and sampling-uncertain, mean pooling does not repeat
them, and the numerical/physical controls remain stronger. Both Qwen feature
heads can memorize the small source training set; zero training outliers do not
translate to physical generalization. Template/sampling/noise
approximations, one random backbone and small independent test-SED counts remain.

**Stage 2 priorities.** First test mask-aware shared band encoders with explicit
response integration and luminosity treated as a nuisance or separately declared
prior, including a rest-frame SED basis with a differentiable band-integration
decoder; compare a numerical set MLP and physical/template baseline. Use a stable
query/set pooling mechanism trained under a matched protocol, with explicit EBV
availability separated from its value. Keep pretrained and matched random frozen
backbones, matched parameter/label budgets, shuffled labels, wrong-metadata and
physically regenerated-response controls, plus held-out SED families and real
survey validation. Do not infer an astrophysical-knowledge advantage from a
supervised fit or shuffled-token degradation. An expensive adapter sweep is not
the next justified default. No Stage-2 model was trained in this execution.
