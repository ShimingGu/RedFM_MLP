# Sequential physics execution — 1028

Executed [Stage 2A](stage2a/1028/report.md) before [Stage 2B](stage2b/1028/report.md), following phys_guide_1027ca.md and phys_guide_1027cb.md. Stage 2A completed physical-input checks, full-response integration tests and baseline evaluation before the handoff. No unresolved correctness error invalidated the Stage 2B pilot. All Stage 2B correctness tests and real-source learnability gates pass; the physical-improvement gate fails, so no population-scale expansion, extra seeds or Stage 2C was launched.

The physical baseline substantially reduces the archived collapse against CLAUDS ZPHOT. The tested neural architectures do not demonstrate credible target transfer. None of these results establishes accuracy against independent CLAUDS spectroscopy.

| Fixed condition | NMAD | Outliers | CE | CRPS | 68% coverage | Width-0.03 concentration |
|---|---:|---:|---:|---:|---:|---:|
| Archived HSC Qwen | 0.1345 | 0.9648 | 19.4920 | 1.4895 | 0.0156 | 1.0000 |
| Stage2A P0 full | 0.1602 | 0.4375 | 28659.7480 | 0.4674 | 0.5391 | 0.0625 |
| Stage2A P2 full | 0.1442 | 0.4141 | 5.9088 | 0.4244 | 0.7617 | 0.0547 |
| N0 | 0.2200 | 0.7422 | 9.5743 | 0.7275 | 0.1914 | 0.8047 |
| N1 | 0.2206 | 0.7227 | 9.0734 | 0.7148 | 0.2266 | 0.4414 |
| N2 | 0.2278 | 0.7109 | 9.2064 | 0.6912 | 0.2227 | 0.3945 |

Explicit physics provides useful object-dependent exploratory ZPHOT agreement: full UV/NIR P0 outliers are 43.8% versus 53.1% with HSC-only measurements, and the source-selected P2 error model gives 41.4%. Errors remain large; P0 is severely overconfident on incompatible objects, and ZPHOT itself is template-derived. Extinction sensitivity is small for these targets but does not prove historical catalogue equivalence.

The shared N1 set encoder improves source validation over N0 in this seed. Adding the N2 SED decoder does not improve source CE and fails the controlled synthetic improvement gate. Its small target outlier change is accompanied by worse CE. The synthetic neural fits select initialization because validation loss worsens during training; this is an explicit limitation, not evidence against astrophysical physics. The fixed numerical runs retain substantial concentration and poor target coverage.

This is a small audited same-galaxy multi-view source experiment, not the definitive independently sampled mixed-source population run. Full Legacy validation (15,053 objects) was recovered, but its original independent training cache remains missing. Source training has no z≥2 spectroscopy while many target references are in that range. One seed, photometric shift, uncertain historical extinction equivalence, nominal HSC r/i responses and different template/label assumptions limit attribution.

CLAUDS labels were excluded from training, checkpoint selection, error calibration, architecture and augmentation selection. All model locks predate target evaluation and remain identical after the empty-set numerical repair. Historical experiment/model hashes are unchanged. See physics_execution_1028_verification.json and the two independent manifests for exact commands, dependencies, input hashes, PDFs and gates.
