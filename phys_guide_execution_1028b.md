# Sequential execution — Stage 2A.5 followed by Stage 2B.5

Execute the attached Stage 2A.5 (./phys_guide_1027caa.md) and Stage 2B.5 (./phys_guide_1027cba.md) prompts sequentially and autonomously.

1. Finish Stage 2A.5, including reproducible numerical audits, fixed-spectroscopic-redshift residual analysis, matched HSC–Legacy comparisons and justified source-only interventions.
2. Save the complete Stage 2A.5 report, numerical checks, per-object diagnostics and reproduction manifests.
3. Review the verified findings for corrections relevant to Stage 2B.5, especially filter-response integration, physical likelihood construction, photometric uncertainty semantics and masking.
4. If a correctness issue is found, repair and validate the shared physical infrastructure where feasible. Preserve the original implementations and archived results.
5. Proceed automatically to Stage 2B.5 after Stage 2A.5, without asking for further confirmation, unless an unresolved correctness or resource blocker would invalidate Stage 2B.5.
6. In Stage 2B.5, prioritise reproducibility of the update-zero failure, elementary numerical learnability, matched-template oracle recovery and neural/physical loss-scale diagnostics.
7. Do not use CLAUDS labels to choose corrections, configurations, checkpoints or training settings.
8. Do not evaluate held-out real-survey test sets.
9. Do not launch Qwen training, Stage 2C, or population-scale mixed-survey training.
10. Keep the stages' reports, outputs, experimental configurations and scientific conclusions separate.

At completion, provide a joint report answering:

- What explains the HSC-versus-Legacy discrepancy in EAZY physical inference?
- Was there a genuine physical forward-model mismatch, numerical implementation issue, observational-convention problem or likelihood error?
- What explains the synthetic neural models' update-zero checkpoint selection?
- Can the numerical and physical architectures learn and generalise under a controlled, identifiable task?
- Which validated architecture and physical components are suitable for a future real CLAUDS transfer experiment?
- What still prevents a scientifically interpretable claim about Qwen's pretrained astrophysical knowledge?

Do not claim successful CLAUDS transfer based on source-only or synthetic diagnostics. Proceed automatically between the two stages unless a genuine correctness blocker arises.