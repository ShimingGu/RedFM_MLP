# Joint Stage3 assessment —1029

1. **Independent Legacy recovery:** yes. Exact historical input fingerprints, sampling/split counts and all15,053 validation IDs/redshifts/native features reproduce. Original59,855 training objects are available; downstream source/test, spatial and identity exclusions are explicit.

2. **Actual source high-z coverage:** no z≥2 objects occur in the complete recovered selected Legacy training population or HSC training. This describes the selected GALAXY quality cohort, not all DESI spectroscopy. Finer eligible counts and joint magnitude/SNR/colour coverage are saved.

3. **Independent CLAUDS confirmation:** yes. Public COSMOS Compilation DR1.1 high-confidence spectroscopic matches were independently reserved by identity, with stringent ambiguity cuts and historical source exclusions.2,045 identities were locked before labels; final eligible N=1976.

4. **Real source cross-survey R2 versus R1:** direction-dependent. HSC→Legacy differences are+14.37pp and−12.39pp (seed reversal). Legacy→HSC differences are−10.60pp and−17.90pp (point gains in both seeds), but CRPS is worse in one seed. Only1/12 primary runs reaches the final source plateau; fully optimized superiority remains unestablished.

5. **Capacity/information controls:** C1 matches parameter count, C2 supplies complete responses, C3 supplies dimension-matched generic smooth kernels. R2 does not survive these controls as a stable FSPS-specific gain. Generic C3 point errors are40.13%/47.93%, versus R2 41.95%/55.92%; seed11 difference is inconclusive and seed23 favours C3. External FSPS information remains a substantive difference from R1.

6. **Independent CLAUDS transfer:** primary full-band Mixed R2−R1 differences are seed11 -13.71pp, seed23 +6.53pp. Both-seed registered outlier criterion=False. The registered two-seed criterion fails: one seed improves and the other worsens; their95% paired intervals lie on opposite sides of zero. Target CE/CRPS likewise improve in seed11 and worsen in seed23.

7. **High-z/faint relevance:** high-z lacks source supervision and is explicitly extrapolation. All14 z≥2 targets are outliers for R1 and R2 in both seeds; this tiny stratum cannot establish population-level high-z performance. T2 is much brighter than the unlabeled local population; do not infer all-CLAUDS faint accuracy from spectroscopic selection. T1 agreement remains exploratory.

8. **Credible physical representation:** a mask-aware set encoder improves the secondary R0 comparison at this budget, but a stable external FSPS representation advantage is not established. R2 changes direction across seeds and does not outperform the generic basis. D12 achieves24.09% point outliers while catastrophically miscalibrated (CE≈163,658;68% coverage12.55%); neither this nor source point gains validates a universal physical likelihood.

9. **Defensible foundation-model comparison:** obtain representative high-z/faint spectroscopy and robust source convergence, keep numerical/physical controls, give all models matched observations/cohorts, include frozen Qwen and randomized/pretraining controls, and preregister any new target partition before model development.

10. **Stage2C decision:** defer and redesign around those unresolved support/attribution requirements. No Qwen training or Stage2C run was started.

[Stage3A readiness](stage3a/1029/report.md) · [Stage3B results](stage3b/1029/report.md) · [Progress figure](stage3b/1029/cross_survey_summary.png)
