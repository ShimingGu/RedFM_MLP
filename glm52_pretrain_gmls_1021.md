# GLM toy pretraining versus random initialization: GalaxiesML-Spec

Date: 2026-10-21

**The toy-pretrained checkpoint showed no substantial validation cross-entropy advantage or disadvantage over random initialization in this short screen. Frozen-head training showed a tiny, consistent CE benefit at 512 objects; QLoRA was essentially tied. Small point-metric advantages were present, but the experiment does not establish useful positive transfer, harmful negative transfer, or equivalence after sufficient training.**

## Experiment and completion checks

- Dataset: quality-selected GalaxiesML-Spec, using native HSC grizy fluxes and errors plus EBV. **True z is `DESI_fibermap/DESI_redshift`, not CLAUDS ZPHOT.**
- Same tokenizer, five-decimal formatting, explicit missing values, nested training subsets, fixed 256-object validation subset, and redshift grid: 300 bins over [0, 4.5].
- Compared the original toy-pretrained base checkpoint with freshly randomized weights from the identical GLM configuration. Both arms started with fresh downstream heads; no previously trained QLoRA adapter was reused.
- Frozen-head and QLoRA comparisons: 128 and 512 training objects, 5 epochs, effective batch 16. Paired arms shared initial heads, LoRA initialization, minibatch order, optimizer settings, and budgets. Both used NF4 double quantization and the same k-bit preparation.
- Broader comparison: 128 objects, 3 epochs, BF16 backbone and 8-bit optimizer. Embeddings, every attention block, dense/shared MLPs, routers, and norms were trainable; routed expert tensors stayed frozen in both initializations. About 432.4 million parameters were trainable.
- Seeds **42, 43, 44** completed, giving **30 completed arms / 15 matched pairs**. All saved histories contain the initial evaluation and every budgeted epoch. Paired update counts and trainable parameter counts match.
- The initial NMAD differences at 128 objects triggered the predeclared 5% repeat rule. That trigger was a screening signal, not a demonstrated transfer benefit.
- The broader pair completed without an OOM exclusion. Recorded experiment wall time was approximately **30.4 minutes**. No held-out test predictions or metrics were produced.

## Validation cross-entropy

Values are means over the three paired seeds. Positive relative advantage favors toy pretraining; negative values favor random initialization. The advantage is the mean of per-seed relative differences, not a significance estimate.

| Method | Training objects | Toy CE | Random CE | Mean paired toy CE advantage |
|---|---:|---:|---:|---:|
| Frozen backbone + head | 128 | 5.2616 | 5.2886 | +0.51% |
| Frozen backbone + head | 512 | 4.0657 | 4.0758 | +0.25% |
| QLoRA + head | 128 | 5.2641 | 5.2803 | +0.31% |
| QLoRA + head | 512 | 4.0708 | 4.0713 | +0.01% |
| Broader backbone tuning | 128 | 5.2954 | 5.2702 | -0.48% |

At 512 objects, the frozen-head toy advantage had the same direction in all three seeds, but was only **0.084–0.382%**. The QLoRA advantage ranged from **-0.209% to +0.263%**, changing direction across seeds. Broader tuning favored random initialization in two of three seeds, but the mean difference was small and both arms predicted poorly.

## NMAD and outliers

These are also three-seed means at the checkpoints selected by validation CE. Point predictions use PDF medians; outliers satisfy `abs(z_pred - spec_z)/(1 + spec_z) > 0.15`.

| Method | Training objects | Toy NMAD | Random NMAD | Toy outliers | Random outliers |
|---|---:|---:|---:|---:|---:|
| Frozen backbone + head | 512 | 0.1667 | 0.1687 | 46.09% | 48.31% |
| QLoRA + head | 512 | 0.1675 | 0.1712 | 46.61% | 48.44% |
| Broader backbone tuning | 128 | 0.3926 | 0.3873 | 100% | 100% |

Toy initialization has modest point-metric advantages in the 512-object comparisons, so the results should not be described as literally identical. Nevertheless, absolute errors remain high. At 128 objects, both initializations had **100% validation outliers** in every method and seed, despite differences in CE and NMAD. NMAD alone therefore does not establish useful predictions in this screen.

## Learning curves and sample efficiency

All **30/30 arms selected their final budgeted epoch**: epoch 5 for frozen-head/QLoRA and epoch 3 for broader tuning. The screen did not establish convergence or a practical performance frontier.

Both initializations improved substantially when increasing from 128 to 512 objects. However, the fixed epoch budget also increased optimizer updates from **40 to 160**; this does not isolate sample quantity from optimization time. QLoRA offered little improvement over frozen-head training during these five epochs. The limited adapter contribution further restricts what this run can establish about initialization under longer adaptation.

The broader experiment reduced the disadvantage of trying to learn from frozen random features, but it used only **24 optimizer updates**, retained frozen routed experts, and still produced very poor predictions. It cannot decide whether random initialization would ultimately catch up with or exceed the toy checkpoint.

## Conclusion and mechanisms

The defensible finding is **no substantial pretraining advantage or harm detected under these specific budgets**, with a small frozen-head CE edge and modest point-metric edges for the toy checkpoint. The broader result is inconclusive rather than evidence that random initialization is generally better.

Three seeds assess initialization/optimization variability on the same small validation subset; they are not three independent validation datasets. The 5% screen threshold and consistent-direction transfer labels are descriptive rules, not formal significance or equivalence tests.

These results do not identify copypasta content, numerical tokenization, expert routing, or another property as the mechanism. They also do not prove that the previously observed numerical sensitivity was caused by toy pretraining. Neither a switch to random initialization nor full fine-tuning is justified by this screen alone.

## Plots and source artifacts

![Training and validation CE](/arc/home/gsm/aion_output/figures/glm52-pretraining-transfer-galaxiesml-spec-1021/learning_curves.png)

![NMAD and outlier curves](/arc/home/gsm/aion_output/figures/glm52-pretraining-transfer-galaxiesml-spec-1021/nmad_outlier_curves.png)

![Sample-efficiency comparison](/arc/home/gsm/aion_output/figures/glm52-pretraining-transfer-galaxiesml-spec-1021/sample_efficiency.png)

- [Full per-seed report](/arc/home/gsm/aion_output/figures/glm52-pretraining-transfer-galaxiesml-spec-1021/report.md).
- [Aggregate comparisons](/arc/home/gsm/aion_output/figures/glm52-pretraining-transfer-galaxiesml-spec-1021/summary.json), [all 30 arm results and histories](/arc/home/gsm/aion_output/figures/glm52-pretraining-transfer-galaxiesml-spec-1021/results.json), [fixed subsets and provenance](/arc/home/gsm/aion_output/figures/glm52-pretraining-transfer-galaxiesml-spec-1021/manifest.json).
- Driver: [glm52_pretraining_transfer.py](notebooks/glm52_pretraining_transfer.py). Execution: [recent_runs.txt](notebooks/recent_runs.txt:302).
