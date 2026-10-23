# Cross-survey calibration: feasibility assessment, 2026-10-22

The proposal in `generalization_prompt_1022.md` is worth a small experiment. The existing Qwen interface can condition its prediction on preceding examples, but useful in-context regression has not been demonstrated. The first experiment should establish whether the fixed model uses support labels correctly, before attempting a broad cross-survey benchmark.

This assessment used repository inspection, checkpoint-file inspection, a CPU tokenizer probe, and a targeted literature review. It contains no new predictive results: no GPU inference, training, or held-out test scoring was performed.

## 1. What the current model actually does

The four survey run manifests identify **Qwen3.5-4B-Base**, not Instruct. Their pipeline is:

1. Serialize a single galaxy's native photometry as text.
2. Run the causal text backbone.
3. Pool the last nonpadding token's final hidden state.
4. Apply the learned photo-z head and softmax over 300 bins spanning redshift 0–4.5.

This is a probabilistic classifier over redshift bins, not autoregressive redshift generation. The supervised redshift is supplied to the loss separately; training inputs do not contain labelled support examples. Consequently, training teaches the head to interpret single-galaxy representations, without explicitly teaching a support-set regression algorithm. Pretraining might provide some relevant ability, but that is a hypothesis.

Because earlier tokens can influence the final query token, labelled-context conditioning is architecturally possible. Whether the resulting representation remains useful to this particular head is an empirical question. Prediction changes alone would establish sensitivity to context, not successful adaptation.

The relevant code is `aion_magnitude/FM_Qwen.py` (`serialize_qwen_feature_row`), `notebooks/qwen_survey_posttraining.py` (`serialization`, `texts_for`, and dataset construction), and `aion_magnitude/qwen_posttraining.py` (`make_text_collator`, `QwenPhotoZModel`).

## 2. A concrete context-length problem

The current collator enables truncation at **512 tokens**. The tokenizer uses right truncation. A CPU probe used the existing HSC cohort's first 16 training rows, its first validation query, and the actual survey serializer. Each support row was formatted as:

```text
Calibration example:
<unchanged native row serialization>
spectroscopic_redshift=<five decimal places>
```

Rows were separated by two newlines. The query was prefixed with `Query galaxy (redshift unknown):`, followed by its unchanged serialization.

| Support examples K | Total tokens | Fits the current 512-token limit? |
|---:|---:|---|
| 0 | 232 | Yes |
| 1 | 481 | Yes |
| 2 | 725 | No |
| 4 | 1,226 | No |
| 16 | 4,237 | No |

The original query row alone occupied 224 tokens. These are measurements for this small probe and template, not worst-case bounds for every row.

With two supports, right truncation already cuts into the query. With four, the query is absent. A naive implementation could therefore return a perfectly valid PDF for the wrong final token sequence.

The local backbone configuration advertises 262,144 positions, so 512 is a training/runtime setting rather than the architecture's nominal ceiling. Increasing the inference limit can accommodate a few supports, but changes sequence length and context structure relative to supervised training. Audit every constructed prompt and fail on overflow; silently switching to left truncation would instead lose support information.

Keep **Ncal**, the number of labelled objects available, separate from **K**, the number shown per query. For example, a 256-object calibration pool can supply four retrieved examples without putting all 256 in context.

For K=0, pass the exact original row text, without the diagnostic query prefix above. First demonstrate numerical equivalence to the original checkpoint inference. Otherwise an apparent calibration effect could simply be a formatting effect.

## 3. Physical meaning of cross-survey transfer

The serializer already includes survey/schema identity, explicit band and measurement names, five-decimal numerical values, `NA`, signed fluxes, uncertainties, and EBV. Preserve these conventions in both supports and queries.

LS grzW1W2 and HSC grizy do not share interchangeable feature positions. Even their similarly named optical filters and flux measurements differ in response and measurement procedure. Equal nanomaggy units do not remove those differences. Retrieval should compare target supports to target queries within the target survey's own feature space.

A small calibration sample might correct a redshift prior, bias, or a local prediction error. It cannot guarantee recovery of information absent from a galaxy's target photometry. Also, the two selected samples differ in footprint, selection, depth, and redshift distribution. An improvement would demonstrate adaptation under their combined shift; attributing it specifically to filter differences needs additional controls.

A later paired-object experiment, using two native photometric views of galaxies excluded from source training, could help isolate the measurement-system contribution. It is unnecessary for the initial interface test.

## 4. The smallest informative pilot

Start with one fixed source checkpoint and a validation-only pilot. Keep checkpoint, head, quantization, native serialization, and output grid fixed.

1. **Restore and reproduce.** Load a stable checkpoint with its matching head and adapter state. Verify K=0 against the existing inference path with evaluation mode and deterministic inputs. Snapshot/hash the checkpoint used; do not load a file while another process is writing it.
2. **Test context use within the source survey.** Use disjoint validation supports and 32–64 fixed validation queries. Begin with K=1, then K=4 under an explicit longer inference limit. This distinguishes failure to use support labels from failure to interpret a new survey.
3. **Run causal controls.** Compare correctly labelled supports with label permutations, supports with labels omitted, and deliberately changed support labels. Repeat a few example orderings. Keep text/token budgets as comparable as practical. The label-omission arm is a format control, not a perfectly token-matched intervention.
4. **Try one cross-survey direction.** Choose the direction whose checkpoint and cohort are readily accessible. Use fixed target validation queries and nested calibration pools of Ncal=4,16,64 initially. Retrieve at most K=4 using only target photometry; random retrieval is a control. Add Ncal=256 and the reverse direction only if warranted.
5. **Compare inexpensive baselines.** Include direct transfer, global residual correction, local residual correction, target-only kNN, and a target-only TabICL regressor using the same labelled calibration pool. Any tuned settings need a separate selection subset or a prespecified rule.

Useful context learning requires more than changed predictions: correct labels should improve held-out predictions relative to appropriate controls, and the effect should recur across calibration draws. Label interventions diagnose dependence on labels, but dependence alone is insufficient. A null result in this small pilot is a reason to deprioritize this interface, not proof that Qwen cannot learn in context under any training or serialization.

Thirty-two queries are sufficient for plumbing and sensitivity checks, not a reliable outlier-rate or cosmological-calibration claim. If an approach looks promising, enlarge the validation evaluation and use multiple calibration draws, paired comparisons on identical queries, and uncertainty intervals before a final test.

## 5. Baselines and attribution need precise names

Keep three mechanisms distinct:

- **Fixed-model context conditioning:** support data enter the Qwen forward pass, with all model parameters fixed.
- **External calibration:** a bias estimate, residual interpolator, or other algorithm adjusts the output. Model weights stay fixed, but calibration statistics or auxiliary parameters can still be fitted.
- **Parameter adaptation:** the head or backbone is optimized on target labels.

The prompt's bias-correction row is correct if “model parameters” means the source neural network. It should not imply that estimating a bias involves no fitting whatsoever. Similarly, TabICL's `.fit()` API must be distinguished from gradient training of its pretrained network.

The strongest practical question is whether source-trained Qwen plus calibration beats target-only predictors given the same target labels. A frozen-backbone Qwen with a source-trained head is also useful, but only isolates the value of backbone adaptation: its head still contains source-label information. A random untrained head is not a meaningful no-source predictive baseline. Do not attach the current head to an unrelated Instruct checkpoint and interpret the comparison as controlled.

If external residual correction wins and correct-label prompts do not, report successful calibration alongside failed direct context conditioning. Do not attribute the external algorithm's gains to Qwen learning in context.

## 6. Data and checkpoint availability in this session

At inspection, none of the four runs had its method-specific final `qlora/result.pt` or `dora/result.pt` under the figure directory. All four did have intermediate checkpoint files in their `run_contract.json` checkpoint directories:

| Run | Latest update filename observed |
|---|---|
| DESI–Legacy QLoRA | `checkpoint-update-0017700.pt` |
| DESI–Legacy DoRA | `checkpoint-update-0017300.pt` |
| GalaxiesML-Spec QLoRA | `checkpoint-update-0011600.pt` |
| GalaxiesML-Spec DoRA | `checkpoint-update-0007100.pt` |

These filenames establish availability, not checkpoint quality, completion, or current process status. An immutable intermediate snapshot could support an explicitly labelled preliminary pilot without stopping a run. Prefer a completed, validation-selected checkpoint for final conclusions.

The HSC cache referenced by its manifest exists here. The DESI–Legacy cache path recorded in its manifest does not exist in this session, and no copy was found in the checked local cache roots. Therefore **the proposed exact-TARGETID overlap audit was not completed**; no leakage-free cross-survey cohort has been established by this assessment. Do not rebuild the large catalogue merely to finish this conceptual review.

Before a cross-survey measurement, intersect the actual source-exposed object IDs with the selected target calibration/query IDs. Conservatively exclude source training objects and objects used for source checkpoint selection from target queries. Calibration and queries must also be disjoint by physical object. Exact TARGETID matches are a useful first check, but alternate IDs can require a targeted positional check. Preserve integer ID precision. No full-sky crossmatch is needed for the small pilot.

## 7. Probabilistic evaluation

Use the same 300-bin grid for all Qwen arms. Report cross-entropy/log score and CRPS alongside NMAD, bias, outliers, and coverage. A translated point prediction is not automatically a calibrated PDF: external corrections must specify how probability mass moves, including the treatment of mass reaching grid boundaries.

There is an existing evaluation detail to handle explicitly. `aion_magnitude/metrics.py::pit_values` returns cumulative mass through the entire bin containing the true redshift. This upper-bin CDF is discretization-biased as a continuous PIT diagnostic. For a piecewise-uniform PDF, interpolate within the bin; for a genuinely categorical interpretation, use a reproducibly randomized PIT. Existing quantiles are bin-center quantiles and also have finite-resolution effects. Keep legacy metrics for comparability and add clearly identified diagnostics rather than silently changing historical results.

TabICL's regression interface exposes quantiles. Do not infer a unique density or directly comparable log score from a few quantiles without documenting a reconstruction. Begin with its supported point/interval metrics; add distribution comparisons only through a defensible common representation.

Tomographic N(z) calibration and individual-galaxy photo-z accuracy are related but distinct outcomes. Neither a lower NMAD nor a better global N(z) establishes all the other outcomes. Keep final test objects untouched during method selection.

## 8. Literature and potential contribution

Generic tabular in-context prediction and retrieval of labelled rows are established ideas. Wen et al. explicitly study retrieval-augmented tabular LLMs with specialized post-training; their work reinforces the need to distinguish an architecture that accepts examples from one trained to exploit them. [Scalable In-Context Learning on Tabular Data via Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2502.03147).

The original TabICL paper demonstrates a model explicitly pretrained for tabular context learning, initially for classification. The current authors' implementation also supplies a regression estimator and quantile predictions, making it a relevant matched-label-budget baseline. [Original TabICL paper](https://arxiv.org/abs/2502.05564), [official regression implementation](https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_sklearn/regressor.py).

Photo-z transfer learning and correction for spectroscopic/photometric distribution differences also predate this proposal. Deepz uses simulation-to-observation transfer, while StratLearn with hierarchical modeling addresses selection differences and tomographic calibration. These are relevant precedents, not demonstrations of this exact fixed-Qwen/native-band experiment. [Deepz](https://arxiv.org/abs/2004.07979), [StratLearn photo-z calibration](https://arxiv.org/abs/2401.04687).

My assessment is that a plausible contribution would be a controlled measurement of **label-efficient, probabilistic transfer between native survey photometric systems**, including when a source-posttrained general-purpose model adds value beyond target-only and classical calibration methods. This targeted review does not establish a first-of-its-kind claim. A small improvement over zero-shot transfer alone would be weak evidence of novelty.

For Paper I, prioritize a trustworthy cross-survey benchmark and clear mechanism attribution. Direct context conditioning can be one economical arm, including a negative result. Episodic source-only training to teach support/query regression would be a separate stage if the current interface fails; target calibration after that stage can still keep weights fixed, but the acquisition of the context-learning ability must be reported as training.

## 9. Suggested refinement to the task

Keep the main hypothesis, but make the first decision gate explicit: **Does the existing fixed checkpoint use labelled support examples to improve independent predictions, beyond formatting and context perturbation controls?** Only then expand calibration sizes, survey directions, and scientific interpretation. If it fails, establish the external-calibration benchmark before deciding whether episodic training is worth the cost.

The prompt's final execution sentence is incomplete (`Begin by independently examining the actual`). This does not prevent the assessment above, but should be completed before handing the document off as an execution specification.
