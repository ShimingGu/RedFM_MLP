# Using the archived Qwen QLoRA photo-z model

This directory contains the trained QLoRA photo-z artifact from the
`qwen-qwen_posttraining_comparison-e10` experiment. It predicts a discrete
redshift probability distribution from eleven catalogue AB magnitudes.

## Artifact inventory

- `adapter/adapter_model.safetensors`: QLoRA adapter weights.
- `adapter/adapter_config.json`: PEFT configuration and base-model provenance.
- `photoz_head.pt`: nonlinear head mapping the final Qwen token to 300 redshift bins.
- `result.pt`: training history, validation/test predictions, metrics, and metadata.

The adapter and `photoz_head.pt` must be used together. This is not a standalone
Qwen checkpoint: it also requires the matching `Qwen3.5-4B-Base` base model.

## Training-time contract

- Base model: `/arc/home/gsm/hf_models/Qwen3.5-4B-Base`
- Inputs: magnitudes only; no morphology, image cutouts, or object metadata.
- Catalogue/domain: CLAUDS COSMOS.
- Feature order: `u_mag`, `u_star_mag`, `g_mag`, `r_mag`, `i_mag`, `z_mag`,
  `y_mag`, `Y_mag`, `J_mag`, `H_mag`, `Ks_mag`.
- Units: AB magnitudes.
- Serialization: five decimal places, schema `clauds_all_magnitude_v1`, prefix
  `galaxy all_magnitudes_ab`.
- Tokenization: padding and truncation, maximum length 2048.
- Representation: final non-padding Qwen token; no embedding normalization.
- Output: 300 bins spanning z = 0.005000030156224966 to
  5.992499828338623.

Missing or non-finite values must be replaced with the training-split medians:

| Feature | Fill value |
|---|---:|
| `u_mag` | 25.977767944335938 |
| `u_star_mag` | 25.803855895996094 |
| `g_mag` | 25.46443271636963 |
| `r_mag` | 25.128427505493164 |
| `i_mag` | 24.86415672302246 |
| `z_mag` | 24.64280128479004 |
| `y_mag` | 24.341413497924805 |
| `Y_mag` | 24.810382843017578 |
| `J_mag` | 24.58324432373047 |
| `H_mag` | 24.365999221801758 |
| `Ks_mag` | 24.095584869384766 |

Do not reorder the features or substitute fluxes for AB magnitudes. Python
dictionaries preserve insertion order, and the text order is part of the model
input contract.

## Minimal batch inference

Run from a checkout of the `RedFM_MLP` repository using its Pixi environment.
The example uses one visible CUDA slice and 4-bit NF4 loading.

```python
import math
import os
from pathlib import Path

import torch
from peft import PeftModel

from aion_magnitude.FM_Qwen import (
    QwenSerializationConfig,
    load_frozen_qwen,
    serialize_qwen_feature_row,
)
from aion_magnitude.metrics import predict_photoz_from_logits
from aion_magnitude.qwen_posttraining import (
    QwenPhotoZModel,
    qwen_hidden_size,
)
from aion_magnitude.utils import make_redshift_grid


ARTIFACT = Path(
    "/arc/projects/ots/Cosmic_Imprint_of_Time/posttrainings/qwen-qlora"
)
BASE_MODEL = Path(
    os.environ.get(
        "AION_QWEN_BASE",
        "/arc/home/gsm/hf_models/Qwen3.5-4B-Base",
    )
)
DEVICE = torch.device("cuda")

FEATURES = (
    "u_mag", "u_star_mag", "g_mag", "r_mag", "i_mag", "z_mag",
    "y_mag", "Y_mag", "J_mag", "H_mag", "Ks_mag",
)
FILLS = {
    "u_mag": 25.977767944335938,
    "u_star_mag": 25.803855895996094,
    "g_mag": 25.46443271636963,
    "r_mag": 25.128427505493164,
    "i_mag": 24.86415672302246,
    "z_mag": 24.64280128479004,
    "y_mag": 24.341413497924805,
    "Y_mag": 24.810382843017578,
    "J_mag": 24.58324432373047,
    "H_mag": 24.365999221801758,
    "Ks_mag": 24.095584869384766,
}

SERIALIZATION = QwenSerializationConfig(
    schema_name="clauds_all_magnitude_v1",
    decimals=5,
    include_hsc_grizy=False,
    include_object_metadata=False,
    prefix="galaxy all_magnitudes_ab",
)


def impute_and_serialize(row):
    ordered = {}
    for name in FEATURES:
        raw = row.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = math.nan
        ordered[name] = value if math.isfinite(value) else FILLS[name]
    return serialize_qwen_feature_row(ordered, serialization=SERIALIZATION)


def load_model():
    base, tokenizer = load_frozen_qwen(
        BASE_MODEL,
        device=DEVICE,
        load_in_4bit=True,
        torch_dtype="bf16",
        local_files_only=True,
        trust_remote_code=True,
    )
    qwen = PeftModel.from_pretrained(
        base,
        ARTIFACT / "adapter",
        is_trainable=False,
    )
    model = QwenPhotoZModel(
        qwen,
        hidden_size=qwen_hidden_size(qwen),
        n_z_bins=300,
        head_hidden_dim=256,
        pooling="last",
    ).to(DEVICE)
    head_state = torch.load(
        ARTIFACT / "photoz_head.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.photoz_head.load_state_dict(head_state)
    model.eval()
    return model, tokenizer


def predict(rows, model, tokenizer):
    texts = [impute_and_serialize(row) for row in rows]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )
    encoded = {name: value.to(DEVICE) for name, value in encoded.items()}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**encoded)

    _, centers = make_redshift_grid(
        0.005000030156224966,
        5.992499828338623,
        300,
    )
    return predict_photoz_from_logits(logits.float().cpu(), centers=centers)


model, tokenizer = load_model()

# Replace these illustrative values with measured AB magnitudes.
rows = [
    {
        "u_mag": 25.3,
        "u_star_mag": None,
        "g_mag": 24.8,
        "r_mag": 24.2,
        "i_mag": 23.9,
        "z_mag": 23.7,
        "y_mag": 23.6,
        "Y_mag": 23.5,
        "J_mag": 23.3,
        "H_mag": 23.1,
        "Ks_mag": 22.9,
    }
]

prediction = predict(rows, model, tokenizer)
for index in range(len(rows)):
    print(
        {
            "z_mean": float(prediction["z_mean"][index]),
            "z_mode": float(prediction["z_mode"][index]),
            "z_p16": float(prediction["z_p16"][index]),
            "z_p50": float(prediction["z_p50"][index]),
            "z_p84": float(prediction["z_p84"][index]),
        }
    )

# prediction["pz"][index] is the full normalized 300-bin redshift PDF.
```

For example, save the code as `infer_qwen_qlora.py` and run:

```bash
cd /path/to/RedFM_MLP
CUDA_VISIBLE_DEVICES=0 pixi run python infer_qwen_qlora.py
```

A small inference batch is safest on a 12 GB MIG slice. Start with 1-8 objects
and increase only after checking GPU memory.

## Output interpretation

- `pz`: normalized probability mass over the 300 bin centers.
- `z_mean`: posterior mean; this was the principal point prediction used by the
  repository metrics.
- `z_mode`: center of the highest-probability bin.
- `z_p16`, `z_p50`, `z_p84`: binned posterior quantiles.

Keep the bin centers with `pz` when saving predictions. Values outside the
training range cannot be represented and will accumulate at an edge bin.

## Validation and limitations

The archived run used 300,000 randomly sampled COSMOS objects with seed 42:
60,000 train, 15,000 validation, and 225,000 test. Its test metrics included
NMAD 0.07294 and catastrophic-outlier fraction 0.18164. These are in-domain
experimental results, not guaranteed performance for another field, survey,
photometric calibration, selection function, or magnitude-depth regime.

Before scientific use on a new catalogue:

1. Confirm every input is an AB magnitude with the same band meaning and
   calibration as the training catalogue.
2. Compare per-band distributions and missingness against the CLAUDS COSMOS
   training domain.
3. Validate bias, NMAD, catastrophic-outlier rate, PIT, and interval coverage on
   a representative spectroscopic subset.
4. Recalibrate the PDF or retrain if the new survey has a significant domain
   shift. Do not use the model as a precision redshift estimator without this
   validation.

## Portability notes

- The adapter was saved with PEFT 0.19.1. Use the repository environment when
  possible.
- The paths recorded inside `result.pt` refer to the original output directory
  and are historical metadata. Load files relative to `ARTIFACT` as shown above.
- If `/arc/home/gsm/hf_models/Qwen3.5-4B-Base` is unavailable, point
  `AION_QWEN_BASE` to an identical local base checkpoint. A different Qwen model
  or revision is not compatible merely because its hidden size is similar.
- `result.pt` is not required for inference; the adapter, head, base model, and
  exact preprocessing contract are required.
