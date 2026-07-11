# AGY 代码审查任务说明：AION / extra-band / MLP 开关真实性检查

日期：2026-07-04  
工作目录：`/Users/shiminggu/Documents/Science/aion_tutorial`

本文档给后续 agent 使用。目标不是继续改模型，也不是解释图像结果，而是逐字阅读当前代码，确认我们前几轮要求的核心功能是否真的被实现，尤其要排除“看起来实现了但实际走 fake path / stale cache / 空 feature”的情况。

请重点阅读这些文件：

```text
aion_magnitude.py
aion_extra_bands.py
clauds_bands.py
aion_mlp_test.ipynb
logs/all_mag_fusion.md
```

备份文件不要改：

```text
aion_magnitude_usable_0703.py
```

## 总体审查目标

请逐项判断以下 claim 是否为真，并给出代码证据：

1. AION embedding 是否真的调用了 pretrained AION encoder，而不是 fake embedding / random embedding / placeholder embedding。
2. `u,u*,Y,J,H,Ks` 是否真的从 catalogue 或 split cache 里读取，并转成 magnitude-like scalar feature。
3. 关闭 AION 时，AION branch 是否真的关闭。
4. 关闭 MLP/tabular feature branch 时，MLP branch 是否真的关闭。
5. `grizy` 是否按设计进入或不进入 MLP：有 AION 时默认不重复进 MLP；无 AION 时默认进入 MLP。
6. 单个 extra band 或多个 extra bands 的开关是否真的影响输入 feature。
7. 是否实现了通过可学习或指定的 linear combination / reweighting 来修改输入 bands 的功能。
8. catalogue 缺少某个 band 时是否有 warning / fill 逻辑，而不是静默失败。
9. AION-only、MLP-only、fusion 三种模式是否真的互相区分。
10. cache 是否可能掩盖新逻辑：旧 cache 是否会被刷新 catalogue-side features，路径是否区分不同 config。
11. notebook 里的 config 是否和代码语义一致。

## 1. AION embedding 是否是真的

需要验证的 claim：

- `use_aion_embedding=True` 时，代码应当加载 `polymathic-ai/aion-base`，并通过真实 AION encoder 生成 embedding。
- 不应使用随机数、全零 tensor、常数 tensor、mock embedding 来冒充 AION embedding。
- `use_aion_embedding=False` 时，允许出现 shape 为 `(N, 0)` 的空 embedding；这是 deliberate no-AION path，不是 fake embedding。

建议阅读位置：

```text
aion_magnitude.py
  load_frozen_aion(...)
  extract_hsc_aion_embedding(...)
  extract_aion_embeddings_to_memory(...)
  build_and_cache_aion_embeddings(...)
```

目前预期证据：

- `load_frozen_aion(...)` 应导入并调用：

```python
from aion.model import AION
from aion.codecs import CodecManager
from aion.modalities import HSCMagG, HSCMagR, HSCMagI, HSCMagZ, HSCMagY
AION.from_pretrained("polymathic-ai/aion-base")
```

- `extract_hsc_aion_embedding(...)` 应把 `g_mag,r_mag,i_mag,z_mag,y_mag` 包装成 AION modalities，然后：

```python
tokens = codec_manager.encode(*modalities)
sequence = aion.encode(tokens, num_encoder_tokens=n_tokens)
embedding = sequence.mean(dim=1)
```

- `extract_aion_embeddings_to_memory(...)` 应在 dataloader 上逐 batch 调用 `extract_hsc_aion_embedding(...)`，最后 `torch.cat(embeddings, dim=0)`。
- `build_and_cache_aion_embeddings(...)` 中：
  - `use_aion_embedding=True` 应走 `load_frozen_aion(...)` 和 `extract_aion_embeddings_to_memory(...)`。
  - `use_aion_embedding=False` 应生成 `torch.empty((len(raw_dataset), 0), dtype=torch.float32)`，metadata 应标记 AION extraction skipped。

需要特别检查：

- 全仓库搜索 `fake`, `random`, `randn`, `zeros`, `empty`，确认没有在 AION-on path 里生成假 embedding。
- 确认 `torch.empty((N, 0))` 只在 `use_aion_embedding=False` path 出现。
- 确认 cached product 的 `metadata["use_aion_embedding"]` 与 `aion_embedding.shape[1]` 一致。
- 如果 cache 文件已经存在，确认 `cache_path.exists() and not force_recompute_embeddings` 的分支不会复用错误维度或错误 metadata 的旧 AION embedding。

通过标准：

- AION-on path 中必须能追踪到真实 `AION.from_pretrained` -> `codec_manager.encode` -> `aion.encode` -> pooled embedding。
- AION-off path 中空 embedding 是预期行为，并且 tabular model 不应使用 AION embedding。

## 2. `u,u*,Y,J,H,Ks` 是否真的被读取

需要验证的 claim：

- `u` 来自 `FLUX_CMODEL_MegaCam-u`。
- `u*` 来自 `FLUX_CMODEL_MegaCam-uS`。
- `Y,J,H,Ks` 来自 `FLUX_CMODEL_VIRCAM-Y/J/H/Ks`。
- 这些 flux 应转成 AB magnitude-like scalar，再进入 `extra_features`。

建议阅读位置：

```text
clauds_bands.py
  BAND_FLUX_COLUMNS
  OPTIONAL_EXTRA_BAND_FLUX_COLUMNS
  ALL_BAND_FLUX_COLUMNS
  OPTIONAL_EXTRA_FLAG_COLUMNS
  ALL_FLAG_COLUMNS

aion_extra_bands.py
  DEFAULT_EXTRA_BANDS
  EXTRA_BAND_ALIASES
  extract_extra_band_magnitudes_from_table(...)
  extract_extra_band_magnitudes_from_split_arrays(...)
  make_extra_band_feature_matrix(...)

aion_magnitude.py
  build_extra_feature_matrix_from_table(...)
  build_raw_clauds_photoz_dataset(...)
```

目前预期列映射：

| band | canonical name | FITS flux column |
|---|---|---|
| `u` | `u` | `FLUX_CMODEL_MegaCam-u` |
| `u*` | `u_star` | `FLUX_CMODEL_MegaCam-uS` |
| `Y` | `Y` | `FLUX_CMODEL_VIRCAM-Y` |
| `J` | `J` | `FLUX_CMODEL_VIRCAM-J` |
| `H` | `H` | `FLUX_CMODEL_VIRCAM-H` |
| `Ks` | `Ks` | `FLUX_CMODEL_VIRCAM-Ks` |

需要特别检查：

- `DEFAULT_EXTRA_BANDS` 是否是：

```python
("u", "u_star", "Y", "J", "H", "Ks")
```

- `resolve_extra_band_names(...)` 是否支持 `u*`, `u_star`, `uS`, `Ks`, `ks` 等 alias。
- `extract_extra_band_magnitudes_from_table(...)` 是否通过 `ALL_BAND_FLUX_COLUMNS` 找到真实 FITS 列。
- flux-to-mag 是否使用：

```python
mag = -2.5 * log10(flux) + mag_zero_point
```

- 质量 flag 是否用于 valid mask：
  - `hasBadPhotometry_*`
  - `isNoData_*`
  - `notObserved_*`
- `make_extra_band_feature_matrix(...)` 是否真的返回 `extra_features` 和 `feature_names`。
- `build_raw_clauds_photoz_dataset(...)` 是否把这些 `extra_features` 放进最终 `CLAUDSPhotoZDataset`。

通过标准：

- 选择 `extra_bands=("u", "u_star", "Y", "J", "H", "Ks")` 时，最终 `feature_names` 至少应包含：

```text
u_mag, u_star_mag, Y_mag, J_mag, H_mag, Ks_mag
```

- 如果 `include_valid_flags=True`，还应包含：

```text
u_mag_valid, u_star_mag_valid, Y_mag_valid, J_mag_valid, H_mag_valid, Ks_mag_valid
```

## 3. 缺 band 时是否有 warning 和 fill

需要验证的 claim：

- COSMOS 有 `u,u*,Y,J,H,Ks`。
- DEEP23 目前只有 `u` extra band。
- 如果 catalogue 没有某个 selected band，代码应 warning，并填充该列，而不是 crash 或静默忽略。

建议阅读位置：

```text
aion_extra_bands.py
  _warn_missing_band(...)
  extract_extra_band_magnitudes_from_table(...)
  extract_extra_band_magnitudes_from_split_arrays(...)
  _fill_magnitude_columns(...)
  make_extra_band_feature_matrix(...)

clauds_bands.py
  split cache optional column handling
```

需要特别检查：

- 缺少 FITS/split column 时，是否：

```python
mag = np.full(n_rows, np.nan, dtype=np.float32)
valid = np.zeros(n_rows, dtype=bool)
```

- fill policy 是否支持：

```text
median
max_valid
numeric value
```

- 全部 invalid 时是否填 0.0，而不是崩溃。
- `EXTRA_BANDS=()` 是否返回 `(N, 0)` 空 feature matrix，而不是触发 `torch.stack([])`。

通过标准：

- missing band path 有 warning。
- missing band path 的 valid count 为 0。
- feature 维度仍与 selected bands 对齐。

## 4. 关闭 AION 是否真的关闭

需要验证的 claim：

- `use_aion_embedding=False` 时，不应加载 AION，不应调用 `load_frozen_aion(...)`，不应调用 `extract_aion_embeddings_to_memory(...)`。
- no-AION 模式应只训练 `tabular` model。

建议阅读位置：

```text
aion_magnitude.py
  build_and_cache_aion_embeddings(...)
  AIONMagnitudeConfig.normalized(...)
  train_single_baseline(...)
  build_baseline_model(...)
  _default_evaluation_model_kind(...)
  run_training_and_evaluation(...)
```

目前预期行为：

- `AIONMagnitudeConfig.normalized(...)` 中，若 `use_aion_embedding=False`，应强制：

```python
model_kinds=("tabular",)
aion_input_bands=tuple()
```

- `build_and_cache_aion_embeddings(...)` 中，若 `use_aion_embedding=False`，应创建：

```python
aion_embeddings = torch.empty((len(raw_dataset), 0), dtype=torch.float32)
```

- `train_single_baseline(...)` 中，如果请求 `aion` 或 `fusion` 但 `aion_dim == 0`，应 raise。

通过标准：

- no-AION config 的 product 中 `aion_embedding.shape[1] == 0`。
- no-AION config 的 `model_kinds == ("tabular",)`。
- no-AION config 的 evaluated model kind 默认为 `tabular`。

## 5. 关闭 MLP 是否真的关闭

需要验证的 claim：

- `use_mlp_features=False` 表示 AION-only。
- 这时不应使用 extra-band MLP branch。
- 这时 `tabular` 和 `fusion` 不应被训练。

建议阅读位置：

```text
aion_magnitude.py
  build_raw_clauds_photoz_dataset(...)
  AIONMagnitudeConfig.normalized(...)
  train_single_baseline(...)
  build_baseline_model(...)
```

目前预期行为：

- `use_mlp_features=False` 要求 `use_aion_embedding=True`，否则 raise。
- config normalization 应强制：

```python
model_kinds=("aion",)
include_grizy_in_mlp=False
```

- `build_raw_clauds_photoz_dataset(...)` 中应创建：

```python
extra_features = torch.empty((n_rows, 0), dtype=torch.float32)
feature_names = []
```

- `train_single_baseline(...)` 中，如果 `model_kind in {"tabular", "fusion"}` 且 `extra_feature_dim == 0`，应 raise。

通过标准：

- AION-only config 的 product 中 `extra_features.shape[1] == 0`。
- AION-only config 的 `model_kinds == ("aion",)`。
- AION-only config 的 evaluated model kind 默认为 `aion`。

## 6. `grizy` 是否按设计进入 MLP

需要验证的 claim：

- AION branch 固定吃 `g,r,i,z,y`。
- 有 AION 时，默认不把 direct `g,r,i,z,y` scalar 重复放入 MLP。
- 无 AION 时，默认把 direct `g,r,i,z,y` scalar 放入 MLP，保证 tabular baseline 有 grizy features。
- 用户仍可用 `include_grizy_in_mlp=True/False` 显式控制。

建议阅读位置：

```text
aion_magnitude.py
  HSC_AION_BANDS
  build_hsc_aion_features_from_table(...)
  build_grizy_mlp_feature_matrix(...)
  resolve_include_grizy_in_mlp(...)
  build_raw_clauds_photoz_dataset(...)
  AIONMagnitudeConfig.normalized(...)
```

目前预期行为：

- `HSC_AION_BANDS` 应为：

```python
["g", "r", "i", "z", "y"]
```

- `build_grizy_mlp_feature_matrix(...)` 应返回：

```text
g_mag, r_mag, i_mag, z_mag, y_mag
```

- `resolve_include_grizy_in_mlp(None, use_aion_embedding=True, ...)` 应返回 `False`。
- `resolve_include_grizy_in_mlp(None, use_aion_embedding=False, ...)` 应返回 `True`。
- `build_raw_clauds_photoz_dataset(...)` 中，若 `include_grizy_in_mlp=True`，应把 grizy features `torch.cat` 到 extra features 前面。

特殊边界：

- 如果 `use_aion_embedding=True`、`extra_bands=()`、`include_grizy_in_mlp=None/False`，则没有 MLP features。这时 config 应自动转为 AION-only：

```python
use_mlp_features=False
model_kinds=("aion",)
```

通过标准：

- `uu*grizyYJHKs-MLP` 应包含 direct `g,r,i,z,y` scalar features。
- `uu*YJHKs-MLP + grizy-AION` 应不包含 direct `g,r,i,z,y` scalar features，但应包含 AION embedding。
- `grizy-only tabular` 应在 no-AION 模式下包含 `g,r,i,z,y` scalar features。

## 7. AION input bands 是否可关闭

需要验证的 claim：

- 当前版本暂时不允许单独关闭 AION 输入里的某个 `grizy` band。
- 如果用户试图改 `aion_input_bands`，代码应 warning，并恢复 full `grizy`。

建议阅读位置：

```text
aion_magnitude.py
  build_and_cache_aion_embeddings(...)
  AIONMagnitudeConfig.normalized(...)
```

目前预期 warning 文本应表达：

```text
Disabling individual HSC grizy bands is not currently supported because
the frozen AION embedding expects the full grizy input...
```

通过标准：

- AION-on path 始终用 full `g,r,i,z,y` modalities。
- 任何非 full `aion_input_bands` 只产生 warning，不真的改变 AION input。

## 8. band linear-combination / reweighting 输入改造是否实现

需要验证的 claim：

- 是否存在一个功能，可以通过手动指定或可学习的 linear combination / reweighting，把原始 bands 改造成 modified inputs。
- 例如，把若干 magnitude columns 组合成新的 band-like scalar，再送入 AION 或 MLP：

```text
modified_band_k = sum_i weight_{k,i} * band_i + bias_k
```

- 这个功能如果存在，应该能明确控制：
  - 使用哪些原始 bands；
  - 每个 band 的权重；
  - modified inputs 是进入 AION branch、MLP branch，还是两者都进入；
  - 权重是固定 config 参数，还是训练中可学习参数；
  - metadata / feature_names 是否记录 modified input 的定义。

建议阅读位置：

```text
aion_magnitude.py
aion_extra_bands.py
clauds_bands.py
aion_mlp_test.ipynb
```

建议搜索关键词：

```text
linear
combination
reweight
weighted
band_weight
modified
synthetic
mix
projection
```

通过标准：

- 如果找到清晰实现，应说明函数名、config 字段、数据流向和 feature_names / metadata 记录方式。
- 如果只找到普通 feature concatenation，例如 `torch.cat([grizy_features, extra_features], dim=1)`，这不算 linear-combination / reweighting 输入改造。


## 9. 三种模型模式是否真的区分

需要验证的 claim：

- `tabular` 只使用 `extra_features`。
- `aion` 只使用 `aion_embedding`。
- `fusion` 同时使用 `aion_embedding` 和 encoded `extra_features`。

建议阅读位置：

```text
aion_magnitude.py
  TabularPhotoZModel
  AIONOnlyPhotoZModel
  CLAUDSPhotoZModel
  forward_model(...)
  build_baseline_model(...)
  train_single_baseline(...)
```

目前预期行为：

- `TabularPhotoZModel.forward(extra_features)` 只接收 MLP features。
- `AIONOnlyPhotoZModel.forward(aion_embedding)` 只接收 AION features。
- `CLAUDSPhotoZModel.forward(aion_embedding, extra_features)` 应把两者 concat。
- `forward_model(...)` 应按 `model_kind` 分发。

通过标准：

- 三种模型的 forward signature 和 `forward_model(...)` 调用一致。
- 空 feature branch 时会被 guard 拦住，不会 silently train。

## 10. cache 是否可能掩盖问题

这是最容易误判的地方。请认真读 cache path 和 cache refresh 逻辑。

需要验证的 claim：

- AION embedding cache 可以复用。
- 但 catalogue-side tensors 应能按当前 config 刷新：
  - `extra_features`
  - `feature_names`
  - `z_spec`
  - `redshift_reference`
  - extra-band metadata
- baseline output dir 应区分不同 extra-band / grizy-in-MLP / no-AION / AION-only config。

建议阅读位置：

```text
aion_magnitude.py
  make_cache_run_tag(...)
  resolve_training_paths(...)
  build_and_cache_aion_embeddings(...)
  refresh_cached_product_catalogue_features(...)
  save_cached_product(...)
  load_cached_product(...)
```

需要特别检查：

- `cache_path` 是否只按 catalogue / zeropoint / row count 区分，而不一定按 feature config 区分。
- 如果 `cache_path` 已存在，是否会调用 `refresh_cached_product_catalogue_features(...)`。
- `baseline_output_dir` 是否包含：
  - selected extra bands tag；
  - `grizy_` tag；
  - `noaion` tag；
  - `aiononly` tag。
- cached product 的 `metadata["feature_names"]` 是否和 `product["extra_features"].shape[1]` 一致。
- 旧 cache 是否可能保留错误 `aion_embedding` shape 或 stale metadata。

建议审查命令：

```bash
./aion_env/bin/python - <<'PY'
import torch
from pathlib import Path
for path in sorted(Path("cache").glob("*.pt")):
    try:
        p = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        continue
    if isinstance(p, dict) and "aion_embedding" in p and "extra_features" in p:
        print(path)
        print("  aion:", tuple(p["aion_embedding"].shape))
        print("  extra:", tuple(p["extra_features"].shape))
        print("  feature_names:", p.get("feature_names"))
        print("  metadata use_aion:", p.get("metadata", {}).get("use_aion_embedding"))
        print("  metadata use_mlp:", p.get("metadata", {}).get("use_mlp_features"))
        print("  metadata include_grizy:", p.get("metadata", {}).get("include_grizy_in_mlp"))
PY
```

通过标准：

- 对于每个实验，cache 中 feature 维度、feature names、metadata 和 config 一致。
- 不同 baseline 输出目录不会互相覆盖。

## 11. notebook config 是否表达正确实验

需要验证的 claim：

- `aion_mlp_test.ipynb` 暴露了核心开关。
- comparison cell 中 config label 和实际 config 顺序一致。
- 不应出现 label 写 `grizy` 但实际没有 grizy scalar / AION 的情况。

建议阅读位置：

```text
aion_mlp_test.ipynb
```

重点检查变量：

```python
EXTRA_BANDS
USE_AION_EMBEDDING
USE_MLP_FEATURES
INCLUDE_GRIZY_IN_MLP
AION_INPUT_BANDS

EXTRA_BANDS2
USE_AION_EMBEDDING2
USE_MLP_FEATURES2
INCLUDE_GRIZY_IN_MLP2
AION_INPUT_BANDS2
comparison_labels
run_config_pair(...)
```

通过标准：

- `comparison_labels` 的顺序必须和 `run_config_pair(config_1, config_2)` 的顺序一致。
- `EXTRA_BANDS2 = ()` 表示没有 `u/u*/Y/J/H/Ks` extra-band columns。
- `EXTRA_BANDS2 = None` 表示使用默认 extra bands，不是“无 extra bands”。
- 若 notebook 使用 AION-only grizy comparison，应明确：

```python
USE_AION_EMBEDDING2 = True
USE_MLP_FEATURES2 = False
EXTRA_BANDS2 = ()
```

- 若 notebook 使用 grizy-only tabular comparison，应明确：

```python
USE_AION_EMBEDDING2 = False
USE_MLP_FEATURES2 = True
EXTRA_BANDS2 = ()
INCLUDE_GRIZY_IN_MLP2 = None
```

## 12. pair comparison / plotting 是否包含

这不是物理功能核心，但属于本轮实现要求。请确认这些 helper 存在且能接收两个 config / 两个 evaluation：

```text
compare_zpred_vs_zphot(...)
compare_pit_histogram(...)
compare_redshift_probability_distribution(...)
compare_nz_lensing_alike(...)
compare_config_loss(...)
run_config_pair(...)
```

通过标准：

- scatter 和 PIT 是左右两个 subplot。
- redshift probability distribution 和 tomographic `n(z)` 是上下两个 subplot。
- loss comparison 能画两个 config 的 train/val loss。

## 13. 建议最小 smoke tests

如果允许执行轻量测试，不要跑完整训练；先跑以下检查。

### Python syntax

```bash
./aion_env/bin/python -m py_compile clauds_bands.py aion_extra_bands.py aion_magnitude.py
```

### Extra-band feature matrix

```bash
./aion_env/bin/python - <<'PY'
import numpy as np
from aion_extra_bands import make_extra_band_feature_matrix

features, names, metadata = make_extra_band_feature_matrix(
    np.empty((5, 0), dtype=np.float32),
    np.empty((5, 0), dtype=bool),
    extra_bands=(),
)
print(features.shape, names, metadata["extra_bands"])
PY
```

预期：

```text
torch.Size([5, 0]) [] []
```

### Config semantics

```bash
./aion_env/bin/python - <<'PY'
import aion_magnitude as am

cases = {
    "no_aion_grizy_tabular": dict(
        use_aion_embedding=False,
        use_mlp_features=True,
        include_grizy_in_mlp=None,
        extra_bands=(),
        model_kinds=("tabular", "aion", "fusion"),
    ),
    "aion_only": dict(
        use_aion_embedding=True,
        use_mlp_features=False,
        include_grizy_in_mlp=None,
        extra_bands=(),
        model_kinds=("tabular", "aion", "fusion"),
    ),
    "aion_no_extra_auto_aion_only": dict(
        use_aion_embedding=True,
        use_mlp_features=True,
        include_grizy_in_mlp=None,
        extra_bands=(),
        model_kinds=("tabular", "aion", "fusion"),
    ),
}

for name, kwargs in cases.items():
    c = am.make_magnitude_config(**kwargs)
    print(name)
    print("  use_aion_embedding:", c.use_aion_embedding)
    print("  use_mlp_features:", c.use_mlp_features)
    print("  include_grizy_in_mlp:", c.include_grizy_in_mlp)
    print("  model_kinds:", c.model_kinds)
    print("  default model:", am._default_evaluation_model_kind(c))
    print("  paths:", am.resolve_training_paths(c)["experiment_tag"])
PY
```

预期大意：

- `no_aion_grizy_tabular` -> tabular-only，路径含 `grizy_noextra_noaion`。
- `aion_only` -> AION-only，路径含 `aiononly`。
- `aion_no_extra_auto_aion_only` -> 自动收敛为 AION-only，路径含 `aiononly`。

## 14. 最终报告格式建议

请后续 agent 输出类似下面的结构：

```text
结论：
- AION embedding: 通过 / 不通过 / 部分通过
- extra bands: 通过 / 不通过 / 部分通过
- AION off switch: 通过 / 不通过 / 部分通过
- MLP off switch: 通过 / 不通过 / 部分通过
- grizy-in-MLP default: 通过 / 不通过 / 部分通过
- band linear-combination / reweighting modified inputs: 通过 / 不通过 / 部分通过
- config/cache separation: 通过 / 不通过 / 部分通过

主要证据：
- 文件名 + 函数名 + 关键代码路径

发现的问题：
- 若有，按严重程度排列

建议修复：
- 只列必要修复，不做无关重构
```

## 当前审查假设

截至本文档创建时，基于前一轮人工检查和 smoke tests，我们的工作假设是：

- `uu*grizyYJHKs-MLP` 是当前最强 empirical baseline。
- frozen AION `grizy` embedding 在当前 late-fusion 实现里尚未带来可测提升。
- 这不等价于 AION 本身无用；下一轮应专门调查 AION branch 为什么没有帮助。

这份文档只用于确认实现是否真实存在，不用于证明科学结论。
