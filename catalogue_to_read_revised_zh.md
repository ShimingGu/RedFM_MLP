# Catalogue 与图像数据说明

> 本文件仅说明数据位置、目录谱系、列含义、单位、缺失值约定、图像结构、对象—图像映射方式以及潜在的数据泄漏风险。  
> 本文件不指定模型家族、特征集合、预处理方法、训练目标、实验路线，也不建议复用任何既有建模实现。  
> 文中的“基础”“派生”“单波段代理”“多波段”等词只描述数据来源和生成关系，不代表优先级或预期性能。

数据内容已于 **2026-07-28** 对照在线文件核验。

## 更新版多波段形态 catalogue

原文件
`COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits` 保留不变；本文件中
已有的可用行数和字段统计描述该原文件。修正 mask/variance 传播、AION
校准拆分、事务式 resume、概率有效性和 cache provenance 后的新输出使用：

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband_updated.fits
```

更新版中的 AION `p_*` 只在 held-out DES Galaxy10 上做温度校准；对
CLAUDS/HSC catalogue 它们仍是未经 target-domain 校准的跨巡天 transfer
score。`u` 继续使用 HSC-G codec proxy，HSC-Y 没有匹配的 Galaxy10 训练波段。
更新版完成前，不应把下文原 multiband 文件的 availability 计数套用于
`multiband_updated`。

---

## 1. 数据根目录与使用约束

规范数据位于：

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/
```

这些 catalogue 和巡天图像体积较大，应直接在原位置读取。不要把整份 catalogue 或图像目录复制到当前代码仓库或 `/scratch`。

---

## 2. Catalogue 位置与谱系

| Catalogue | 行数 | 列数 | 约大小 | 数据内容 |
| --- | ---: | ---: | ---: | --- |
| `/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros.fits` | 5,474,883 | 235 | 6.80 GB | COSMOS 场的基础 catalogue，包含测光、质量标记、分类字段和 Phosphoros photo-z 输出。 |
| `/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological.fits` | 5,474,883 | 243 | 6.94 GB | 基础 COSMOS catalogue 加 8 个无波段后缀的形态字段。这些字段使用同一个 CLAUDS `u/uS` 空间模板。 |
| `/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits` | 5,474,883 | 307 | 8.18 GB | 基础 COSMOS catalogue 加 72 个分波段形态字段；分别来自实际的 `u` 和 HSC `g,r,i,z,y` 图像。 |
| `/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/DEEP23-HSCpipe-Phosphoros.fits` | 3,966,735 | 131 | 2.61 GB | 独立的 DEEP2-3 天区 catalogue，含 `u+grizy`；不是 COSMOS 的派生文件，也没有附加形态字段。 |

已核验的谱系关系：

```text
COSMOS-HSCpipe-Phosphoros.fits
├── + 8 个共享空间模板产生的无后缀形态字段
│   └── COSMOS-HSCpipe-Phosphoros_morphological.fits
└── + 72 个实际分波段图像产生的形态字段
    └── COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits

DEEP23-HSCpipe-Phosphoros.fits
└── 独立天区、独立对象；不与 COSMOS 按行对齐
```

两个 COSMOS 形态 catalogue 均保留基础文件的全部 5,474,883 行，`ID` 值及其顺序也保持不变。

即便如此，在新生成的子样本或中间表之间做显式连接时，仍应使用 `ID`，不要默认行顺序始终不变。

不要按行号连接 DEEP23 与 COSMOS。跨天区组合时，安全身份键应写成：

```text
(field, ID)
```

因为不同天区的 `ID` 不保证全局唯一。原始 FITS 文件中没有 `field` 列；组合数据时需自行增加外部标签：

```text
COSMOS
DEEP23
```

---

## 3. 基础 catalogue 列的物理含义

COSMOS 基础 catalogue 由 HSC pipeline 测光、CLAUDS 和 VIRCAM 数据，以及后续 Phosphoros 模板拟合 photo-z 产品组成。

### 3.1 身份与天空位置

| 列 | 含义 |
| --- | --- |
| `ID` | Catalogue 对象标识符。它是身份字段，不是物理特征。 |
| `RA`, `DEC` | 源中心的 J2000 赤经、赤纬，单位为度；也是通过 WCS 定位图像切块时使用的坐标。 |
| `tract`, `patch` | HSC sky-map 的空间分区标识，描述处理几何，不是星系内禀性质。 |

天空位置、`tract` 和 `patch` 可能编码观测深度、seeing、覆盖条件及选择效应。它们不应被当作普通内禀星系量解释；若纳入任何推断，应单独记录并检验 selection-function 影响。

### 3.2 测光波段

COSMOS catalogue 包含 11 个测光波段：

| 逻辑波段 | FITS 后缀 | 设施／仪器 |
| --- | --- | --- |
| `u` | `MegaCam-u` | CFHT MegaCam `u` |
| `u_star` | `MegaCam-uS` | CFHT MegaCam `u*` |
| `g`, `r`, `i`, `z`, `y` | `HSC-G`, `HSC-R`, `HSC-I`, `HSC-Z`, `HSC-Y` | Subaru Hyper Suprime-Cam |
| `Y`, `J`, `H`, `Ks` | `VIRCAM-Y`, `VIRCAM-J`, `VIRCAM-H`, `VIRCAM-Ks` | VISTA/VIRCAM 近红外 |

DEEP23 只包含：

```text
MegaCam-u + HSC grizy
```

它不包含：

- `MegaCam-uS`
- VIRCAM `YJHKs`
- `*_NIR` photo-z 字段族
- `*_6B` photo-z 字段族

### 3.3 Flux、误差与尺寸列

每个可用波段后缀均对应以下测量：

| 列模式 | 单位与含义 |
| --- | --- |
| `FLUX_APER_2_<band>` | 2 角秒孔径中的 flux density，单位为微央斯基（µJy）。 |
| `FLUXERR_APER_2_<band>` | 2 角秒孔径 flux 的不确定度，单位为 µJy。 |
| `FLUX_APER_3_<band>` | 3 角秒孔径中的 flux density，单位为 µJy。 |
| `FLUXERR_APER_3_<band>` | 3 角秒孔径 flux 的不确定度，单位为 µJy。 |
| `FLUX_PSF_<band>` | PSF 拟合得到的 flux；对未分辨源最自然。 |
| `FLUXERR_PSF_<band>` | PSF flux 的不确定度。 |
| `FLUX_KRON_<band>` | 自适应 Kron 孔径中的 flux。 |
| `FLUXERR_KRON_<band>` | Kron flux 的不确定度。 |
| `RADIUS_KRON_<band>` | Kron 半径，单位为角秒；可反映表观尺寸和测量行为。 |
| `FLUX_CMODEL_<band>` | 复合星系模型 cmodel 拟合得到的 flux。 |
| `FLUXERR_CMODEL_<band>` | cmodel flux 的不确定度。 |

同一波段内的五种 flux 估计高度相关，但物理上并不相同。它们之间的差异可能包含以下信息：

- 紧致度或延展性；
- blending；
- 孔径损失；
- PSF 与扩展源模型的差异；
- 测量失败或不稳定性。

Catalogue 文献将 flux 定义为微央斯基。对应的物理 AB 星等转换为：

```text
m_AB = 23.9 - 2.5 log10(flux_microJy)
```

对于非正 flux，不能直接取对数；应结合误差、缺失状态及所采用的表示方式显式处理。

### 3.4 分波段状态与质量标记

每个可用波段重复以下 7 个 Boolean pipeline 字段：

| 列模式 | 含义 |
| --- | --- |
| `hasBadPhotometry_<band>` | pipeline 判断该波段测光存在问题。 |
| `isDuplicated_<band>` | 该源被标为重复对象，而非 primary source。 |
| `isNoData_<band>` | 该源没有从有效观测图像中提取。 |
| `isSky_<band>` | 该条目被识别为天空／背景对象。 |
| `isParent_<band>` | 该行代表 blended parent source。 |
| `notObserved_<band>` | 该源在该波段没有可用观测。 |
| `isClean_<band>` | 与相关测光问题有关的标记均为 false。 |

这些字段描述覆盖、缺失、deblending 和 pipeline 状态，不是星系物理量。它们可用于质量筛选、缺失机制分析或选择效应诊断，但其含义不能与直接观测量混淆。

### 3.5 跨波段分类与 mask 字段

| 列 | 含义 |
| --- | --- |
| `isCompact_HSC-G` 至 `isCompact_HSC-Y` | 各 HSC 波段中该源是否被判定为紧致。 |
| `isCompact` | 跨 HSC 波段合并后的紧致分类。 |
| `FLAG_FIELD_BINARY` | 七分量的巡天／覆盖标记向量；来源文献将分量标为 `HSC`、`u`、`u*`、`J`、`VIRCAM`、`u* deep` 和 `HSC deep`。 |
| `isOutsideMask` | 该源是否位于更新后的亮星和卫星轨迹 mask 之外。 |
| `Likelihood-Log_star` | 最优恒星模板的 log likelihood；它是分类统计量，不是星系红移。 |
| `isStarTemp` | 恒星模板是否优于星系模板。 |
| `isStar` | 结合模板拟合与紧致度得到的恒星分类。 |

### 3.6 Phosphoros photo-z 产品

| 列模式 | 含义 |
| --- | --- |
| `ZPHOT` | 主 redshift PDF 的中位数。 |
| `Z_LOW68`, `Z_HIGH68` | 包含 68% redshift PDF 的下界和上界。 |
| `Z_CHI` | 最高似然模板拟合对应的红移。 |
| `Z_PEAK` | redshift PDF 峰值位置。 |
| `Posterior-Log` | 最优 posterior 的对数值。 |
| `Likelihood-Log` | 最优星系模板 likelihood 的对数值。 |
| `*_6B` 变体 | 六波段 `u+grizy` 拟合产生的同类字段。 |
| `*_NIR` 变体 | 加入可用近红外数据后的同类字段。 |

这些字段均为 Phosphoros 从测光数据推断出的结果，不是原始观测量。

如果某个 Phosphoros 红移字段被定义为预测目标，则下列字段可能构成直接或近直接 target leakage：

- 所有 `Z*` 字段；
- `Posterior-Log*`；
- 星系模板的 `Likelihood-Log*`；
- 与目标来自同一拟合流程的置信区间或派生量。

使用前必须依据具体目标显式排除泄漏字段。

### 3.7 缺失值

数值列可能使用以下方式表示缺失：

- sentinel 值，例如 `-99`；
- `NaN`；
- `Inf` 或其他非有限值；
- 与其配套的 Boolean 状态标记。

处理时不能只检查单一缺失形式。

---

## 4. 无波段后缀的 8 个形态字段

文件：

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological.fits
```

在基础 COSMOS catalogue 后附加：

```text
p_spiral
p_bar
p_elliptical_type
axis_ellipticity
concentration_C
asymmetry_A
possible_morphological_mismatch
morphology_available
```

### 4.1 数据来源

该文件不包含彼此独立测量的 `g,r,i,z,y` 形态。

其空间信息来自一个 CLAUDS `u/uS` cutout。对于五个 HSC 通道，使用 catalogue 中 HSC `grizy` cmodel flux 比例缩放同一空间模板，形成五通道代理输入。三个像素统计量也来自该 CLAUDS cutout。

因此，这些无后缀字段描述的是一个共享空间模板下的形态表征，不应解释为六个光学波段各自独立测得的形态。

`morphology_available=True` 的行数为：

```text
1,426,259
```

### 4.2 字段含义

- `p_spiral`  
  经温度校准的 AION/Galaxy10 分类分数；将 barred spiral、紧／松旋臂 unbarred spiral 以及 edge-on spiral 类别求和。

- `p_bar`  
  barred-spiral 类概率。它是 `p_spiral` 的子集，不是与 `p_spiral` 互斥的独立类别。

- `p_elliptical_type`  
  三个 smooth／elliptical Galaxy10 类别的校准分数之和。

- `axis_ellipticity`  
  由 flux-weighted 二阶矩得到的：
  ```text
  1 - b/a
  ```

- `concentration_C`  
  ```text
  5 log10(r80/r20)
  ```

- `asymmetry_A`  
  对 180° 旋转残差计算并做噪声修正后的不对称度。

- `possible_morphological_mismatch`  
  `p_elliptical_type` 与像素圆度明显不一致时的诊断标记。它不是物理错误标签。

- `morphology_available`  
  所需代理输入和像素测量是否有效。

Galaxy10 中的 disturbed 和 merging 类别不计入 `p_spiral` 或 `p_elliptical_type`，所以这两个分数一般不会相加为 1。

---

## 5. 分波段多波段形态字段

文件：

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/catalogs/COSMOS-HSCpipe-Phosphoros_morphological_multiband.fits
```

该文件直接在基础 COSMOS catalogue 上附加形态字段，不以无后缀形态 catalogue 为父文件。

对每个后缀：

```text
u, g, r, i, z, y
```

增加 12 列，共 72 列。

### 5.1 各字段的物理／统计含义

| 列 | 含义 |
| --- | --- |
| `p_spiral_x` | 从波段 `x` 图像测得，经温度校准的 AION/Galaxy10 spiral 类概率总和，包括 barred、unbarred 和 edge-on spiral。 |
| `p_bar_x` | barred-spiral 类概率；包含在 `p_spiral_x` 内。 |
| `p_elliptical_type_x` | 三个 smooth／elliptical Galaxy10 类别的概率总和。 |
| `axis_ellipticity_x` | 从 96×96 cutout 中正值、扣背景、flux-weighted 二阶矩计算：`1 - sqrt(lambda_minor/lambda_major)`。0 表示圆，接近 1 表示拉长。 |
| `concentration_C_x` | 从正值、扣背景像素计算：`5 log10(r80/r20)`；值越大表示光分布越中心集中。 |
| `asymmetry_A_x` | cutout 与其 180° 旋转版本之间的绝对残差，做噪声修正后除以绝对源 flux；值越大表示旋转不对称性越强。 |
| `possible_morphological_mismatch_x` | 当 `abs(p_elliptical_type_x - (1 - axis_ellipticity_x)) >= 0.5` 时为 true；这是模型分数与像素圆度不一致的警告，不是物理错误标签。 |
| `surface_brightness_24_x` | 有效中央 24×24 像素内扣背景后的带符号总和。虽然列名含 `surface_brightness`，它实际是积分孔径 flux，而不是单位立体角 flux。 |
| `surface_brightness_96_x` | 全部有效 96×96 cutout 内扣背景后的带符号总和，是积分 cutout flux。 |
| `mean_per_sqarcsec_12_x` | 有效中央 12×12 像素内未经背景扣除的均值，再除以 WCS 像素面积（平方角秒）；有意保留局部天空背景。 |
| `mean_per_sqarcsec_24_x` | 有效中央 24×24 像素内未经背景扣除的均值，再除以 WCS 像素面积。 |
| `morphology_available_x` | 只有当覆盖、亮度、像素形态量和概率输出均有效时才为 true。 |

每个波段包括：

- 10 个 float 字段；
- 2 个 Boolean 字段。

不可用的 float 为 `NaN`，Boolean 为 false。

因此：

```text
possible_morphological_mismatch_x == false
```

本身存在歧义；必须同时检查：

```text
morphology_available_x
```

### 5.2 可用行数

| 波段 | `morphology_available_x=True` 的行数 |
| --- | ---: |
| `u` | 1,686,667 |
| `g` | 107,540 |
| `r` | 100,411 |
| `i` | 72,444 |
| `z` | 46,007 |
| `y` | 313,133 |
| 六个波段同时可用 | 1,658 |

要求六个波段全部可用会得到一个非常小且高度选择性的样本。

任何使用形态字段的分析都应明确说明：

- 是要求完整交集；
- 使用逐波段 availability mask；
- 还是采用其他缺失值策略。

不能在未报告的情况下默认使用 1,658 行的六波段交集。

### 5.3 形态概率与跨波段比较的限制

概率头在单波段 DES Galaxy10 的孤立样本上训练并做温度校准，随后迁移到 CLAUDS/HSC 图像。

- `u` cutout 通过 AION 的 HSC-G 通道编码，属于 codec proxy；
- HSC `g,r,i,z,y` 使用对应的 AION 通道；
- 这是跨巡天迁移结果，不是经过本 catalogue 独立校准的内禀形态真值。

不同波段图像没有进行 PSF matching。因此，波段间差异可能同时反映：

- 波长依赖的真实形态；
- seeing；
- 空间分辨率；
- 深度；
- mask 和有效像素差异。

亮度相关字段的 FITS metadata 为：

```text
MORPHUNT = scaled-native
```

在线文件中各波段 scale factor 均为 1.0。它们不保证具有统一的物理 flux 标定。

在没有额外标定和明确记录的情况下，不要直接把 CLAUDS 与 HSC 波段的以下列数值作为同一物理单位横向比较：

```text
surface_brightness_*
mean_per_sqarcsec_*
```

---

## 6. 图像位置与文件含义

### 6.1 CLAUDS `u/uS` 图像

目录：

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/clauds/images/tilesv5/
```

在线目录包含 2,496 个 FITS 文件：

- 671 个 `Mega-u_*.fits` science tile；
- 671 个与之对应的 weight map；
- 577 个 `Mega-uS_*.fits` science tile；
- 577 个与之对应的 weight map。

命名规则：

```text
Mega-u_<tract>_<patch>.fits
Mega-u_<tract>_<patch>.weight.fits
Mega-uS_<tract>_<patch>.fits
Mega-uS_<tract>_<patch>.weight.fits
```

含义：

- `Mega-u` 是 CFHT/MegaCam `u`；
- `Mega-uS` 是不同的 CFHT/MegaCam `u*`；
- science 文件的 primary HDU 是 coadded sky image；
- 对应的 `.weight.fits` primary HDU 是 weight map；
- `weight > 0` 表示该像素具有可用覆盖；
- 代表性在线 tile 大小为 4200×4200 像素；
- CLAUDS 官方图像 photometric zero point 为 30.0；
- 已检查的在线 header 没有提供 `BUNIT`。

在多波段形态 catalogue 中，`Mega-u` 与 `Mega-uS` 的覆盖均映射到单一后缀：

```text
_u
```

不存在单独的 `_u_star` 形态字段族。

### 6.2 HSC PDR3 `grizy` 图像

目录：

```text
/arc/projects/ots/pdr3_dud/
```

在线目录包含 7,514 个相关 `calexp` 文件：

| Filter | 文件数 |
| --- | ---: |
| HSC-G | 1,496 |
| HSC-R | 1,498 |
| HSC-I | 1,503 |
| HSC-Z | 1,527 |
| HSC-Y | 1,490 |

命名规则：

```text
calexp-HSC-<FILTER>-<tract>-<patch>.fits
```

示例：

```text
calexp-HSC-G-8767-7%2C5.fits
```

其中 `%2C` 是 URL 编码的逗号，所以 patch 为：

```text
7,5
```

这些文件是 HSC PDR3 Deep/UltraDeep coadd `calexp` 产品，采用 local sky subtraction，用于 HSC 源检测和测量。

官方 coadd zero point 为：

```text
27.0 mag/DN
```

文档同时说明图像层面的 aperture correction 存在百分之几量级的限制。

多扩展 FITS 结构：

| HDU | 含义 |
| ---: | --- |
| 0 | Primary metadata，包括 filter。 |
| 1 | Science image。 |
| 2 | 整数 bit-mask plane。 |
| 3 | Variance image。 |
| 4 及以后 | PSF、标定、aperture correction 和其他 HSC metadata 表。 |

一个像素只有在以下条件全部成立时才可视为有效：

- science 值有限；
- variance 值有限；
- variance 为正；
- 未触发所配置的 invalid mask bits。

被排除的 mask 含义包括：

- bad pixel；
- saturation；
- interpolation；
- cosmic ray；
- edge；
- suspect／no-data；
- bright object；
- crosstalk；
- failed deblending；
- 未 mask 的 NaN；
- rejected／clipped pixel；
- sensor edge。

同一目录还包含：

```text
pdr3_catalog.fits
pdr3_dud_catalog.parquet
checksum sidecar
notebook
```

这些不是图像 patch，也不应与 `calexp` science image 混淆。

---

## 7. Catalogue 对象与图像的连接方式

不是每个 catalogue 对象对应一个独立图像文件。

每个巡天 FITS 文件是一块包含许多对象的大型天空 tile；不同 tile 还可能重叠。

对象与图像之间的连接原则为：

1. 从 COSMOS 基础 catalogue 读取 `ID`、`RA`、`DEC` 和必要的对象状态字段。
2. 对图像文件建立包含以下信息的 manifest：
   - 文件路径；
   - filter；
   - WCS footprint；
   - 图像尺寸；
   - WCS 像素面积。
3. 对每个对象和每个波段，使用 `RA/DEC` 与 tile WCS 找到包含该位置的图像及像素坐标。
4. 若某项分析需要排除恒星，应依据明确的分类字段执行，并记录筛选标准。
5. 从对应 science image 中提取以对象为中心的 cutout。
6. 使用 CLAUDS weight map，或 HSC mask 与 variance plane，定义有效像素。
7. 如计算扣背景统计量，应显式说明背景估计区域和方法。
8. 生成派生表时，应保留 `ID`，并记录 catalogue 来源及波段。

基础 catalogue 测光与图像像素来自相关巡天产品，但二者不可互换：

- catalogue `FLUX_*` 是 hscPipe 输出的源级微央斯基测光量；
- 图像像素是具有空间结构的巡天图像单位；
- 形态概率和像素统计来自图像 cutout；
- `tract` 和 `patch` 描述 HSC 处理几何；
- 跨图像定位的权威连接方式是 `RA/DEC + WCS`。

---

## 8. 数据解释与防泄漏检查清单

在使用数据之前，至少确认：

- 使用的是哪个 catalogue；
- 是否混合 COSMOS 与 DEEP23；
- 身份键是否为 `(field, ID)`；
- 目标字段是什么；
- 是否排除了由同一推断流程产生的目标派生字段；
- 是否误把 `ID`、`RA/DEC`、`tract/patch` 当作内禀物理量；
- `-99`、`NaN` 和非有限值是否都被识别；
- Boolean availability 字段是否与 float 形态字段一起解释；
- 是否因要求完整多波段形态而形成强选择样本；
- CLAUDS 与 HSC 图像量是否经过可比的物理标定；
- 是否记录了图像 mask、有效像素和背景处理方式；
- 是否在跨天区或跨波段比较时保留了来源标签。

---

## 9. 数据来源参考

- CLAUDS available data and image access  
  https://www.clauds.net/available-data

- Desprez et al. 2023：hscPipe／Phosphoros catalogue 及 Table E.2  
  https://www.aanda.org/articles/aa/pdf/2023/02/aa43363-22.pdf

- HSC PDR3 available data and `calexp` definition  
  https://hsc-release.mtk.nao.ac.jp/doc/index.php/available-data__pdr3/

- HSC PDR3 FAQ：local-sky coadds 与 zero point  
  https://hsc-release.mtk.nao.ac.jp/doc/index.php/faq__pdr3/
