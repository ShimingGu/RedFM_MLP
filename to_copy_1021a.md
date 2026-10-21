已经确定了：**当前 DESI–Legacy 训练集采用 DESI DR1 的 Iron 光谱红移目录，配合 Legacy Surveys DR9 的 Tractor 测光。**

具体的 spec-\(z\) 来源是：

```text
/arc/projects/ots/Cosmic_Imprint_of_Time/DESI-LS/catalogues/desi_dr1/iron/zcatalog/v1/zall-pix-iron.fits
```

以其中的 **`Z` 列作为监督标签**，通过 `TARGETID` 与配套的 `catalogues/ls_dr9/iron/v1.1/tractorphot` 目录精确匹配。

实际送入 Qwen 的是 **11 个数值特征**：

| 原始列 | 模型输入 |
|---|---|
| `FLUX_G`, `FLUX_R`, `FLUX_Z`, `FLUX_W1`, `FLUX_W2` | 五个波段的 flux |
| 对应的五列 `FLUX_IVAR_*` | 转换成一倍标准差误差：\(\sigma=1/\sqrt{\mathrm{IVAR}}\) |
| `EBV` | 银河系消光信息 |

Flux 单位为 **AB nanomaggy**，保留有效的负 flux；没有额外做消光修正，而是将 `EBV` 一起提供。每个波段要求有限 flux、`IVAR > 0`、`NOBS > 0`，每个对象至少三个有效波段；缺失值显式标记为 `NA`。

光谱样本筛选采用：

- `SURVEY=main`，`PROGRAM=bright/dark`；
- `SPECTYPE=GALAXY`、`OBJTYPE=TGT`、`ZCAT_PRIMARY=True`；
- `ZWARN=0`、`COADD_FIBERSTATUS=0`、`DELTACHI2>40`；
- 有限的 \(0\le Z\le4.5\)，且 `ZERR` 有限、非负。

这是当前实验统一的质量筛选；**没有采用 DESI 按 tracer 分别定义的 LSS／[O II] 成功判据**。

`TARGETID`、成像对象标识、`RA/DEC` 和质量列仅用于匹配、去重或筛选，不进入模型；也没有输入图像或光谱像素。当前抽样去重后共 **299,988** 个对象，其中训练 **59,855**、验证 **15,053**、测试 **225,080**，QLoRA 与 DoRA 使用相同样本及划分。

以上已核对当前运行保存的配置记录：

```text
/arc/home/gsm/aion_output/figures/qwen-desi-ls-qlora-specz-e10/prepared.json
```
