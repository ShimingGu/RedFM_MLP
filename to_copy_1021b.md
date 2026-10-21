**当前 GalaxiesML-Spec loader 输入的是 HSC `grizy` 测光，没有使用它自带的 Legacy `grz/W1/W2`。**

具体从 HDF5 读取：

- `HSC_metadata/{g,r,i,z,y}_cmodel_mag`
- `HSC_metadata/{g,r,i,z,y}_cmodel_magsigma`
- `DESI_fibermap/EBV`

将 magnitude 转成 AB nanomaggy flux，并传播误差，最终得到 **5 个 flux + 5 个一倍标准差误差 + EBV，共 11 个特征**。监督标签是 `DESI_fibermap/DESI_redshift`，没有输入图像或光谱像素。

因此，当前实验的测光配置是：

| 训练样本 | 输入波段 |
|---|---|
| DESI-LS | Legacy `grz/W1/W2` |
| GalaxiesML-Spec | HSC `grizy` |

已核对当前运行保存的配置记录：

```text
/arc/home/gsm/aion_output/figures/qwen-galaxiesml-spec-qlora-specz-e10/prepared.json
```
