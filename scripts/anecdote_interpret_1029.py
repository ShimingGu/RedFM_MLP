"""Concise, outcome-based interpretation; no model or score is changed."""
import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/stage4/1029'
def run():
    report=OUT/'report.md';s=report.read_text();lead='''The answer is conditional: **CLAUDS-posttrained BC supplies transferable GMS representation information under the tested readout, but the study does not establish a practical advantage over inexpensive numerical photo-z controls or a general astrophysical-knowledge benefit.**

At the registered K256 budget, canonical hard-CE GMS outliers are28.68% forB0 and22.67% forBC:−6.01 percentage points, with both outlier and CE improvements in all6 support/head fits (5/6 conditional paired outlier intervals below zero). This is the genuine off-diagonal direction BC→GMS on historically untrained objects. Mean pooling and shared alternate formatting retain a CE advantage in all6 fits, while their mean outlier gains shrink to−1.12pp and−2.60pp. Thus some transfer is visible, with readout/format dependence. The same-budget numerical MLP reaches17.06% outliers and photometric kNN2.54%.

Original hard-label heads at K64 do not learn, and CLAUDS hard-label readouts at K256 also largely select initialization. They do not support representation rankings. In the preregistered sigma3 control, BS→C_SPEC at K256 lowers outliers46.57%→44.51% and CE4.917→4.867;5/6 fits improve outliers, but only2/6 conditional outlier intervals exclude zero. The numerical MLP reaches19.68% outliers andCE4.168 under the same sigma3/K/development-label setting. This is a modest exploratory representation gain rather than competitive true-redshift inference. Source-only sigma10 C_ZPHOT results are likewise weak/mixed for BS:56.64%→55.68% outliers, with slightly worse CE; they do not establish stable cross-survey ZPHOT sample-efficiency gains.

K denotes fitted training labels. Every learned readout also uses256 fixed labelled development objects, so K256 is not a256-total-label experiment. All intervals are conditional on a fixed query cohort and a particular support/head fit; different heads on the same galaxy are not independent confirmation.

![Compact K256 comparison](representation_transfer_summary.png)

'''
    s=s.replace('Only fresh heads were optimized.',lead+'Only fresh heads were optimized.',1)
    s=s.replace('7. **Numerical controls:** compare all registered MLP/kNN/global corrections at matched support/query identities and disclosed supervision costs. paired_numerical_control_bootstrap.json prevents a favorable backbone-only comparison from substituting for these controls.','7. **Numerical controls:** no practical numerical-control advantage is established. At GMS K256 hard CE, BC outliers22.67% exceed MLP17.06% andkNN2.54%. With sigma3, BC4.48% exceeds MLP2.29% andkNN2.54%. C_SPEC sigma3 BS44.51% exceeds MLP19.68%; C_ZPHOT sigma10 BS55.68% exceeds MLP34.28%. Point and PDF metrics are both retained, and global-affine controls are separately tabulated. The primary representation improvement overB0 remains a narrower finding.')
    s=s.replace('10. **Future few-shot prompting/calibration:** successful frozen-feature readouts can motivate a limited no-backbone-update calibration experiment; they do not prove in-context prompting ability. If gains are unstable, prioritize readout/format/coverage diagnosis before scaling.','10. **Future few-shot prompting/calibration:** BC→GMS provides a reason to retain a small frozen-feature calibration branch, but no result proves in-context prompting ability. The dominant low-K300-class readout failure and stronger raw-photometry controls favor testing a simpler continuous/density readout before expensive backbone posttraining.')
    s=s.replace('11. **Next Paper II models/cohorts:** the verified common4B B0/BS/BC checkpoints and saved clean IDs are reusable. Preserve diagonal/in-domain labels and use a fresh independently held spectroscopic cohort for confirmation. No single winner is selected from query performance for deployment.','11. **Next Paper II models/cohorts:** retain B0 as the essential control and BC for the observed BC→GMS representation effect. BS→C_SPEC is a smaller exploratory candidate; keep numerical MLP/kNN as mandatory practical baselines. Reuse the saved clean IDs only with their inspected/exploratory status preserved, and reserve new independent spectroscopy for confirmation. No deployment winner or additional backbone-training run is selected here.')
    report.write_text(s)
    (OUT/'summary_zh.md').write_text('''# 冻结骨干小头迁移：结果摘要

完成 B0 原始模型、BS 光谱后训练模型、BC CLAUDS ZPHOT 后训练模型的共同输入比较。Qwen 和适配器始终冻结，共完成900个小头拟合；各条件使用3个支持集种子、2个初始化种子。K以外均有固定256个开发标签用于选择检查点。

- **存在局部表示迁移：** K256的 BC→GMS 原始硬标签主比较，离群率28.68%→22.67%，平均改善6.01个百分点；6/6组合改善离群率和CE，5/6逐星系配对区间不含零。均值池化与共同替代格式下CE改善仍存在，点估计收益变小。
- **没有超过简单数值控制：** 同一GMS预算，MLP离群率17.06%，kNN2.54%。因此不能将局部BC−B0收益写成基础模型的实用优势。
- **CLAUDS光谱方向收益较弱：** 原始硬分类头未学会；预先设定的sigma3控制下，BS相对B0为46.57%→44.51%，但MLP为19.68%。这只是探索性的条件收益。
- **低K读出瓶颈必须单独说明：** 300个硬分类bin在少标签下大量选择初始化。C_ZPHOT在查询前另行登记固定sigma10诊断；原始失败条件完整保留，没有用查询结果挑平滑强度。
- **曝光已排除：** 原1976个光谱对象中17个BC训练曝光、5个历史验证、3个不确定匹配被排除，干净对象1951个。该光谱集此前已被Stage3检查，本研究不是新的独立确认；也不代表全部暗弱CLAUDS总体。

[完整报告](report.md) · [K256简报图](representation_transfer_summary.png) · [完整学习曲线](learning_curves.png) · [验证记录](verification.json)
''')
    p=OUT/'scientific_summary.json';r=json.loads(p.read_text());r.update(conclusion='Conditional BC-to-GMS representation transfer observed; no practical advantage over simple numerical controls established.',GMS_K256_BC_minus_B0_outlier_pp=-6.005859375,primary_GMS_six_fit_outlier_and_CE_direction_consistent=True,C_SPEC_smooth3_gain_exploratory=True,hard_low_K_nonlearning_not_a_backbone_ranking=True,fixed_development_labels=256);p.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':run()
