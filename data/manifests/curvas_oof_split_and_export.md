# CURVAS OOF 患者名单冻结与 nnU-Net 原始输入导出

日期：2026-09-25。只处理 CURVAS；不包含 QUBIQ，也未启动 nnU-Net 规划、预处理或训练。

## 1. 冻结患者名单

配置在 [`config/data.yaml`](../../config/data.yaml)，固定随机种子 `20260925`。按患者 ID 将 90 例分为训练 60、校准 10、测试 20；原官方压缩包仅作为来源分层，**不是本实验的训练/测试定义**。来源配额如下：

| 原包 | 训练 | 校准 | 测试 |
|---|---:|---:|---:|
| `training_set` | 13 | 2 | 5 |
| `validation_set` | 3 | 1 | 1 |
| `testing_set` | 44 | 7 | 14 |

训练 60 例再固定成五个互斥留出折，每折 12 例；每折训练名单为其余 48 例。完整名单、每折训练/留出患者 ID、源清单哈希在 [`curvas_patient_split.json`](./curvas_patient_split.json)。该文件 SHA-256 为 `3660d71c9db5d7f407f3649d49fb3a2c1be87feb28588e160a70aa9c74a5a316`。名单生成脚本拒绝覆盖与现有名单不同的结果。

依用户决定，`UKCHLL007`、`UKCHLL008` 与 `UKCHLL082` 均保留。前两例落在训练集，`UKCHLL082` 落在校准集。`UKCHLL007` 的头信息异常和 `UKCHLL082` 的空专家胰腺标注仍按原自动质检标记保留；后者不进入本轮 OOF 分割网络训练，但以后使用校准集专家标签时必须单独处理其可靠性。

名单环节自校验：重读冻结文件并重新按相同配置/源清单计算，结果完全一致；另独立核对 90 位患者无重复或遗漏，训练/校准/测试互斥，五折各为 48 训练/12 留出，训练 60 人恰好各留出一次。

## 2. 导出 nnU-Net v2 原始输入

目录：[`data/nnunet_raw/Dataset501_CURVASPancreasOOF/`](../nnunet_raw/Dataset501_CURVASPancreasOOF/)。仅复制训练 60 例的 CT 至 `imagesTr/UKCHLLxxx_0000.nii.gz`，对应多数票胰腺掩膜至 `labelsTr/UKCHLLxxx.nii.gz`；**没有导出校准或测试患者，也没有 `imagesTs`**。`dataset.json` 定义单通道 `CT`、`background=0`、`pancreas=1`、`numTraining=60`、`.nii.gz`。源与目标路径、每例 OOF 留出折、SHA-256 见 [`curvas_nnunet_export.csv`](./curvas_nnunet_export.csv)。原 CT、专家标注及多数票掩膜未被改写。

导出环节自校验：60 份 CT 与 60 份掩膜的目标 SHA-256 均等于源文件；文件名与冻结训练名单完全一致；全部 CT/掩膜对的 SimpleITK 尺寸、spacing、原点、方向一致；`dataset.json` 与单通道二值任务设置一致。导出大小约 19 GiB。

另外在 `pancreas_qc` 环境中调用 nnU-Net v2 自带的 `verify_dataset_integrity`，使用 `SimpleITKIO` 读取全部 60 例，完整性检查正常结束且没有错误报告。这里只调用输入验证，**未运行数据指纹提取或预处理规划**。

复核命令（项目根目录、`pancreas_qc` 环境）：

```bash
python scripts/freeze_curvas_split.py --verify-only
python scripts/export_curvas_nnunet.py --verify-only
python -c 'from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity; verify_dataset_integrity("data/nnunet_raw/Dataset501_CURVASPancreasOOF", num_processes=4)'
```

## 下一环节边界

这个 60 例目录是**仅训练患者的原始输入汇集目录**。若按当前方案要求每折的强度统计和空间规划只从该折 48 例拟合，不要直接在这个 60 例目录运行一次全局 `nnUNetv2_plan_and_preprocess`。

2026-09-26 已继续按冻结名单派生五个 48 例任务，编号 511–515；任务设置、名单清单与自校验记录见 [`curvas_oof_fold_tasks.md`](./curvas_oof_fold_tasks.md)。下一步才是各任务独立规划、以 `fold all` 训练，并在对应 12 例上生成 OOF 浮点概率图。
