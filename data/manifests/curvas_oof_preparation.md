# CURVAS：OOF 分割监督数据准备记录

本记录描述 CURVAS 的原始数据解压、自动校验和胰腺多数票标签生成阶段。该阶段**没有执行患者级划分，也没有创建 nnU-Net `imagesTr`/`labelsTr`**；后续已完成的名单冻结和仅训练患者导出见[CURVAS OOF 名单及导出记录](./curvas_oof_split_and_export.md)。QUBIQ 未参与处理。

## 目录与规则

| 路径 | 内容 |
|---|---|
| `data/raw/curvas/` | 三个原始 ZIP，保持不变 |
| `data/extracted/curvas/{training_set,validation_set,testing_set}/` | 按官方原目录解压的 CT 与三份专家标注 |
| `data/derived/curvas/pancreas_majority_vote/` | 每例一份胰腺二值多数票掩膜，文件名为患者 ID |
| `data/manifests/curvas_oof_label_manifest.csv` | 病例来源、尺寸、spacing、专家与多数票体素数、自动质检标记 |
| `data/manifests/curvas_case_dispositions.csv` | 用户明确作出的病例保留决定；与自动质检标记分开记录 |

CURVAS 原标注中 `0=背景、1=胰腺、2=肾脏、3=肝脏`。对同一 CT 网格上的三份专家标签，逐体素统计值为 `1` 的票数，票数 ≥2 记为胰腺，否则记为背景。原始多器官标注不修改。生成的掩膜保持原 CT 的尺寸和物理网格；**尚未做重采样、强度归一化、裁剪或手工标注审核**。

部分源标注的 NIfTI 头带有缩放参数，读取后的标签值可能是非常接近整数的浮点数。处理脚本根据存储编码及缩放参数还原语义标签，在 `1e-3` 容差内确认其属于 `{0,1,2,3}`，避免直接用浮点 `== 1` 漏选胰腺。

## 自动校验

- 官方三个包应分别包含 20、5、65 例；每例必须恰有一份 `image.nii.gz` 和三份 `annotation_*.nii.gz`，患者 ID 不重复。
- 解压后的所有 360 个 `.nii.gz` 均已通过 `gzip -t`；验证包和测试包由 `unzip` 正常解压。训练包的 ZIP 包头被当前 `7z` 报错，但其 MD5 与官方一致，`7z` 实际解出了全部 80 个 NIfTI 文件，且这些内部 gzip 文件通过完整性测试。
- 处理脚本逐例检查 CT 为 3D、三份标注与 CT 的尺寸及有效仿射矩阵一致、标注语义值合法；生成后重新读取掩膜头，检查 dtype 为 `uint8` 且几何不变。
- `PASS_AUTOMATED` 只表示上述机器检查通过，不等于三位专家标注已通过人工质量审核。若出现空专家掩膜则标记 `REVIEW_EMPTY_EXPERT`。

最终产物为 90 份多数票掩膜，均通过 `gzip -t`；用 SimpleITK 再次读取头信息，90 份掩膜的尺寸、spacing、原点与方向均与对应 CT 一致。逐例清单有 90 行：88 例 `PASS_AUTOMATED`，另有以下两例待复核。

另用独立的“读取缩放后标签、按整数容差直接投票”方式复算了 `UKCHLL009`、`UKCHLL007`、`UKCHLL082` 三例，与写出的掩膜逐体素比较，差异均为 0。

`testing_set/UKCHLL007/annotation_2.nii.gz` 是唯一发现的空间头信息异常：与 CT 数组尺寸相同，但记录的层厚为 0.7 mm，CT 及另外两份标注为 1.0 mm，仿射矩阵最大差约 1028.84 mm。按数组索引比较，三位专家胰腺掩膜两两 Dice 为 0.8831、0.8952、0.9005，支持“标注内容按索引已对齐、头信息错误”的判断。脚本对此**唯一显式例外**按索引投票，写出的多数票掩膜使用 CT 空间，并要求与第 1 位专家的索引 Dice ≥0.8；自动清单仍标记 `REVIEW_HEADER_MISMATCH`。原标注未改动。用户已决定保留该病例及现有 CT 空间多数票掩膜；这项保留决定不等同于人工影像叠加复核通过。

`testing_set/UKCHLL082/annotation_3.nii.gz` 的胰腺体素数为 0，另外两份分别为 88706、91625 个体素；多数票掩膜有 85870 个体素，实际是两份非空标注的交集。自动清单仍标记 `REVIEW_EMPTY_EXPERT`。用户已决定保留该病例及现有掩膜；这项决定不等同于第三份专家标注已被修复，后续合理差异监督仍需单独处理。

## 病例保留决定（2026-09-25）

用户明确要求保留 `UKCHLL007`、`UKCHLL008` 和 `UKCHLL082` 及其已生成的 CT 空间多数票掩膜。三例都保留在后续患者划分的候选池中，不重写原始 CT、专家标注或派生掩膜；具体记录见 `curvas_case_dispositions.csv`。`UKCHLL007` 与 `UKCHLL082` 的自动质检警告继续保留，不能将保留决定解读为人工标注质量审核通过。

复现命令（在项目根目录、`pancreas_qc` 环境中）：

```bash
python scripts/prepare_curvas_oof_labels.py --workers 8
```

患者名单与 nnU-Net 原始输入的后续处理见[CURVAS OOF 名单及导出记录](./curvas_oof_split_and_export.md)。自动质检警告不因名单冻结而消失，`UKCHLL007` 与 `UKCHLL082` 的叠加/标注复核仍建议单独完成。
