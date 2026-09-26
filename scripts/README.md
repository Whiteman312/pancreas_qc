# 阶段命令入口

脚本与实验阶段一一对应：

1. `00_prepare_data.py`：数据核验、清单和患者划分；
2. `01_train_segmentation.py`：nnU-Net 五折训练与 OOF 概率；
3. `02_build_candidates.py`：候选标签与三值监督；
4. `03_train_detector.py`：M2/M3 错误检测器；
5. `04_calibrate.py`：阈值与病例级校准；
6. `05_evaluate_curvas.py`：CURVAS 同域测试；
7. `06_infer_qubiq.py`：QUBIQ 外部推理。

以上编号入口目前均为占位，运行时会明确退出，不会启动训练。

CURVAS 数据准备已有可运行脚本（项目根目录、`pancreas_qc` 环境）：

- `prepare_curvas_oof_labels.py`：CT 空间多数票胰腺监督标签。
- `freeze_curvas_split.py`：冻结 60/10/20 患者划分及五个 48/12 折，支持 `--verify-only`。
- `export_curvas_nnunet.py`：仅导出训练 60 例至 Dataset501，支持 `--verify-only`。
- `prepare_curvas_oof_tasks.py`：根据冻结名单制作 Dataset511–515，每个任务 48 例；默认逐环节自校验，支持 `--verify-only` 和 `--verify-nnunet`（对五个任务运行原生完整性检查）。

上述准备脚本不启动 nnU-Net 指纹提取、规划、预处理或训练。折专属任务记录见 [`curvas_oof_fold_tasks.md`](../data/manifests/curvas_oof_fold_tasks.md)。
