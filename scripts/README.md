# 阶段命令入口

脚本与实验阶段一一对应：

1. `00_prepare_data.py`：数据核验、清单和患者划分；
2. `01_train_segmentation.py`：nnU-Net 五折训练与 OOF 概率；
3. `02_build_candidates.py`：候选标签与三值监督；
4. `03_train_detector.py`：M2/M3 错误检测器；
5. `04_calibrate.py`：阈值与病例级校准；
6. `05_evaluate_curvas.py`：CURVAS 同域测试；
7. `06_infer_qubiq.py`：QUBIQ 外部推理。

当前均为占位入口，运行时会明确退出，不会启动训练。
