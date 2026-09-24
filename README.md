# Pancreas QC

增强腹部 CT 胰腺分割标签质控实验。项目骨架已建立，训练与推理代码尚待实现。

实验设计见 [第一版训练与外部推理实施方案](docs/第一版训练与外部推理实施方案.md) 和 [第一版分阶段实验配置说明](docs/第一版分阶段实验配置说明.md)。环境依赖见 `config/experiment_environment.yml`。

原始数据与生成的大文件放在 `data/`、`artifacts/` 和 `runs/` 对应目录，不提交到版本控制。患者划分清单与数据清单分别保存在 `data/splits/` 和 `data/manifests/`。
