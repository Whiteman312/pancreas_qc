# CURVAS 五个折专属 nnU-Net 原始任务

日期：2026-09-26。仅准备 CURVAS OOF 分割网络的输入；不包含 QUBIQ，不启动指纹提取、规划、预处理或训练。

## 1. 不重新划分患者

使用已经冻结的 [`curvas_patient_split.json`](./curvas_patient_split.json)，SHA-256 仍为 `3660d71c9db5d7f407f3649d49fb3a2c1be87feb28588e160a70aa9c74a5a316`。原有训练 60、校准 10、测试 20 及五折名单不变；没有重新抽样、重新分折或改变多数票监督。

配置入口为 [`config/data.yaml`](../../config/data.yaml) 的 `curvas.nnunet.fold_tasks`。训练与留出名单、留出 CT 路径、源清单哈希见 [`curvas_oof_tasks.json`](./curvas_oof_tasks.json)；240 条训练 CT/掩膜引用及其哈希见 [`curvas_oof_task_export.csv`](./curvas_oof_task_export.csv)。

| 冻结折 | nnU-Net 任务 | 训练 | 留出 | 训练病例原包来源：training / validation / testing |
|---|---|---:|---:|---|
| 0 | `Dataset511_CURVASPancreasOOF_Fold0` | 48 | 12 | 10 / 3 / 35 |
| 1 | `Dataset512_CURVASPancreasOOF_Fold1` | 48 | 12 | 10 / 3 / 35 |
| 2 | `Dataset513_CURVASPancreasOOF_Fold2` | 48 | 12 | 11 / 2 / 35 |
| 3 | `Dataset514_CURVASPancreasOOF_Fold3` | 48 | 12 | 11 / 2 / 35 |
| 4 | `Dataset515_CURVASPancreasOOF_Fold4` | 48 | 12 | 10 / 2 / 36 |

原包名称只表示来源，不表示本实验的训练/测试用途。每个训练患者进入四个任务，并恰好被一个任务留出。校准 10 例和测试 20 例不进入任何任务。

## 2. 文件结构与链接方式

五个任务均位于 `data/nnunet_raw/`。每个任务仅有以下内容：

```text
Dataset511_CURVASPancreasOOF_Fold0/  # 其余四个任务结构相同
├── dataset.json                   # numTraining=48
├── imagesTr/                      # 48 个 UKCHLLxxx_0000.nii.gz
└── labelsTr/                      # 48 个 UKCHLLxxx.nii.gz
```

`imagesTr` 与 `labelsTr` 中使用相对符号链接，精确指向已有 `Dataset501_CURVASPancreasOOF` 中的相应文件，不重复复制约四份训练数据。链接例如 `../../Dataset501_CURVASPancreasOOF/imagesTr/UKCHLL001_0000.nii.gz`。nnU-Net 完整性检查将实际读取链接目标。

五个任务均定义单通道 `CT`、`background=0`、`pancreas=1`、`numTraining=48`、`.nii.gz`。没有 `imagesTs`，也没有将该折 12 例留出患者或其标签放入该任务；留出病例仅在任务清单中记录 CT 推理输入路径。

注意：符号链接是节省空间的文件引用，不是隔离权限或只读保护。不要从链接路径改写 CT/掩膜；移动数据时，应保留五个任务与 `Dataset501` 的相对位置并一起迁移，或者显式物化链接后重新核对哈希。不要删除 `Dataset501`。

`UKCHLL007`、`UKCHLL008` 的保留决定继续生效；007 在折 0 留出，在折 1–4 训练，008 按冻结名单安排。`UKCHLL082` 仍在校准集，不进入五个任务。007 的原始专家头信息异常、082 的空专家胰腺标注和未完成的人工叠加复核记录均未被清除；自动检查通过不等于已做人工复核。

## 3. 分环节自校验

构建脚本 [`scripts/prepare_curvas_oof_tasks.py`](../../scripts/prepare_curvas_oof_tasks.py) 在各环节后运行检查，拒绝覆盖已有的不匹配数据或元信息。独立 `--verify-only` 模式不会创建或修改任务。

1. 输入环节：核对冻结名单及输入清单哈希，重新按冻结配置计算名单以确认没有漂移；重算 60 对唯一源文件和 `Dataset501` 文件哈希，检查 CT/掩膜的尺寸、spacing、原点、方向。
2. 逐折构建：每完成一个任务就核对恰好 48 对文件、单通道配置、每个相对链接的准确目标，确认 12 例留出患者不在该任务；同时排除校准/测试病例。
3. 全局核对：五个任务合计 240 对引用；每个训练患者恰好训练四次、留出一次；任务 JSON 和 CSV 必须与冻结输入可重复生成且完全一致。
4. 回归测试：验证正常构建与重复运行，以及重复患者、重复折、留出/校准/测试病例混入、错误或断开的链接、错误配置和既有文件保护。
5. 原生检查：使用 nnU-Net `verify_dataset_integrity` 分别读取五个任务，检查标签值、通道数、图像/标签尺寸与空间信息等；该调用不提取指纹，也不执行规划或训练。

本次执行结果：上述检查全部通过。60 对源文件与汇集文件的哈希、空间信息一致；五个任务各为 48 对文件，留出/校准/测试患者排除正确；独立 `--verify-only` 复核成功；17 项回归测试通过。原生完整性检查依次完成 511、512、513、514、515，均使用 `SimpleITKIO`，进程正常结束且无错误报告。

另外用实际五个任务重复执行链接/元信息构建逻辑，480 个链接与五份 `dataset.json` 的 inode 和修改时间均未变化。五个目录的文件引用避免了约 74 GiB 的重复复制；原始 CT、专家标注、多数票掩膜、60/10/20 名单及已有 Dataset501 均未改写。

## 4. 复现与独立复核

在项目根目录、`pancreas_qc` 环境执行：

```bash
python scripts/prepare_curvas_oof_tasks.py
python scripts/prepare_curvas_oof_tasks.py --verify-only --verify-nnunet --num-processes 4
python -m unittest discover -s tests -p 'test_curvas_oof_tasks.py' -v
```

构建可重复执行，正确的既有链接和元信息保持原样；发现不同结果时直接报错，不重新分配患者或静默覆盖。

## 5. 后续训练边界

`Dataset501` 只作为 60 例训练患者的汇集输入，不能用它统一拟合严格 OOF 的强度统计或规划。下一步对 511–515 **分别**提取该折 48 例的指纹并规划；各任务以 `fold all` 训练全部 48 例，再仅对任务清单中的对应 12 例生成 OOF 浮点概率图。每位患者使用的 OOF 模型必须是未见过该患者的那一个。

例如完成任务 511 的独立规划之后，训练入口形式为 `nnUNetv2_train 511 3d_fullres all`，不是 `nnUNetv2_train 511 3d_fullres 0`，也不是在每个 48 例任务内部再做五折。这里的外层折 0 对应数据集 511，nnU-Net 的内部训练折参数为 `all`，两者不要混淆。

当前安装的 nnU-Net 默认 trainer 在 `fold all` 下会将这 48 例同时用于训练和内部验证。因此内部验证指标**不是独立 OOF 指标**；12 例留出数据不参与该折训练、早停或模型选择，仅用于固定训练方案后的 OOF 推理/评估。五个模型的输出应按各自规划逆变换回原 CT 空间，不能直接平均不同任务的预处理网格。
