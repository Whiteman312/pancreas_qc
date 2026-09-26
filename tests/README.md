# 关键检查

已有 `test_curvas_oof_tasks.py`：17 项小型合成数据测试，覆盖五折 48/12 协议、留出/校准/测试患者排除、符号链接目标、配置、重复运行和既有文件保护。不读取真实医学数据，不启动训练。

```bash
python -m unittest discover -s tests -p 'test_curvas_oof_tasks.py' -v
```

真实数据的哈希与空间检查、五个任务的原生 nnU-Net 完整性检查由准备脚本完成：

```bash
python scripts/prepare_curvas_oof_tasks.py --verify-only --verify-nnunet --num-processes 4
```

训练后仍需检查 OOF 模型未见预测患者、预测与原 CT 空间一致；检测器实现后还需测试三值监督中的 `-1` 区域被损失函数忽略。
