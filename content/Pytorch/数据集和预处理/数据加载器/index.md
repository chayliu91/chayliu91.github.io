---
title: "数据加载器"
date: 2026-03-24
draft: false
categories: ["Pytorch", "数据集和预处理"]
tags: ["Pytorch"]
weight: 2
---

# DataLoader 类
`torch.utils.data.DataLoader` 是一个迭代器，包装了 `Dataset`。负责批量加载、打乱数据、多进程并行读取，将单个样本组装成 Batch 供模型训练使用等操作。

```
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

for batch in loader:
    print(batch)
```

其核心参数包括：
| 参数 | 说明 | 典型设置 |
| :--- | :--- | :--- |
| **dataset** | 上面定义的 Dataset 实例。 | `MyCustomDataset(...)` |
| **batch_size** | 每个批次包含多少个样本。 | `32`, `64`, `128` |
| **shuffle** | 每个 epoch 开始时是否打乱数据顺序。训练集设为 `True`，验证集设为 `False`。 | `True` (训练), `False` (测试) |
| **num_workers** | 使用多少个子进程来加载数据。`0` 表示主进程加载。Linux/Mac 可设大些 (如 4, 8)，Windows 建议从 0 开始调试。 | `4`, `8` |
| **pin_memory** | 如果为 `True`，数据加载器会将张量复制到 CUDA 固定内存中，加速 GPU 传输。 | `True` (使用 GPU 时) |
| **drop_last** | 如果数据集大小不能被 `batch_size` 整除，是否丢弃最后一个不完整的 batch。 | `True` (BN 层需要固定 batch 大小时) |
| **collate_fn** | 自定义函数，用于将一批样本列表合并成一个批次。处理变长数据（如 NLP 中的句子）时非常有用。 | `default_collate` 或自定义 |