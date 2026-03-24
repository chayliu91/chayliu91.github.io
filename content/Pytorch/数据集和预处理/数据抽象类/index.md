---
title: "数据抽象类"
date: 2026-03-24
draft: false
categories: ["Pytorch", "数据集和预处理"]
tags: ["Pytorch"]
weight: 1
---

# Dataset 基类
`class torch.utils.data.Dataset`  是一个抽象基类，所有自定义数据集都必须继承它。自定义数据集必须实现两个方法:

- `__len__(self)`: 返回数据集的总大小（样本数量）
- `__getitem__(self, index)`: 根据索引 `index` 读取并返回单个样本（可以为: 单值、元组、字典）


```
from torch.utils.data import Dataset
from PIL import Image
import os

# 定义一个自定义数据集类，继承自 PyTorch 的 Dataset 基类
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path)

        label = 0  # 假设标签

        return img, label
```

# TensorDataset
当你已经将数据加载为 PyTorch 张量（例如从 NumPy 转换而来，或生成的模拟数据）时，使用 TensorDataset 是最快的方法。它会将输入的特征张量和标签张量按第一维（样本维度）进行绑定。
```
import torch
from torch.utils.data import TensorDataset, DataLoader

# 假设有 1000 个样本，每个样本 20 个特征
data_x = torch.randn(1000, 20)
# 标签数据: (样本数,) -> 必须是 1 维，长度与 data_x 一致
data_y = torch.randint(0, 2, (10000,))

# 它会自动将 data_x 和 data_y 捆绑在一起
dataset = TensorDataset(data_x, data_y)
x_sample, y_sample = dataset[0]
print(x_sample.shape) # torch.Size([20])
print(y_sample)
```

# ImageFolder
`torchvision.datasets.ImageFolder` 专门用于加载按文件夹结构组织的图像分类数据。它假设每个子文件夹代表一个类别：
```
root_dir/
├── cats/
│   ├── cat_01.jpg
│   └── cat_02.png
├── dogs/
│   ├── dog_01.jpg
│   └── dog_02.jpeg
└── birds/
    └── bird_01.png
```
PyTorch 会自动将 cats 映射为 label 0, dogs 为 1, birds 为 2。

使用于标准的图像分类任务（如 ImageNet 格式数据）：
```
dataset = datasets.ImageFolder(root='./data/images', transform=transform)
```

# ConcatDataset
当你有多个独立的数据集（例如分散在不同文件夹，或者一个是本地数据一个是网络数据），想要把它们拼成一个大数据集进行训练时使用。
```
import torch
from torch.utils.data import ConcatDataset, DataLoader


# 假设有两个包含 1000 个样本的数据集
dataset1 = torch.utils.data.TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
dataset2 = torch.utils.data.TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
combined_dataset = ConcatDataset([dataset1, dataset2])
print(len(combined_dataset))  # 2000
```

# 划分数据集
`torch.utils.data.random_split` 将一个数据集按指定长度随机切分为多个子集（返回 `Subset` 对象）。

按具体数量划分:
```
import torch
from torch.utils.data import Dataset, random_split


# 假设有一个包含 1000 个样本的数据集
dataset = torch.utils.data.TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))

# 定义划分长度：训练集 600, 验证集 200, 测试集 200
# 注意：长度之和必须等于原数据集长度 (600+200+200 = 1000)
train_size = 600
val_size = 200
test_size = 200

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size]
)

print(f"训练集大小: {len(train_dataset)}") # 训练集大小: 600
print(f"验证集大小: {len(val_dataset)}") # 验证集大小: 200
```

按比例 +  固定随机种子划分：

```
import torch
from torch.utils.data import Dataset, random_split


# 假设有一个包含 1000 个样本的数据集
dataset = torch.utils.data.TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))

dataset_len = len(dataset)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 计算各部分长度
train_size = int(dataset_len * train_ratio)
val_size = int(dataset_len * val_ratio)
test_size = dataset_len - train_size - val_size # 确保总和一致，避免余数问题

# 创建生成器并设置种子 (关键步骤)
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator # 传入生成器以固定随机性
)

print(f"训练集大小: {len(train_dataset)}") # 训练集大小: 800
print(f"验证集大小: {len(val_dataset)}") # 验证集大小: 100
```

# 提取子集
虽然 `random_split` 内部使用了 `Subset`，但你也可以手动使用它来根据特定的索引列表提取数据。这比 `random_split` 更灵活，适用于你需要精确控制哪些样本进入训练集的情况（例如：只取前 100 个样本做快速调试）。

```
import torch
from torch.utils.data import Subset

full_dataset = torch.utils.data.TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
# 我想只取索引为 0, 5, 10, ..., 95 的样本 (共20个)
indices = list(range(0, 100, 5))
subset_dataset = Subset(full_dataset, indices)
# 访问子集的第 0 个元素，实际上对应原数据集的 indices[0] (即索引 0)
img, label = subset_dataset[0]
```