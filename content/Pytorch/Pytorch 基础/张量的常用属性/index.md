---
title: "张量的常用属性"
date: 2026-03-13
draft: false
categories: ["Pytorch", "Pytorch 基础"]
tags: ["Pytorch"]
weight: 2
---

# 形状与维度
| 属性/方法 | 类型 | 功能描述 | 参数支持 | 返回值示例 (假设 `x` 为 $3 \times 4 \times 5$ 的**连续**张量) |
| :--- | :--- | :--- | :--- | :--- |
| **`.shape`** | **属性** | 返回张量的形状。<br>本质是 `torch.Size` 对象。 | **无** | `torch.Size([3, 4, 5])` |
| **`.size()`** | **方法** | 返回形状。<br>支持获取指定维度的长度。 | **可选** `dim` | 1. `x.size()` $\to$ `torch.Size([3, 4, 5])`<br>2. `x.size(1)` $\to$ `4` |
| **`.ndim`** | **属性** | 返回张量的维度数量（阶数）。 | **无** | `3` |
| **`.numel()`** | **方法** | 返回张量中**元素的总个数**。<br>等价于所有维度大小的乘积 ($3 \times 4 \times 5$)。 | **无** | `60` |

```
import torch


t = torch.randn(2, 3, 4)

print(t.shape)    # torch.Size([2, 3, 4])
print(t.size())   # torch.Size([2, 3, 4])
# 3 (获取第1个维度的长度)
print(t.size(1))  # 3
print(t.ndim)     # 3 (这是一个3维张量)
```

使用 `Tensor.numel()` (方法) 或 `Tensor.size()` (配合乘积)可以回张量中元素的总个数：
```
import torch
import math

x = torch.randn(3, 4, 5)

# 直接返回张量中元素的总数 (number of elements)，效率最高且语义最清晰
print(x.numel())  # 60

# 使用 Python 标准库 math.prod() 计算尺寸元组的乘积
print(math.prod(x.size()))  # 60
```

# 数据类型
PyTorch 支持多种数据类型，必须显式或隐式匹配才能进行运算，`.dtype` 可以用于返回数据类型。

PyTorch 常用数据类型：

| 数据类型类别 | PyTorch 类型常量 (备注) | 别名 (简写) | 位宽 | 核心应用场景与特点 |
| :--- | :--- | :---: | :---: | :--- |
| **浮点型** | `torch.float32` (**默认浮点**) | `torch.float` | 32-bit | **通用计算**。深度学习最常用，平衡数值稳定性与内存。 |
|  | `torch.float64` | `torch.double` | 64-bit | **高精度计算**。科学计算专用，内存大，计算慢。 |
| | `torch.float16` | `torch.half` | 16-bit | **加速训练**。混合精度训练 (AMP)，省显存，需防溢出。 |
| | `torch.bfloat16` | `torch.bfloat16` | 16-bit | **新一代加速**。动态范围大，适合大模型，不易溢出。 |
| **整型** | `torch.int64` (**默认索引**) | `torch.long` | 64-bit | **索引与标签**。张量索引、Loss 目标标签**必须**用此类型。 |
| | `torch.int32` | `torch.int` | 32-bit | **常规整数**。一般计数，比 int64 节省内存。 |
| | `torch.int8` | `torch.char` | 8-bit | **量化压缩**。模型量化、图像像素值 (0-255)。 |
| **布尔型** | `torch.bool` | - | 8-bit* | **掩码操作**。逻辑判断、布尔索引、条件筛选。 |

```
import torch

# 在 PyTorch 中，浮点数的默认精度是 32 位 (float32)
a = torch.tensor([1.5])

# 在 PyTorch 中，整数的默认精度是 64 位 (int64)，这也叫 "long"
b = torch.tensor([1, 2, 3])

print(a.dtype, b.dtype)  # torch.float32 torch.int64
```

## 数据类型转换
| 方法 | 语法示例 | 核心特点 | 推荐场景 |
| :--- | :--- | :--- | :--- |
| **`.to()`** | `x.to(torch.float16)`<br>`x.to('cuda')` | **全能首选**。可同时转换设备和数据类型；支持字符串简写；链式调用友好。 | **所有场景**。特别是涉及 GPU 迁移、混合精度训练时。 |
| **类型函数** | `x.float()`<br>`x.long()` | **简洁快捷**。无需 `torch` 前缀；仅转换数据类型，不改变设备。 | 快速脚本、交互式调试、仅需简单类型转换时。 |
| **`.type()`** | `x.type(torch.int8)`<br>`x.type()` (查询) | **旧式写法**。功能同 `.to(dtype)`；不带参数时可查询类型字符串。 | **查询类型**或兼容旧代码。 |
| **`.half()`** | `x.half()` | **专用简写**。等价于 `.to(torch.float16)`。 | **混合精度训练** (FP16) 以节省显存/加速。 |
| **`.double()`** | `x.double()` | **专用简写**。等价于 `.to(torch.float64)`。 | **高精度科学计算**、需要极高数值稳定性时。 |

```
import torch

# 默认情况下，torch.randn 生成的数据类型是 torch.float32
x = torch.randn(5, 5)
print(x.dtype)  # torch.float32

# .half() 是将 float32 转换为 float16 (半精度) 的快捷方法
x_half = x.half()
print(x_half.dtype) # torch.float16

# .double() 是将当前类型转换为 float64 (双精度) 的快捷方法
x_double = x_half.double()
print(x_double.dtype) # torch.float64

# 使用 .to() 进行强制类型转换
probs = torch.tensor([0.9, 1.2, 3.9])
indices = probs.to(torch.long)
print(indices.dtype) # torch.int64

count = torch.tensor(5)
total = torch.tensor(20)
# 为了得到准确的浮点结果 (0.25)，必须先将其中一个操作数转换为浮点型
accuracy = count.to(torch.float32) / total
print(accuracy.dtype)  # torch.float32


# 显式指定 dtype=torch.float32 (虽然不写也是默认值，但显式写出代码更清晰)
weights = torch.randn(10, 10, dtype=torch.float32)

mask = weights > 0.0         # 自动生成 torch.bool
print(mask.dtype)  # torch.bool
```

另外需要注意类型提升（Type Promotion） 机制：当不同精度的浮点数进行运算时，结果会自动转换为精度更高的类型，以防止精度丢失。
```
import torch

# 当两个不同数据类型的 Tensor 进行运算时，PyTorch 会将它们转换为共同的“更宽”类型。
# 规则：float32 + float64 -> 结果自动提升为 float64
# 原因：如果降级为 float32，b 中的高精度信息会丢失；为了保持数值精度，系统选择最高精度。
a = torch.tensor([1.0], dtype=torch.float32)
b = torch.tensor([2.0], dtype=torch.float64)
c = a + b
print(c.dtype) # torch.float64
```

# 设备信息
在 PyTorch 中，`torch.device` 对象用于指定张量（Tensor）或模型所在的设备（CPU 或 GPU）。主要有 2 种 常见的表示方法：
- 直接使用字符串表示：
```
device = "cpu"
device = "cuda"       # 默认使用第一块显卡 (cuda:0)
device = "cuda:0"     # 显式指定第 0 号显卡
device = "cuda:1"     # 指定第 1 号显卡
```
-  使用 `torch.device` 类构造函数：
```
device = torch.device("cpu")
device = torch.device("cuda")      # index 默认为 0
device = torch.device("cuda", 0)   # 显式传入 index
device = torch.device("cuda:1")    # 也可以直接传完整字符串
```

`torch.device` 对象都包含两个核心属性：

| 属性 | 类型 | 说明 |
| :--- | :--- | :--- |
| `.type` | `str` | 设备类型，只能是 `'cpu'` 或 `'cuda'`。 |
| `.index` | `int` 或 `None` | 设备索引。对于 `'cpu'` 是 `None`；对于 `'cuda'` 是整数（如 `0`, `1`）。 |

`Tensor.device` 返回张量所在的设备对象。可以使用 `Tensor.to` 操作转换设备类型。
```
import torch

# 在 CPU 上创建一个包含 [1.0, 2.0] 的浮点型张量
# 默认数据类型为 torch.float32
x = torch.tensor([1.0, 2.0], device="cpu")
print(x.device) # cpu
print(x.dtype)  # torch.float32

# 使用 .to() 方法将张量从 CPU 移动到指定的 CUDA 设备 (显卡 0)
x = x.to(torch.device("cuda:0"))
print(x.device) # cuda:0
print(x.device.type) # cuda
print(x.device.index) # 0

# 同时改变数据类型和设备
x = x.to(dtype=torch.int32, device=torch.device("cpu"))
print(x.device) # cpu
print(x.dtype)  # torch.int32

# 基于张量 x 所在的设备，创建一个新的标量张量 y
y = torch.tensor(1.0, device=x.device)
print(y.device) # cpu
```