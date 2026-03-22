---
title: "张量的基本操作"
date: 2026-03-13
draft: false
categories: ["Pytorch", "Pytorch 基础"]
tags: ["Pytorch"]
weight: 6
---

# 形状变换
改变张量的维度结构，但不改变数据总量（元素个数）。

| 函数 | 特性 | 内存行为 | 适用场景 |
| :--- | :--- | :--- | :--- |
| `view(*shape)` | 严格视图。若内存非连续则报错。内部调用 `Tensor::view` 检查 `is_contiguous()`。 | 零拷贝 | 确定张量连续时（如刚创建的或 `contiguous()` 后），追求极致性能。 |
| `reshape(*shape)` | 智能变形。优先尝试 `view`，若失败（非连续且步长不匹配）则调用 `clone()` 创建副本。 | 可能拷贝 | 通用场景，代码更健壮，不关心底层连续性。 |
| `flatten(start, end)` | 将指定范围的维度展平为一维。本质是计算新形状后调用 `reshape`。 | 零拷贝 (通常) | 全连接层前的展平操作 (`x.flatten(1)`)。 |
| `unflatten(dim, shape)` | `flatten` 的逆操作，将一维拆分为多维。内部验证维度乘积并调用 `reshape`。 | 零拷贝 | 恢复被展平的维度结构。 |

- `view`：要么成功（零拷贝），要么报错。绝不偷偷拷贝
- `reshape/flatten`：
    - 首选策略：尝试修改元数据（`stride/shape`）来实现，零拷贝
    - 降级策略：只有当数学上无法通过调整步长来匹配新形状时（通常是因为切片或复杂的非连续布局），才会分配新内存并拷贝数据

```
import torch

# 创建一个形状为 (2, 3, 4) 的随机张量，内存默认是连续的
x = torch.randn(2, 3, 4)

# view: 必须连续，否则报错
# 这里 x 是连续的，所以可以成功变形为 (6, 4)，且不复制数据（零拷贝）
y1 = x.view(6, 4)
print(y1.shape)  # torch.Size([6, 4])

# permute 操作会改变张量的步长（stride），导致张量在内存中变得不连续
x_t = x.permute(0, 2, 1)

# reshape: 智能变形。如果张量不连续，它会自动创建一个连续的副本并变形
y2 = x_t.reshape(2, 4, 3)

# 能优化成视图，不执行拷贝
print(x_t.data_ptr() == y2.data_ptr())

# flatten: 将指定范围 [start, end) 的维度展平为一维
# 这里将第1维和第2维（即形状中的 3 和 4）合并，结果形状为 (2, 12)
# 只要涉及的维度在内存逻辑上是相邻的，通常也是零拷贝
print(x.shape)  # torch.Size([2, 3, 4])
y3 = x.flatten(1, 2)
print(y3.shape)  # torch.Size([2, 12])
print(y3.data_ptr() == x.data_ptr())  # True

# unflatten: flatten 的逆操作
# 将第1维（大小为12）重新拆分为 (2, 6)。注意：原维度乘积必须等于新形状乘积 (2*6=12)
# 这里是将之前展平的部分恢复，但拆分成了不同的形状组合 (2, 6) 而不是原来的 (3, 4)
y4 = y3.unflatten(1, (2, 6))
print(y4.shape)  # torch.Size([2, 2, 6])


x = torch.randn(2, 3, 4)

# permute(0, 2, 1): 交换维度，形状变为 (2, 4, 3)
# [:, ::2, :]: 对中间维度进行切片（每隔一个取一个）
# 这不仅保持了非连续性，还让步长（stride）变得非常复杂/破碎

y = x.permute(0, 2, 1)[:, ::2, :]
y = y.reshape(2, -1)
print(y.shape)
print(y.stride())  # (6, 1)
# 此时内存布局已经无法用简单的公式映射到新的形状
print(x.data_ptr() == y.data_ptr())  # False


x = torch.randn(2, 3, 4)
# permute(0, 2, 1): 交换最后两维，形状从 (2, 3, 4) 变为 (2, 4, 3)
# 交换后，新的第1维(原第2维)和新的第2维(原第1维)在物理内存上不再连续相邻
x_trans = x.permute(0, 2, 1)
# 要合并这两个维度，它们在内存中必须是连续存放的块
# 由于 permute 导致这两维在物理内存上是交错的（不连续），无法直接通过调整步长来“假装”它们合并了
y_flatten = x_trans.flatten(0, 1)
# flatten 必须执行一次拷贝操作，重新排列数据以使其连续
print(y_flatten.data_ptr() == x_trans.data_ptr())  # False
```

# 增删维度
| 函数 | 功能 | 示例 | 备注 |
| :--- | :--- | :--- | :--- |
| `unsqueeze(dim)` | 在指定位置插入大小为 1 的维度。内部调用 `Tensor::unsqueeze` 修改 `sizes` 和 `strides`。 | `x.unsqueeze(0)`<br>`x.unsqueeze(1)` | 零拷贝，仅修改元数据。<br>- 场景：标量变向量 (5,)→(1, 5) (行) 或 (5, 1) (列)；添加 Batch 维。<br>- 注意：`dim` 可为负数（从后往前数）。 |
| `squeeze(dim)` | 移除指定位置大小为 1 的维度。若未指定 `dim`，移除所有为 1 的维度。内部检查 `sizes[dim] == 1`。 | `x.squeeze(0)`<br>`x.squeeze()` | 零拷贝。<br>- 特性：若指定 `dim` 但该维大小不为 1，不报错，直接返回原张量。<br>- 场景：去除多余的单例维度，如 (1, 32, 1, 1)→(32,)。 |

```
import torch


vec = torch.tensor([1, 2, 3])
print(vec.shape) # torch.Size([3])

# 在第 0 维插入大小为 1 的维度，将形状从 [3] 变为 [1, 3] (行向量)
row_vec = vec.unsqueeze(0)
# 在第 1 维插入大小为 1 的维度，将形状从 [3] 变为 [3, 1] (列向量)
col_vec = vec.unsqueeze(1)
print(row_vec.shape) # torch.Size([1, 3])
print(col_vec.shape) # torch.Size([3, 1])

mat = torch.randn(1, 3, 1, 5, 1,9)
# 这里移除第 0 维 (大小为 1)，其他大小为 1 的维度保留
squeezed = mat.squeeze(0)
print(squeezed.shape) # torch.Size([3, 1, 5, 1, 9])
# squeeze(): 不指定维度时，移除所有大小为 1 的维度
squeezed_all = mat.squeeze()
print(squeezed_all.shape) # torch.Size([3, 5, 9])

```

# 交换维度
重新排列维度的顺序，常用于图像处理 (NCHW ↔ NHWC) 或注意力机制。
| 函数 | 功能 | 内存行为 | 注意 |
| :--- | :--- | :--- | :--- |
| `t()` | 2D 转置 (交换 0 和 1 维)。 | 零拷贝 | 仅限 2D 张量。导致非连续。 |
| `transpose(dim0, dim1)` | 交换任意两个指定维度。 | 零拷贝 | 导致非连续。 |
| `permute(*dims)` | 按任意顺序重排所有维度。 | 零拷贝 | 最灵活，常用于多维权重变换。 |
| `moveaxis(source, dest)` | 将某个轴移动到新位置，其他轴相对顺序不变。 | 零拷贝 | 语义更直观。 |

```
import torch

mat = torch.arange(6).reshape(3, 2)
# t() 示例：仅限 2D 张量的转置
# t() 是 transpose(0, 1) 的快捷方式，交换第 0 维和第 1 维
mat_t = mat.t()
print(mat_t.shape)
# 转置操作仅修改元数据 (stride)，不复制内存，因此通常会导致内存不连续
# 原始矩阵行步长为 2，列步长为 1；转置后行步长变 1，列步长变 2 (非连续)
print(mat_t.is_contiguous())  # False

# 模拟一张图片数据：3 通道，高 320，宽 224 (格式：CHW)
img = torch.randn(3, 320, 224)
# unsqueeze(0) 在第 0 维插入一个维度，模拟添加 Batch 维度
# 形状从 (3, 320, 224) 变为 (1, 3, 320, 224) -> (N, C, H, W)
img = img.unsqueeze(0)
print(img.shape)

# 目标：将格式从 (N, C, H, W) 转换为 (N, W, H, C) 即 NHWC 格式
img_nhwc = img.transpose(1, 3)
print(img_nhwc.shape)  # torch.Size([1, 3, 320, 224])
# transpose 也是零拷贝视图，打乱了内存顺序，因此不连续
print(img_nhwc.is_contiguous())
# 验证原张量未被修改 (视图操作不影响源数据形状)
print(img.shape)

# 移动过程：
# 原顺序: [0, 1, 2, 3] -> [N, C, H, W]
# 移出 1 (C): 剩余 [N, H, W]
# 插入到 3: 变成 [N, H, W, C]
img_nhwc = img.moveaxis(1, 3)
print(img_nhwc.shape) # torch.Size([1, 3, 320, 224])
print(img_nhwc.is_contiguous()) # False
print(img.shape)

# 映射关系:
# 新第 0 维 <- 旧第 0 维 (N)
# 新第 1 维 <- 旧第 2 维 (H)
# 新第 2 维 <- 旧第 3 维 (W)
# 新第 3 维 <- 旧第 1 维 (C)
img_nhwc =  img.permute(0, 2, 3, 1)
print(img_nhwc.shape) # torch.Size([1, 3, 320, 224])
print(img_nhwc.is_contiguous()) # False
print(img.shape)
```
如果后续操作需要连续内存 (如某些卷积层、view 操作或保存文件)，需要在上述操作后调用 `.contiguous()`，这会触发一次物理内存复制。

# 填充
```
torch.nn.functional.pad(input, pad, mode='constant', value=None) → Tensor
```
- `pad`: 填充元组。是从“最后一维”开始写的，每个维度对应两个值：例如对于 4D 张量 `(N, C, H, W)`:填充顺序 W → H，`pad = (left, right, top, bottom)`
- `mode`: 填充模式:
  - `'constant'`: 常数填充（默认）
  - `'reflect'`: 镜像反射（以边界为轴反射，不包含边界像素本身）
  - `'replicate'`: 边缘复制（重复边界像素）
  - `'circular'`: 循环填充。
- `value`: 当 mode='constant' 时使用的填充值

这是 PyTorch 中用于对张量进行填充的核心函数。它不仅仅是“补零”，还支持多种填充模式，广泛应用于卷积神经网络（CNN）的边界处理、图像增强等场景。

```
import torch
import torch.nn.functional as F

# 创建一个形状为 (1, 2) 的张量: [[1., 1.]]
x = torch.ones(1, 2)
# 对于 2D 张量 (Row, Col)，pad=(1, 2, 3, 4) 解析如下：
#   - 最后两个数 (3, 4): 对应第 0 维 (行/高)。上方补 3 行，下方补 4 行
#   - 前两个数 (1, 2):   对应第 1 维 (列/宽)。左侧补 1 列，右侧补 2 列
x_padded = F.pad(x, pad=(1, 2, 3, 4), mode= "constant", value=0)
# 计算新形状：
#   - 行数: 1 (原) + 3 (上) + 4 (下) = 8
#   - 列数: 2 (原) + 1 (左) + 2 (右) = 5
print(x_padded.shape)  # torch.Size([8, 5])
print(x_padded)

# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 1., 1., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])

x = torch.arange(6).reshape(2, 3).float()
print(x)

# tensor([[0., 1., 2.],
#         [3., 4., 5.]])

# pad=(1, 2): 左侧补 1 列，右侧补 2 列
# 规则：以边界像素为轴进行反射，不包含边界像素本身
x_padded = F.pad(x, pad=(1, 2), mode= "reflect")
print(x_padded)
# tensor([[1., 0., 1., 2., 1., 0.],
#         [4., 3., 4., 5., 4., 3.]])

# pad=(1, 2): 左侧补 1 列，右侧补 2 列
# 规则：直接重复边界像素
x_padded = F.pad(x, pad=(1, 2), mode= "replicate")
print(x_padded)
# tensor([[0., 0., 1., 2., 2., 2.],
#         [3., 3., 4., 5., 5., 5.]])

# pad=(1, 2): 左侧补 1 列，右侧补 2 列
# 规则：像环一样，从另一头取值
x_padded = F.pad(x, pad=(1, 2), mode= "circular")
print(x_padded)

# tensor([[2., 0., 1., 2., 0., 1.],
#         [5., 3., 4., 5., 3., 4.]])
```

# 复制
| 特性 | `expand` / `expand_as` | `repeat` |
| :--- | :--- | :--- |
| **核心机制** | **逻辑视图**：**零拷贝**。仅修改元数据（stride），将维度为 1 的轴“虚拟”拉伸。 | **物理复制**：分配新内存，真正执行数据拷贝（memcpy），重复存储多份数据。 |
| **参数含义** | `shape`: **目标形状** (Target Shape)。<br>需与原张量兼容（原维度为 1 或相等）。 | `sizes`: **重复倍数** (Repetition Counts)。<br>表示每个维度重复多少次 (如 3, 2)。 |
| **内存行为** | **不分配新内存**。<br>返回原数据的视图 (View)，修改结果会影响源数据。 | **分配新内存**。<br>内存占用 = 原大小 × 重复倍数。 |
| **数据限制** | **严格限制**：原张量中非 1 的维度必须与目标形状完全一致，否则报错。 | **无限制**：可对任意维度的任意大小进行重复。 |
| **典型场景** | 广播计算 (Broadcasting)、批量加减偏置 (Bias)、Attention Mask。 | 数据增强 (平铺)、需要独立副本且后续会修改数据的场景。 |
| **性能提示** | **极快** (仅修改指针和 stride)。 | **高开销** (显存消耗大，涉及大量数据拷贝)。 |
| **代码示例** | `x.expand(3, 5)`<br>`x.expand_as(y)` | `x.repeat(3, 2)` |


```
import torch

base = torch.tensor([[1, 2]])
# 步长 (stride): 第 0 维跨 2 个元素，第 1 维跨 1 个元素
print(base.stride()) # (2, 1)
# 将第 0 维从 1 扩展到 3。底层通过将第 0 维的 stride 设为 0 实现
# 内存中只有一份 [1, 2] 的数据，读取时重复访问
exp = base.expand(3, 2)
print(exp)
# tensor([[1, 2],
#         [1, 2],
#         [1, 2]])

x = torch.arange(3).reshape(1, 3)
y = torch.randn(3, 3)
# 将 x 扩展为与 y 相同的形状 (3, 3)
# 原理：零拷贝。将 x 第 0 维的 stride 设为 0，逻辑上重复该行 3 次
# 注意：并没有在内存中真正复制数据，exp 只是 x 的一个视图 (View)
exp = x.expand_as(y)
print(exp)

# tensor([[0, 1, 2],
#         [0, 1, 2],
#         [0, 1, 2]])

print(exp.shape)
# torch.Size([3, 3])


print(exp.shape) # torch.Size([3, 2])
print(exp.stride()) # (0, 1)

# 真正地在内存中复制数据。形状 [1, 2] -> 在 0 维重复 3 次，1 维重复 2 次 -> [3, 4]
rep = base.repeat(3, 2)
print(rep)
# tensor([[1, 2, 1, 2],
#         [1, 2, 1, 2],
#         [1, 2, 1, 2]])

# 验证内存地址：repeat 创建了新内存，所以地址不同
print(rep.data_ptr() == base.data_ptr())  # False

# 非单例维度 (大小不为 1 的维度) 必须与目标形状完全匹配，不能通过 expand 改变其大小
exp2 = base.expand(2, 4)  # The expanded size of the tensor (4) must match the existing size (2) at non-singleton dimension 1.  Target sizes: [2, 4].  Tensor sizes: [1, 2]
```

# 合并
用于将多个张量组合成一个更大的张量，常用于批量数据组装或特征拼接。

| 函数 | 功能描述 | 关键参数 | 核心区别与要求 |
| :--- | :--- | :--- | :--- |
| **`cat`** | **拼接**：沿**现有**维度连接张量。 | `dim`: 指定沿哪个维度拼接。 | **形状要求**：除拼接维度外，其他所有维度必须**完全一致**。<br>**输出维度**：与输入张量维度数相同，仅指定维度长度相加。 |
| **`stack`** | **堆叠**：沿**新**维度连接张量。 | `dim`: 指定新维度插入的位置。 | **形状要求**：所有输入张量的形状必须**完全一致**。<br>**输出维度**：比输入张量维度数 **+1**。 |
| **`unbind`** | **解绑**：移除指定维度，返回该维度上所有切片的列表。 | `dim`: 指定要移除的维度。 | **逆操作**：是 `stack` 的逆操作。<br>**返回值**：返回一个张量列表 (List)，而非单个张量。 |

```
import torch

a = torch.ones(2, 3)
b = torch.zeros(2, 3)

# 沿第 0 维 (行方向) 拼接
# 逻辑：将 b 接在 a 的下面
# 形状变化：[2, 3] + [2, 3] -> [4, 3] (第0维相加: 2+2=4)
c1 = torch.cat([a, b], dim=0)
print(c1.shape)  # torch.Size([4, 3])
print(c1)

# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [0., 0., 0.],
#         [0., 0., 0.]])


# 沿第 1 维 (列方向) 拼接
# 逻辑：将 a 接在 b 的右边 (注意列表顺序是 [b, a])
# 形状变化：[2, 3] + [2, 3] -> [2, 6] (第1维相加: 3+3=6)
c2 = torch.cat([b, a], dim=1)
print(c2.shape) # torch.Size([2, 6])
print(c2)

# tensor([[0., 0., 0., 1., 1., 1.],
#         [0., 0., 0., 1., 1., 1.]])


# 在第 0 维插入新维度进行堆叠
# 逻辑：创建一个新轴，a 是索引 0，b 是索引 1
# 形状变化：[2, 3] -> [2, 2, 3] (新增第0维，长度为输入张量个数 2)
c3 = torch.stack([a, b], dim=0)
print(c3.shape)  # torch.Size([2, 2, 3])
print(c3)

# tensor([[[1., 1., 1.],
#          [1., 1., 1.]],

#         [[0., 0., 0.],
#          [0., 0., 0.]]])


# 在第 1 维插入新维度进行堆叠
# 逻辑：原第0维保持不动，在中间插入新轴
# 形状变化：[2, 3] -> [2, 2, 3] (新增第1维，长度为 2)
# 此时结构变为：[ [a的第0行, b的第0行], [a的第1行, b的第1行] ]
c4 = torch.stack([a, b], dim=1)
print(c4.shape)  # torch.Size([2, 2, 3])
print(c4)

# tensor([[[1., 1., 1.],
#          [0., 0., 0.]],

#         [[1., 1., 1.],
#          [0., 0., 0.]]])


# 在第 2 维 (最后一个维度/列深度) 插入新维度进行堆叠
# 输入形状: a=[2, 3], b=[2, 3]
# 操作逻辑: 保持原有的第0维(行)和第1维(列)不变，在每一对对应的元素后面“追加”一个新维度
# 形状变化: [2, 3] -> [2, 3, 2]
#   - 第0维 (行): 保持 2
#   - 第1维 (列): 保持 3
#   - 第2维 (新): 长度为 2 (因为堆叠了2个张量)
c5 = torch.stack([a, b], dim=2)
print(c5.shape)  # torch.Size([2, 3, 2])
print(c5)

# tensor([[[1., 0.],
#          [1., 0.],
#          [1., 0.]],

#         [[1., 0.],
#          [1., 0.],
#          [1., 0.]]])

# 移除 c3 的第 0 维
# 逻辑：这是 stack(dim=0) 的逆操作。将第 0 维“拆开”，返回该维度下所有切片的列表
# 结果：一个包含 2 个张量的列表，分别对应原来的 a 和 b
list_a_b = torch.unbind(c3, dim=0)
print(len(list_a_b)) # 2
print(list_a_b[0])
print(list_a_b[1])

# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
```

# 分割
| 函数 | 功能描述 | 关键参数 | 核心区别与行为 |
| :--- | :--- | :--- | :--- |
| **`split`** | **按大小分割**：<br>将张量按指定的**每块大小**进行切分。 | `size`: 每块的大小。<br>`dim`: 分割维度。 | **行为**：尽可能按 `size` 切割。<br>**尾部处理**：如果无法整除，最后一片会小于 `size` (包含剩余元素)。<br>**返回**：元组 (Tuple) 或 列表 (若 `size` 为整数则返回元组，若 `size` 为列表则返回列表)。 |
| **`chunk`** | **按份数均分**：<br>将张量尽量均匀地切成**指定份数**。 | `chunks`: 想要切成的份数。<br>`dim`: 分割维度。 | **行为**：计算每块大致大小，尽量均分。<br>**尾部处理**：如果无法整除，最后一片可能会略小 (但前面的片大小一致)。<br>**返回**：元组 (Tuple)。 |

```
import torch

data = torch.randn(4, 3)

# 沿第 0 维 (行) 切割，每块大小为 1
# 总行数 4 / 每块 1 = 4 块，刚好整除
split_data = torch.split(data, split_size_or_sections=1, dim=0)
print(len(split_data)) # 4

# 沿第 0 维 (行) 切割，每块大小为 3
# 总行数 4 / 每块 3 = 1 块余 1 行
# 第一块取 3 行，剩下的所有行 (1行) 作为最后一块
split_data = torch.split(data, split_size_or_sections=3, dim=0)
print(len(split_data)) # 2
print(split_data[0].shape) # torch.Size([3, 3])
print(split_data[1].shape) # torch.Size([1, 3])

data = torch.randn(10, 3) # 第0维长度为 10
# 传入列表 [2, 3, 5] -> 2 + 3 + 5 = 10 (刚好匹配)
result = torch.split(data, split_size_or_sections=[2, 3, 5], dim=0)
print(len(result))       # 3
print(result[0].shape)   # torch.Size([2, 3])
print(result[1].shape)   # torch.Size([3, 3])
print(result[2].shape)   # torch.Size([5, 3])


# 沿第 1 维 (列) 切割，切成 3 份
# 总列数 3 / 3 份 = 每份 1 列，刚好整除
chunk_data = torch.chunk(data, chunks=3, dim=1)
print(len(chunk_data)) # 3

# 沿第 1 维 (列) 切割，切成 2 份
# 策略：前面的块尽可能大且相等，最后一块容纳剩余部分
# 计算逻辑：3 = 2 + 1。第一块 2 列，第二块 1 列
chunk_data = torch.chunk(data, chunks=2, dim=1)
print(len(chunk_data)) # 2
print(chunk_data[0].shape) # torch.Size([4, 2])
print(chunk_data[1].shape) # torch.Size([4, 1])

# 总列数 3 / 5 份 -> 份数 > 维度大小
# 最多只能切出 3 块 (每块 1 列)，多余的请求被忽略
chunk_data = torch.chunk(data, chunks=5, dim=1)
print(len(chunk_data))
```

# Numpy 和 Tensor 互操作

## Tensor 与 ndarray 的相互转换

当 Tensor 位于 CPU 上时，PyTorch Tensor 和 NumPy ndarray 可以指向同一块物理内存地址。
- 双向同步：修改 Tensor 的值，NumPy 数组会立即变化；反之亦然
- 零开销：转换过程不复制数据，仅创建一个新的视图（View），速度极快

限制：
- 仅适用于 CPU Tensor
- 数据类型必须是 NumPy 支持的数值类型（如 float32, int64）。不支持字符串或复杂对象
- 如果 Tensor 需要梯度（requires_grad=True），转换后的 NumPy 数组将无法追踪梯度（因为 NumPy 没有自动微分机制），但内存依然共享

当 Tensor 位于 CUDA (GPU) 上时，无法直接共享内存。这是因为 NumPy 只能管理 CPU 内存，无法直接访问 GPU 显存。必须先调用 `.cpu()` 将数据从显存拷贝到内存，然后再转换为 NumPy 数组。此时生成的是数据的副本（Copy），不再共享内存。修改副本不会影响原 GPU Tensor。

```
import torch
import numpy as np


a = np.arange(4).reshape(2, 2)
print(a)

# 从 NumPy 创建 PyTorch Tensor (零拷贝，共享内存)
# torch.from_numpy() 创建的 tensor 与原 numpy 数组共享同一块内存地址
t = torch.from_numpy(a)
# 结果为 True，证明两者指向同一物理内存
print(t.data_ptr() == a.__array_interface__["data"][0]) # True

t[0][0] = 100
print(a[0]) # [100   1]

# .cpu() 将数据从 GPU 拷贝到 CPU 内存
# .numpy() 将 CPU tensor 转换为新的 numpy 数组
# 这个过程发生了内存复制，因此地址不同
t = torch.tensor(1.0, device="cuda")
a = t.cpu().numpy()
print(t.data_ptr() == a.__array_interface__["data"][0])  # False

a = t.numpy() # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

## Tensor 与 ndarray 的核心差异
| 特性 | PyTorch Tensor | NumPy ndarray |
| :--- | :--- | :--- |
| **主要用途** | 深度学习训练、推理（支持自动微分、GPU 加速） | 科学计算、数据分析、通用数值处理 |
| **设备支持** | 支持 CPU 和 GPU (CUDA/MPS) | 仅支持 CPU |
| **自动微分** | 支持 (`requires_grad`, `backward()`) | 不支持 (需配合 JAX 或 Autograd 库) |
| **动态图** | 支持动态计算图 (Define-by-Run) | 静态执行 (即时计算) |
| **内存布局** | 支持 `contiguous` 和 `non-contiguous` (如切片后) | 同样支持，但 PyTorch 对非连续内存的处理更灵活 |
| **默认浮点类型** | **`torch.float32`** (单精度，节省显存/加速) | **`np.float64`** (双精度，追求最高计算精度) |
| **默认整型** | `torch.int64` | `np.int64` |
| **生态整合** | 与 `torch.nn`, `torch.optim` 深度整合 | 与 `pandas`, `scikit-learn`, `matplotlib` 深度整合 |
| **互操作注意** | 从 NumPy 导入时**保留原精度** (易意外变为 float64) | 从 Tensor 导出时**保留原精度** (易意外变为 float32) |

```
import torch
import numpy as np


t = torch.tensor(1)
print(t.dtype) # torch.int64

t = torch.tensor(1.0)
print(t.dtype)  # torch.float32

a = np.array(1)
print(a.dtype) # int64

a = np.array(1.0)
print(a.dtype)  # float64

# torch.from_numpy() 不会改变数据类型，而是严格继承
t = torch.from_numpy(a)
print(t.dtype)  # torch.float64
```
