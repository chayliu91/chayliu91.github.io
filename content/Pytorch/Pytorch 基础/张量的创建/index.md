---
title: "张量的创建"
date: 2026-03-13
draft: false
categories: ["Pytorch", "Pytorch 基础"]
tags: ["Pytorch"]
weight: 3
---

# 直接创建

## torch.tensor
使用工厂函数 `torch.tensor()` 直接创建张量:

```
def tensor(data, *, dtype=None, device=None, pin_memory=False, requires_grad=False)
```

```
import torch


t1 = torch.tensor(9)
print(t1.dtype)   # 输出: torch.int64 (默认整型为 64 位)
print(t1.shape)   # 输出: torch.Size([]) (0 维，即标量)
print(t1.ndim)    # 补充: 0 (验证维度数)

t2 = torch.tensor(3.0)
print(t2.dtype)   # 输出: torch.float32 (默认浮点为 32 位)
print(t2.shape)   # 输出: torch.Size([]) (0 维，即标量)

t3 = torch.tensor([1.0, 2.0])
print(t3.dtype)   # 输出: torch.float32
print(t3.shape)   # 输出: torch.Size([2]) (1 维张量)

original_list = [5.0, 6.0]
t4 = torch.tensor(original_list)
original_list[0] = 999.0  # 修改原始数据
print(t4[0].item())       # 输出: 5.0 (证明 t4 是独立副本，未受影响)

x = torch.tensor([1.0, 2.0], requires_grad=True) # 一个需要梯度的张量
y = torch.tensor(x)      # 用 tensor() 包装它
print(y.requires_grad)   # 输出: False (源码中调用了 detach_(), 默认不追踪梯度)
                         # 注意: 此时还会抛出一个 UserWarning 建议改用 clone()

t5 = torch.tensor([1, 2.5]) # 列表包含 int 和 float
print(t5.dtype)             # 输出: torch.float32 (自动提升为浮点以保留精度)
print(t5)                   # 输出: tensor([1.0000, 2.5000]) (整数部分被转换)()
```

其行为逻辑为：

**数据拷贝:** 它接收用户提供的数据，并在内存中分配新的存储空间，将数据拷贝进去，修改原始数据不会影响生成的 Tensor

**类型推断:**
- 用户指定优先: 如果传入了 `dtype` 参数，直接使用，不进行推断
- 动推断规则 (当 `dtype=None` 时):
  - 输入是 `int (9)` 推断为 `torch.int64 (LongTensor)`
  - 输入是 `float (9.0)`  推断为 `torch.float32(FloatTensor)`
  - 输入是复杂结构 (list/tuple/np.ndarray)：
    - 递归扫描所有元素以确定“最宽”的类型
    -  如果列表中混合了 `int` 和 `float`，整体推断为 `float32` (整型部分会被转为浮点)
    -  空列表: `[]` 默认推断为 `float32`

**维度确定:**
维度确定：它会根据输入数据的结构确定 `shape`
- 输入标量: 0 维张量 (Scalar)，shape 为 `torch.Size([])`
- 输入一维列表/元组:  1 维张量
- 输入嵌套列表:  多 维张量
- 输入 NumPy 数组：直接映射 NumPy 的 `shape` 属性到 Tensor

**梯度切断:**
- 除非显示传递了 `requires_grad=True` 参数，否则即使输入数据是一个需要梯度的 `Tensor (requires_grad=True)`，`torch.tensor` 创建的新 Tensor 默认 `requires_grad=False`
- 如果输入是 Tensor，源码会发出 UserWarning，建议改用 `.clone().detach()`

**设备默认值:**
- 如果未指定 `device` 参数，默认强制在 CPU 上创建，即使当前 CUDA 可用


## torch.Tensor
需要注意与 `torch.Tensor` 的区别，`torch.Tensor` 是一个类，默认只创建浮点型张量:
```
import torch

# 传入单个整数 -> 被解析为尺寸
t1 = torch.Tensor(9)
print(t1)
print(t1.dtype) # torch.float32
print(t1.shape) # torch.Size([9])

# 传入列表 -> 被解析为数据
t2 = torch.Tensor([9])
print(t2)       # tensor([9.])
print(t2.dtype) # torch.float32
print(t2.shape) # torch.Size([1])

# 传入嵌套列表 -> 被解析为多维数据
t3 = torch.Tensor([[1, 2],[3,4]])
print(t3)
# tensor([[1., 2.],
#         [3., 4.]])
print(t3.dtype)  # torch.float32
print(t3.shape)  # torch.Size([2, 2])

# 传入单个浮点数 -> 报错
# 既不是合法的尺寸定义，也不是一个序列
t4 = torch.Tensor(9.0)  # TypeError: new(): data must be a sequence (got float)

```

`torch.Tensor` 能够自动根据传入参数的类型切换行为模式：
- 传入纯整数，将整数当成尺寸，分配内存并执行随机初始化
- 传入可迭代对象，采用数据拷贝模式，它不会把列表里的数字当成尺寸，而是把它们当成具体的数据内容：
  - 计算所需内存大小，分配内存
  - 拷贝数据
  - 强制将类型转换成默认的浮点类型


但在现代 PyTorch 开发规范中，推荐使用 `torch.tensor()` 创建张量，原因只有一个：类型推断的确定性。


# 从外部数据创建

| 方法 | 是否共享内存 | 适用场景 | 注意事项 |
| :--- | :--- | :--- | :--- |
| `torch.tensor(data)` | **否** (总是复制) | 通用场景，从 List 或非 NumPy 数据创建 | 最安全，但可能有额外的内存拷贝开销 |
| `torch.from_numpy(ndarray)` | **是** | 处理大型 NumPy 数据集 | 修改需小心；若原 np 数组需保持不变，则不能直接用于梯度计算（因为反向传播会修改原数组） |
| `torch.as_tensor(data)` | **视情况而定** | 高性能数据加载管道 | 如果输入已经是 Tensor 则返回自身，否则复制。若输入是 NumPy 数组，行为类似 `from_numpy` (共享内存) |

```
import torch
import numpy as np

# torch.tensor() 总是分配新内存并复制数据，切断与源数据的联系
lst = [1, 2, 3, 4, 5]
t = torch.tensor(lst)
lst[-1] = 100
print(lst) # [1, 2, 3, 4, 100]
print(t) # tensor([1, 2, 3, 4, 5])

# from_numpy 不复制数据，Tensor 直接指向 NumPy 的内存地址
array = np.array([1, 2, 3, 4, 5])
t = torch.from_numpy(array)
array[-1] = 100
print(array) # [1, 2, 3, 4, 100]
print(t) # [1, 2, 3, 4, 100]
print(f"共享内存? {t.storage().data_ptr() == array.ctypes.data}") # True

# 输入是 List (行为同 torch.tensor，必须复制)
lst = [1, 2, 3, 4, 5]
t = torch.as_tensor(lst)
lst[-1] = 100
print(lst) # [1, 2, 3, 4, 100]
print(t) # tensor([1, 2, 3, 4, 5])

# 输入是 NumPy (行为同 from_numpy，共享内存)
array = np.array([1, 2, 3, 4, 5])
t = torch.as_tensor(array)
array[-1] = 100
print(array) # [  1   2   3   4 100]
print(t) # tensor([  1,   2,   3,   4, 100])
print(f"共享内存? {t.storage().data_ptr() == array.ctypes.data}") # True

# 输入已是 Tensor (行为同引用，零拷贝)
t1 = torch.tensor([100, 200, 300])
t2 = torch.as_tensor(t1)
print(f"是否是同一个对象? {t1 is t2}") # True! 完全没复制
```

# 依数值创建
## 全0/全1


```
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor

torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```
| 方法 | 内存初始化行为 | 适用场景 | 注意事项 |
| :--- | :--- | :--- | :--- |
| `torch.zeros(*size)` | 全部置为 0 | 初始化偏置、注意力掩码、梯度累加器 | 支持指定 `dtype` 和 `device`；默认浮点型为 `float32` |
| `torch.ones(*size)` | 全部置为 1 | 初始化缩放因子、乘法单位元、全连接层初始权重 | 同上；注意在乘法运算中可能触发广播机制 |

```
import torch

# 标量 (0维 Tensor)
t0 = torch.zeros(())

# 传入整数 0，表示创建一个长度为 0 的一维数组
# 这与 t0 不同！t0 是“有值但没维度”，t1 是“有维度但没值”
t1 = torch.zeros(0)

# 标准二维张量 (元组传参)
t2 = torch.zeros((2, 3))

# 标准二维张量 (列表传参)
t3 = torch.zeros([3, 3])

# 高维张量 (可变参数传参)
# PyTorch 会将这些参数依次解析为各个维度的长度
# 等价于 torch.zeros((2, 3, 4, 5))
t4 = torch.zeros(2, 3, 4, 5)

print(t0) # tensor(0.)
print(t1) # tensor([])

print(t0.shape) # torch.Size([])
print(t1.shape) # torch.Size([0])
print(t2.shape) # torch.Size([2, 3])
print(t3.shape) # torch.Size([3, 3])
print(t4.shape) # torch.Size([2, 3, 4, 5])
```

## 特定值填充
当需要初始化为 0 或 1 以外的特定常数（如 -1 表示无效索引，或极小值防止除零）时使用:
```
torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

| 方法 | 内存初始化行为 | 适用场景 | 注意事项 |
| :--- | :--- | :--- | :--- |
| `torch.full(*size, fill_value)` | **全部置为指定值** | 初始化特定常数标记、占位符、默认状态 | `fill_value` 会被强制转换 (Cast) 为 Tensor 的 `dtype`，可能导致精度丢失或截断 |

```
import torch


# 在 NLP 处理中，常用 -1 表示 Padding 位置
pad_mask = torch.full((2, 5), -1, dtype=torch.long)
print(f"无效索引掩码:\n{pad_mask}")

# 防止除零或 Softmax 泄露
attn_mask = torch.full((3, 3), float('-inf'), dtype=torch.float32)
# 模拟：允许对角线关注 (设为 0)
attn_mask.fill_diagonal_(0.0)
print(attn_mask)

# 注意隐式类型转换
val = 3.9
t_int = torch.full((2,), val, dtype=torch.int32)
print(f"\n尝试将 3.9 填充到 int32: {t_int}")
```

## 未初始化内存
这类方法只分配内存空间，不进行任何数值写入。它们是速度最快的，但也是最危险的:
```
torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format) → Tensor
```

| 方法 | 内存初始化行为 | 适用场景 | 注意事项 |
| :--- | :--- | :--- | :--- |
| `torch.empty(*size)` | **不初始化** (保留内存垃圾值) | **高性能场景**：确定会立即完全覆写数据时 (如自定义算子输出) | 最快，但内容是随机的旧内存数据；若未完全覆写直接使用会导致严重 Bug |

```
import torch
import time

size = (1000, 1000, 100) # 约 400MB 数据 (float32)
# 方法 A: torch.zeros (安全，但需要操作系统清零内存)
start = time.time()
t_zero = torch.zeros(size)
time_zero = time.time() - start

# 方法 B: torch.empty (极快，仅分配指针，内存内容未定义)
start = time.time()
t_empty = torch.empty(size)
time_empty = time.time() - start

print(f"zeros 耗时: {time_zero:.4f} 秒")
print(f"empty 耗时: {time_empty:.4f} 秒")
print(f"加速比: {time_zero / time_empty:.2f}x")

# empty 创建的 Tensor 包含之前内存残留的数据，完全随机
print(f"\nempty 前 5 个值 (随机垃圾): {t_empty.flatten()[:5]}")
```


# 依概率分布创建
在深度学习、强化学习及生成模型（如 VAE、GAN）中，经常需要从特定的概率分布中采样数据或初始化参数。PyTorch 提供了 `torch.distributions` 模块以及部分内置随机函数来实现这一目标。

所有 `torch.distributions` 对象均支持以下通用方法：
- `.sample()`: 采样（不可导，梯度无法回传到分布参数）
- `.rsample()`: 重参数化采样（可导，梯度可回传，仅适用于某些连续分布如 Normal）
- `.log_prob(value)`: 计算给定值的对数概率密度（用于计算 Loss，如 RL 中的 Policy Gradient）
- `.entropy()`: 计算分布的熵

## 基础分布
基础分布通常指统计学中最常见的分布，PyTorch 既提供了专用的快速函数，也支持通过 `distributions` 模块构建。

###  均匀分布

| 特性维度 | 快捷函数模式 (`torch.rand`) | 分布对象模式 (`dist.Uniform`) |
| :--- | :--- | :--- |
| 关键参数 | 无分布参数<br>仅接受 `*size` (形状)<br>*(隐含固定为  $ U(0, 1) $ )* | 显式分布参数<br>`low`: 下界 (标量或 Tensor)<br>`high`: 上界 (标量或 Tensor) |
| 本质 | 纯函数 (Function)<br>一次性生成随机数的工具<br>“| 分布类实例 (Class Object)<br>概率分布的数学对象<br> |
| 逐元素控制 | 不支持<br>所有元素严格服从相同的  $ U(0, 1) $ <br>*(需手动公式变换实现其他范围)* |  原生支持<br>`low` 和 `high` 可为 Tensor<br>可实现每个位置拥有独立的采样区间  $ [low_{ij}, high_{ij}] $  |
| 返回值 | `torch.Tensor`<br>直接返回采样得到的随机数据矩阵 | `Uniform` 对象 (初始化时)<br>需调用 `.sample()` 才返回 Tensor<br>(调用 `.rsample()` 可返回可导 Tensor) |
| 应用场景 | • 权重初始化 (Xavier/Uniform)<br>• Dropout 掩码生成<br>• 简单数据增强 (随机裁剪/翻转)<br>• 快速原型验证 | • 生成模型 (VAE, Flow) 需要计算 `log_prob`<br>• 强化学习 需要梯度回传至分布边界<br>• 复杂噪声模拟 (每个样本噪声范围不同)<br>• 需要计算熵 (Entropy) 或 KL 散度 |

```
import torch
import torch.distributions as dist

# 生成形状为 (1, 2) 的 Tensor，元素服从标准均匀分布 U(0, 1)
uniform_sample1 = torch.rand(1, 2)
print(uniform_sample1)

# low = [0.0, 1.0], high = [4.0, 5.0]
# 这意味着生成的随机数中，第1列服从 U(0, 4)，第2列服从 U(1, 5)
uniform_dist = dist.Uniform(torch.tensor([0.0, 1.0]), torch.tensor([4.0, 5.0]))
uniform_sample2 = uniform_dist.sample([3, 4])
print(uniform_sample2)
print(uniform_sample2.requires_grad) # False

# 源头参数 (low/high) 没有开启梯度追踪 (requires_grad=False)
# 即使使用了支持梯度的 .rsample() 计算图认为没有需要优化的变量，因此返回的 Tensor 依然不可导
uniform_sample3 = uniform_dist.rsample([3, 4])
print(uniform_sample3.requires_grad)  # False

# 显式设置 requires_grad=True
low_param = torch.tensor([0.0], requires_grad=True)
high_param = torch.tensor([5.0], requires_grad=True)
# 使用可导参数构建分布
uniform_dist2 = dist.Uniform(low_param, high_param)

# 1. 方法使用 .rsample() (构建了 x = low + (high-low)*epsilon 的计算图)
# 2. 参数 low_param, high_param 需要梯度
# -> 结果成功开启梯度追踪
uniform_sample4 = uniform_dist2.rsample([3, 4])
print(uniform_sample4.requires_grad)  # True
```

### 正态分布
| 特性 | `torch.randn(*size)` | `torch.normal(mean, std, ...)` | `torch.distributions.Normal(loc, scale)` |
| :--- | :--- | :--- | :--- |
| **本质** | 快捷函数 (Function) | 灵活函数 (Function) | 分布类 (Class / Object) |
| **分布参数** | **固定**：均值=0, 标准差=1 | **自定义**：支持标量或 Tensor 作为 mean/std | **自定义**：初始化时指定 loc/scale |
| **返回值** | 直接返回 `Tensor` (随机数据) | 直接返回 `Tensor` (随机数据) | 返回 `Normal` **分布对象** (需调用 `.sample()` 获取数据) |
| **逐元素控制** |  不支持 (所有元素同分布) |  **支持** (mean/std 可为 Tensor，实现每个位置不同分布) |  **支持** (loc/scale 可为 Tensor) |
| **高级统计功能** |  无 (仅生成数据) |  无 (仅生成数据) |  **丰富** (`log_prob`, `entropy`, `kl_divergence`, `cdf` 等) |
| **可导性 (梯度)** |  不可导 (无法回传至参数) |  不可导 (无法回传至 mean/std) |  **支持** (通过 `.rsample()` 实现重参数化，梯度可回传) |
| **典型场景** | 权重初始化 (如 Xavier)、简单噪声 | 需要特定 $\mu, \sigma$ 的采样、逐元素不同方差的噪声 | VAE (重参数化 trick)、强化学习 (Policy Gradient)、概率模型 Loss 计算 |

```
import torch
import torch.distributions as dist

# 创建一个正态分布对象：均值(loc)=0.0, 标准差(scale)=1.0
normal_dist = dist.Normal(loc=0.0, scale=1.0)
# 从该分布中采样指定形状的数据，每个元素都独立服从 N(0, 1)
normal_sample = normal_dist.sample((3, 3))
print(normal_sample)

# 直接生成一个 3x3 的张量，数据服从标准正态分布 N(0, 1)
normal_data = torch.randn((3 ,3))
print(normal_data)

# 每个位置的均值和标准差都不同
# mean: [1.0, 2.0, ..., 9.0] (共10个元素)
# std:  [1.0, 0.9, ..., 0.1] (共10个元素，步长-0.1)
# 结果：第 i 个元素服从 N(mean[i], std[i])
normal_data2 = torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
print(normal_data2)

# 所有元素均值相同 (0.5)，但每个位置的标准差不同
# mean: 标量 0.5 (会自动广播)
# std:  [1.0, 2.0, 3.0, 4.0, 5.0] (5个元素)
# 结果：生成一个长度为 5 的张量，第 i 个元素服从 N(0.5, std[i])
normal_data3 = torch.normal(mean=0.5, std=torch.arange(1., 6.))
print(normal_data3)

# 所有元素都服从同一个正态分布 N(2, 3)
# 这是最接近 torch.randn 的用法，只是允许自定义 mean 和 std
normal_data4 = torch.normal(2, 3, size=(1, 4))
print(normal_data4)
```
### 伯努利分布
| 特性 | `torch.bernoulli(input)` | `torch.distributions.Bernoulli(probs)` |
| :--- | :--- | :--- |
| **本质** | 快捷函数 (Function) | 分布类 (Class / Object) |
| **分布参数** | **输入即概率**：`input` Tensor 中的每个元素代表该位置采样为 1 的概率 ($p$) | **自定义**：初始化时指定 `probs` (概率) 或 `logits` (对数几率) |
| **返回值** | 直接返回 `Tensor` (0 或 1 的随机数据) | 返回 `Bernoulli` **分布对象** (需调用 `.sample()` 获取数据) |
| **逐元素控制** | **支持** (输入 Tensor 的每个元素可不同，实现逐元素不同概率) | **支持** (`probs` 可为 Tensor，实现每个位置不同分布) |
| **高级统计功能** | 无 (仅生成数据) | **丰富** (`log_prob`, `entropy`, `kl_divergence`, `cdf` 等) |
| **可导性 (梯度)** | **不可导** (采样结果是离散的 0/1，无法直接回传梯度至 `input` 概率) | **部分支持**：<br>1. `.sample()`: 不可导 (离散采样断点)<br>2. `.rsample()`: **不支持** (伯努利分布是离散分布，数学上无法直接重参数化)<br>*(注：若需梯度，通常需配合 Gumbel-Softmax 或 Straight-Through Estimator)* |
| **典型场景** | Dropout 层实现、二值化神经网络掩码、简单的随机丢弃策略 | 强化学习中的离散策略 (Policy Gradient)、变分推断中的离散潜变量、计算概率 Loss |

```
import torch
import torch.distributions as dist

# 定义伯努利分布，概率参数 probs 是一个 Tensor
#   - 第1个位置采样为1的概率是 0.3
#   - 第2个位置采样为1的概率是 0.4
#   - 第3个位置采样为1的概率是 0.7

bernoulli_dist = dist.Bernoulli(torch.tensor([0.3, 0.4, 0.7]))

# 采样形状 [2, 3]
# 广播机制：定义的分布形状是 [3]，采样形状是 [2, 3]
# 结果将是一个 2x3 的矩阵，每一行都遵循上述三个不同的概率分布
bernoulli_sample = bernoulli_dist.sample([2,3])
print(bernoulli_sample)

# 创建一个 3x3 的 Tensor，填充 [0, 1) 之间的均匀分布随机数
# 这里的每个元素将被视为“该位置采样为1的概率”
# 如果 a[i,j] = 0.8，则该位置有 80% 几率变为 1，20% 几率变为 0
# 如果 a[i,j] = 0.1，则该位置有 10% 几率变为 1，90% 几率变为 0
a = torch.empty(3, 3).uniform_(0, 1)
print(a)
bernoulli_sample2 = torch.bernoulli(a)
print(bernoulli_sample2)

# 创建一个全 1 的 Tensor，意味着每个位置采样为 1 的概率是 100%
a = torch.ones(2, 2)
bernoulli_sample3 = torch.bernoulli(a)
print(bernoulli_sample3)

# 创建一个全 0 的 Tensor，意味着每个位置采样为 1 的概率是 0%
a = torch.zeros(2, 2)
bernoulli_sample4 = torch.bernoulli(a)
print(bernoulli_sample4)
```

### 整数随机
`torch.randint` 返回一个张量，其中填充了从低值（包括）到高值（不包括）之间均匀生成的随机整数。
```
torch.randint(low=0, high, size, \*, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```
```
import torch

# 指定范围 [low, high)
# 生成整数，范围包含 low，但不包含 high (即 [3, 5))
t1 = torch.randint(3, 5, (3,))
print(t1)

# 指定上限 [0, high)
# 当只传一个整数参数时，默认为 high，low 默认为 0
# 范围: [0, 10) -> 可能的整数值: 0, 1, ..., 9
t2 = torch.randint(10, (2, 2))
print(t2)
```

## 高级分布
对于复杂的建模需求（如多分类策略、混合高斯模型），需要使用 `torch.distributions` 模块中的高级类。

### 分类分布
```
class torch.distributions.categorical.Categorical(probs=None, logits=None, validate_args=None)[sour
```
- `probs`：概率向量，和为 1。例如 `[0.1, 0.3, 0.6]`
- `logits`：未归一化的对数概率（Log-probabilities）。PyTorch 内部会自动通过 `softmax` 将其转换为概率。在深度学习中更常用 `logits`，因为它数值更稳定且直接来自神经网络的输出层

```
import torch
from torch.distributions import Categorical

# 定义一个3分类的分布 (K=3)
# 类别0概率10%, 类别1概率30%, 类别2概率60%
probs = torch.tensor([0.1, 0.3, 0.6])
dist = Categorical(probs)

# 采样 1 次 -> 返回一个标量 (Tensor(0-dim))
sample = dist.sample()
print(sample)

# 采样 5 次 -> 返回形状 (5,)
samples = dist.sample((5,))



# 定义一个 Batch 大小为 2, 类别数 K=3 的分布
# 第 0 个分布: [0.8, 0.1, 0.1] (极度偏向类别 0)
# 第 1 个分布: [0.1, 0.1, 0.8] (极度偏向类别 2)
probs_batch = torch.tensor([
    [0.8, 0.1, 0.1],
    [0.1, 0.1, 0.8]
])
dist_batch = Categorical(probs=probs_batch)

print(f"批次形状 (Batch Shape): {dist_batch.batch_shape}") # torch.Size([2])
print(f"事件形状 (Event Shape): {dist_batch.event_shape}") # torch.Size([]) 因为采样结果是标量索引

# 采样 (3, 4) 次 (对 batch 中的每个分布各采 3x4 个样)
# 结果形状: (3, 4) + (2,) = (3, 4, 2)
samples_many = dist_batch.sample((3, 4))
print(samples_many.shape) # torch.Size([3, 4, 2])
```

## 可复现性控制
在科研和调试中，保证随机结果的可复现性至关重要。PyTorch 的随机性来源较多，需全面控制。

全局种子设置：
```
import torch
import random
import numpy as np

seed = 42
random.seed(seed)       # Python 原生随机
np.random.seed(seed)    # NumPy 随机
torch.manual_seed(seed) # CPU GPU 随机
torch.cuda.manual_seed_all(seed) # 多卡 GPU 随机
```

# 序列与范围生成

## 整数步长生成
`torch.arange` 似于 Python 原生的 `range()` 函数，但返回的是 Tensor。适用于生成索引、时间步等离散整数序列。
```
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```
  | 特性 | 说明 |
| :--- | :--- |
| **区间** | 左闭右开 $[start, end)$ ，不包含 `end` |
| **步长** | 通过 `step` 指定固定增量（可以是负数） |
| **数据类型** | 若未指定 `dtype`，默认根据输入推断（通常为 `int64`） |
| **精度风险** | 浮点数步长可能导致累积误差，建议用于整数或显式指定 `dtype` |

```
import torch

# 创建一个从 1 开始，到 2.5 结束（不包含 2.5），步长为 0.5 的张量
t1 = torch.arange(1, 2.5, 0.5)
print(t1) # 输出: tensor([1.0000, 1.5000, 2.0000])

# 修正后的代码：创建一个从 3 开始，到 1 结束（不包含 1），步长为 -1 的张量
t2 = torch.arange(3, 1, -1, dtype=torch.int32)
print(t2) # tensor([3, 2], dtype=torch.int32)

# 创建一个从 0 开始（默认 start=0），到 5 结束（不包含 5），步长为 1（默认 step=1）的张量
t3 = torch.arange(5, dtype=torch.int32)
print(t3) # tensor([0, 1, 2, 3, 4], dtype=torch.int32)
```

## 定点数生成
当你需要精确控制采样点的数量，并确保包含起始点和结束点时，使用此类函数。常用于生成坐标轴、插值点、学习率调度曲线。

线性空间:
```
torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```

对数空间:
```
torch.logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```
| 特性 | `torch.linspace` | `torch.logspace` |
| :--- | :--- | :--- |
| **核心逻辑** | 固定点数 (Count-based) | 固定点数 (对数尺度) |
| **区间端点** | 包含 end $[start, end]$ | 包含两端 $[base^{start}, base^{end}]$ |
| **控制参数** | `start`, `end`, `steps` | `start`, `end`, `steps`, `base` |
| **典型用途** | 坐标轴、插值、线性衰减 | 学习率搜索、频率分析、超参数扫描 |
| **数据类型** | `float` | `float` |
| **数学公式** | $x_i = start + i \times \frac{end-start}{N-1}$ | $x_i = base^{start + i \times \frac{end-start}{N-1}}$ |

```
import torch


# 参数顺序: (start, end, steps)
# 含义: 在闭区间 [3, 10] 内，均匀生成 5 个点
#   步长 = (10 - 3) / (5 - 1) = 7 / 4 = 1.75
#   序列: 3.0, 4.75, 6.5, 8.25, 10.0
t1 = torch.linspace(3, 10, 5)
print(t1) # tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])

# 在闭区间 [-10, 10] 内，均匀生成 5 个点
t2 = torch.linspace(-10, 10, steps=5)
print(t2)  # tensor([-10.,  -5.,   0.,   5.,  10.])

# 参数: start=-10, end=10, steps=1, 生成 1 个点
t3 = torch.linspace(start=-10, end=10, steps=1)
print(t3)  # tensor([-10.])


# 标准对数空间生成 (底数默认为 10)
# 在 10^(-10) 到 10^(10) 之间，均匀生成 5 个点
t1 = torch.logspace(start=-10, end=10, steps=5)
print(t1) # tensor([1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10])

# 小范围对数空间 (底数默认为 10)
t2 = torch.logspace(start=0.1, end=1.0, steps=5)
print(t2)  # tensor([ 1.2589,  2.1135,  3.5481,  5.9566, 10.0000])

# 自定义底数与单点生成
# 由于 start == end 且 steps == 1，结果只有一个值：2^2 = 4
t3 = torch.logspace(start=2, end=2, steps=1, base=2)
print(t3)  # tensor([4.])
```

# 基于模板创建
基于模板创建 tensor 指的是利用一个已存在的张量（Tensor）作为参考（模板），来创建一个新的张量。

新创建的张量会继承模板张量的某些属性（如设备 `device`、数据类型 `dtype`、甚至形状 `shape`），但具体的数值或内容可以由你重新定义。


| 函数 | 含义 | 继承的属性 |
| :--- | :--- | :--- |
| `torch.zeros_like(input)` | 创建全 0 张量 | `shape`, `dtype`, `device`, `layout` |
| `torch.ones_like(input)` | 创建全 1 张量 | 同上 |
| `torch.empty_like(input)` | 创建未初始化张量 (最快) | 同上 |
| `torch.full_like(input, val)` | 创建填充指定值的张量 | 同上 |
| `torch.randn_like(input)` | 创建标准正态分布随机数 | 同上 |
| `input.clone()` | 完整克隆 (复制数据) | 同上 + 数据内容 |

`torch.clone()` 执行的是深拷贝（Deep Copy）。它会分配一块全新的内存空间来存储数据，并将原张量的数据复制过去。因此，原张量和克隆后的张量在内存中是完全独立的，修改其中一个不会影响另一个，它们的内存地址（数据指针）也不同。

```
import torch

# torch.zeros_like: 创建一个与输入张量 a 形状、类型、设备相同的全 0 张量
a = torch.empty(2, 3)
t = torch.zeros_like(a)
print(t)

# torch.ones_like: 创建一个与输入张量 a 形状、类型、设备相同的全 1 张量
a = torch.tensor([[1, 2], [3, 4]])
t = torch.ones_like(a)
print(t.dtype)  # torch.int64
print(t.shape)  # torch.Size([2, 2])

# torch.full_like: 创建一个与输入张量 a 属性相同，但填充指定值 val 的张量
a = torch.randn(2, 3)
print(a.shape) # torch.Size([2, 3])
t = torch.full_like(a, 2.0)
print(t)

# torch.randn_like: 创建一个与输入张量 a 属性相同，但元素服从标准正态分布的随机张量
a = torch.randn(3, 4)
t = torch.randn_like(a)
print(a)
print(t)

# input.clone(): 完整克隆张量，包括数据、梯度信息、requires_grad 标志等
a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True, device="cpu")
t = a.clone()
print(t)  # tensor(1., dtype=torch.float64, grad_fn=<CloneBackward0>)
print(t.device)  # cpu

#
ptr_a = a.data_ptr()
ptr_t = t.data_ptr()
print(f"原始张量 a 的内存地址: {ptr_a}")
print(f"克隆张量 t 的内存地址: {ptr_t}")
```

# 特殊结构创建

## 单位矩阵
单位矩阵指的是主对角线元素为 1，其余元素为 0 的矩阵，常用于矩阵乘法中的恒等变换。
```
torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
```
- `n, m`: 行数和列数，支持非方阵，此时只有 `min(n .m)` 个对角元素

```
import torch

# 创建 3x3 单位矩阵
I = torch.eye(3)
print(I)

# 创建 2x3 非方阵 (对角线为 1)
I_rect = torch.eye(2, 3)
print(I_rect)

# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
# tensor([[1., 0., 0.],
#         [0., 1., 0.]])
```

## 对角矩阵
对角矩阵指的是只有主对角线（或偏移对角线）上有非零值，其余全为 0 的矩阵。常用于表示缩放操作或特征值的对角化。
```
torch.diag(input, diagonal=0, *, out=None) → Tensor
```
- `input`:
  - 输入是向量 (1D) → 输出是矩阵 (2D)：把向量放在对角线上，其余补 0
  - 输入是矩阵 (2D) → 输出是向量 (1D)：提取矩阵对角线上的元素
- `diagonal`，决定了操作的目标对角线位置：
  - `==0`：主对角线
  - `>0`：主对角线上方
  - `<0`：主对角线下方

向量 → 矩阵 (构造):
```
import torch

# 主对角线 (diagonal=0, 默认)
# 结果是一个 3x3 矩阵
v = torch.tensor([1, 2, 3])
d0 = torch.diag(v)
print(d0)

# tensor([[1, 0, 0],
#         [0, 2, 0],
#         [0, 0, 3]])

# 上方对角线 (diagonal=1)
# 向量被放在主对角线右上方
# 为了容纳这条线，矩阵变大为 4x4 (原长度 3 + 偏移量 1)
d1 = torch.diag(v, diagonal=1)
print(d1)

# tensor([[0, 1, 0, 0],
#         [0, 0, 2, 0],
#         [0, 0, 0, 3],
#         [0, 0, 0, 0]])

# 下方对角线 (diagonal=-1)
# 向量被放在主对角线左下方
# tensor([[0, 0, 1, 0, 0],
#         [0, 0, 0, 2, 0],
#         [0, 0, 0, 0, 3],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]])
# 矩阵同样变大为 4x4
d2 = torch.diag(v, diagonal=-1)
print(d2)

# tensor([[0, 0, 0, 0],
#         [1, 0, 0, 0],
#         [0, 2, 0, 0],
#         [0, 0, 3, 0]])


# 上方对角线 (diagonal=1)
# 向量被放在主对角线右上方
# 为了容纳这条线，矩阵变大为 5x5 (原长度 3 + 偏移量 2)
d3 = torch.diag(v, diagonal=2)
print(d3)

# tensor([[0, 0, 1, 0, 0],
#         [0, 0, 0, 2, 0],
#         [0, 0, 0, 0, 3],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]])
```

矩阵 → 向量 (提取)：
```
import torch

m = torch.arange(16).reshape(4, 4).float()
print(m)

# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.]])

# 提取主对角线
d0 = torch.diag(m, diagonal=0)
print(d0)

# tensor([ 0.,  5., 10., 15.])

# 提取上方第一条对角线
d1 = torch.diag(m, diagonal=1)
print(d1)

# tensor([ 1.,  6., 11.])

# 提取下方第一条对角线
d2 = torch.diag(m, diagonal=-1)
print(d2)

# tensor([ 4.,  9., 14.])

# 提取上方第二条对角线
d3 = torch.diag(m, diagonal=2)
print(d3)

# tensor([2., 7.])
```

# 稀疏张量
稀疏张量指的是绝大多数元素为 0 的张量。为了节省内存和计算资源，只存储非零元素的值及其索引坐标。

COO 格式(Coordinate Format，坐标格式) 是 PyTorch 中最基础、最通用的稀疏格式，也是其他格式转换的基础。
```
torch.sparse_coo_tensor(indices, values, size=None, *, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None, is_coalesced=None) → Tensor
```
- 显式存储所有非零元素的坐标 (indices) 和 值 (values)
- `indices`: 形状为 `[d, nnz]` 的张量，`d` 是维度，`nnz` 是非零元素个数
- `values`: 形状为 `[nnz]` 的张量

```
import torch

# 创建一个 3x3 的 COO 稀疏矩阵
indices = torch.tensor([[0, 1, 2], [0, 1, 2]]) # 对角线坐标
values = torch.tensor([1., 2., 3.])
sparse_coo = torch.sparse_coo_tensor(indices, values, (3, 3))
print(sparse_coo)

# tensor(indices=tensor([[0, 1, 2],
#                        [0, 1, 2]]),
#        values=tensor([1., 2., 3.]),
#        size=(3, 3), nnz=3, layout=torch.sparse_coo)
```

`torch.Tensor.to_sparse()` 和 `torch.Tensor.to_dense()` 是 PyTorch 中用于在稠密张量 (Dense Tensor) 和 稀疏张量 (Sparse Tensor) 之间进行转换的核心方法：

```
Tensor.to_sparse(sparseDims) → Tensor

Tensor.to_dense(dtype=None, *, masked_grad=True) → Tensor
```
- `torch.Tensor.to_dense()`: 将任意布局的稀疏张量转换为标准的稠密张量
- `torch.Tensor.to_sparse()`: 将稠密张量转换为稀疏张量，默认为 `sparse_coo` 布局的稀疏张量

```
import torch

# 创建一个稠密张量 (Dense Tensor)
# 大部分元素为 0，只有两个非零元素 (9 和 10)
d = torch.tensor([[0, 0, 0], [9, 0, 10], [0, 0, 0]])

# 转换为稀疏张量 (Sparse Tensor)
# 默认转换为 COO (Coordinate Format) 格式
# 它只存储非零元素的坐标 (indices) 和值 (values)，忽略所有的 0
sparse = d.to_sparse()
print(sparse)

# tensor(indices=tensor([[1, 1],
#                        [0, 2]]),
#        values=tensor([ 9, 10]),
#        size=(3, 3), nnz=2, layout=torch.sparse_coo)

# 从稀疏张量还原为稠密张量
# .to_dense() 会根据 indices 和 values 重建完整的矩阵，缺失位置自动补 0
dense = sparse.to_dense()
print(dense)

# tensor([[ 0,  0,  0],
#         [ 9,  0, 10],
#         [ 0,  0,  0]])
```