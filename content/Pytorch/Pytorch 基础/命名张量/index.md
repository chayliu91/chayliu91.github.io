---
title: "命名张量"
date: 2026-03-13
draft: false
categories: ["Pytorch", "Pytorch 基础"]
tags: ["Pytorch"]
weight: 9
---

允许给张量的每个维度赋予名称（如 `batch, channel, height, width`），而不是仅仅依赖整数索引`（0, 1, 2, 3）`。这极大地提高了代码的可读性，并减少了因维度顺序错误（如` (N, C, H, W)` 写成了 `(N, H, C, W)）` 导致的 Bug。

# 创建命名张量
创建时指定名称:
```
weights_named = torch.tensor([0.2126, 0.7152, 0.0722],
                             names=['channels'])
print(weights_named)
# 输出: tensor([0.2126, 0.7152, 0.0722], names=('channels',))
```

# 张量改名字

| 特性 | rename | refine_names |
| :--- | :--- | :--- |
| **能改名字** | 支持修改已有名称 |  不支持修改已有名称 |
| **能补 None** | 支持将 None 命名为具体值 | 支持将 None 命名为具体值 |
| **一致性检查** |  无严格检查，直接覆盖 | 严格检查，冲突则报错 |
| **主要用途** | 灵活重命名、重置名称 | 安全地细化名称、断言维度语义 |


给每个维度一个名称:
```
import torch

# 注意：默认创建的张量没有命名维度 (names 为 (None, None))
x = torch.randn(2, 3)
print(x.names) # (None, None)

# 为所有维度批量命名
# 按顺序为张量的每个维度指定名称
# 这里将第 0 维命名为 "batch"，第 1 维命名为 "feature"
x = x.rename("batch", "feature")
print(x.names) # ('batch', 'feature')

# 通过关键字参数仅重命名指定的维度
# 这里将名为 "feature" 的维度重命名为 "channel"
# 其他维度 ("batch") 保持不变
x = x.rename(feature="channel")
print(x.names) # ('batch', 'channel')
```

`refine_names` 要求更加严格:
- 只能将 None 命名为具体字符串，不能修改已有的名称
- 提供的名称数量必须与张量的维度数 (ndim) 一致
- 如果维度已经有名字，再次调用 refine_names 赋予相同的名字是允许的 (幂等操作)
- 持使用 `...` 省略中间维度

```
import torch

x = torch.randn(2, 3, 4)
x = x.refine_names("batch", "channel", "width")
print(x.names) # ('batch', 'channel', 'width')

# 再次调用 refine_names 赋予完全相同的名称
# 这是安全的 (OK)，因为它是幂等的 (Idempotent)
x = x.refine_names("batch", "channel", "width")  # OK
print(x.names) # ('batch', 'channel', 'width')

# 如果这里尝试赋予不同的名字 (如 "B", "C", "W")，则会报错 RuntimeError
# x = x.refine_names("B", "C", "W")

x = torch.randn(2, 3, 4, 5)
# ... 代表“前面所有未指定的维度”，保持它们为 None
x = x.refine_names(..., 'rows', 'columns')

print(x.names)  # (None, None, 'rows', 'columns')
```

有些函数或操作可能还不支持命名张量，这时需要移除名称:
```
# 使用 rename(None) 移除所有名称
gray_plain = gray_named.rename(None)
print(gray_plain.names)  # 输出: (None, None)
```

# 按名字重新排列维度
| 特性 | `align_to(*names)` | `align_as(other)` |
| :--- | :--- | :--- |
| **定义** | 显式指定目标维度名称顺序 | 隐式继承另一个张量的维度顺序 |
| **参数** | 字符串列表 (如 `"A", "B"`) | 一个命名张量对象 |
| **灵活性** | 高 (可任意重排、部分指定) | 低 (完全匹配参考张量) |
| **场景** | 定义标准格式、固定重排逻辑 | 两个张量运算前的对齐、动态适配 |
| **示例** | `x.align_to("N", "C", "H", "W")` | `x.align_as(weight)` |
| **类比** | 按**图纸**摆放家具 | 照搬**隔壁房间**的布局 |

```
import torch

x = torch.randn(2, 3, 4).refine_names("batch", "channel", "width")

# 目标顺序: ("batch", "width", "channel")
# 内部逻辑: PyTorch 会自动查找名为 "batch", "width", "channel" 的维度并按此顺序排列
# 等价于传统操作: x.permute(0, 2, 1)
y = x.align_to("batch", "width", "channel")
print(y.names) # ('batch', 'width', 'channel')
print(y.shape) # torch.Size([2, 4, 3])


# "channel" 和 "width" 在 x 中存在，直接对齐
# "batch" 在 x 中不存在
# align_to 会自动在 "batch" 位置插入一个大小为 1 的新维度 (Unsqueezing)
x = torch.randn(3, 4).refine_names("channel", "width")
y = x.align_to("batch", "channel", "width")
print(y.names) # ('batch', 'channel', 'width')
print(y.shape) # torch.Size([1, 3, 4])

x = torch.randn(2, 3).refine_names("batch", "channel")
y = torch.randn(3, 2).refine_names("channel", "batch")


#  获取目标张量 y 的维度名称顺序 -> ("channel", "batch")
#  将源张量 x 的维度按照该顺序重新排列。
#  内部等价于执行: x.align_to("channel", "batch")
z = x.align_as(y)
print(z.names) # ('channel', 'batch')
```

# 按名称操作
现在可以按名称进行数学运算和聚合操作，无需记忆数字索引:
```
# 按元素相乘 (现在维度已对齐)
weighted_named = img_named * weights_aligned

# 按名称求和: 在 'channels' 维度上求和
gray_named = weighted_named.sum('channels')

print(gray_named.shape)  # 输出: torch.Size([5, 5])
print(gray_named.names)  # 输出: ('rows', 'columns')
```

# 类型安全：维度名称检查
命名张量会在编译/运行时检查维度名称的兼容性，防止常见错误：
```
# 错误示例：尝试将 [channels, rows, columns] 与 [channels] 直接相乘
# (未对齐) 会导致运行时错误
try:
    result = img_named[..., :3] * weights_named
except RuntimeError as e:
    print(e)

# 输出: Error when attempting to broadcast dims ['channels', 'rows', 'columns']
# and dims ['channels']: dim 'columns' and dim 'channels' are at the same
# position from the right but do not match.
```
这种错误提示比"尺寸不匹配"更加清晰，直接告诉你是哪个维度的名称冲突。