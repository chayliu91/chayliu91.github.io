---
title: "torchvision"
date: 2026-03-24
draft: false
categories: ["Pytorch", "数据集和预处理"]
tags: ["Pytorch"]
weight: 3
---


# torchvision.datasets

提供即插即用的数据集类。常见的数据集有：
```
import torchvision.datasets as datasets
import inspect

# 定义需要排除的“非具体数据集”类名
exclude_names = {
    'VisionDataset',   # 基类
    'DatasetFolder',   # 通用文件夹加载器
    'ImageFolder',     # 通用图片文件夹加载器
    'VideoFolder',     # 通用视频文件夹加载器 (如果有)
    'FakeData',        # 随机数据生成器 (虽能跑，但不是真实数据集)
    'Samplers'         # 有时会被误抓到的其他类
}

# 筛选：是类 + 大写开头 + 不在排除名单中
dataset_classes = [
    name for name, obj in inspect.getmembers(datasets)
    if inspect.isclass(obj)
    and name[0].isupper()
    and name not in exclude_names
]

print("真正的具体数据集列表 (共 {} 个):".format(len(dataset_classes)))
for ds in sorted(dataset_classes):
    print(f"- {ds}")
```
## 数据集简介
### 图像分类数据集

| 数据集 | 简介 | 特点/用途 |
| :--- | :--- | :--- |
| **MNIST** | 手写数字 (0-9) | **入门必用**，28x28灰度图，6万训练+1万测试。 |
| **FashionMNIST** | 时尚商品 (衣服/鞋等) | MNIST的替代品，难度稍高，格式完全一致。 |
| **CIFAR10** | 10类自然图像 (飞机/猫/狗等) | **基准测试**，32x32彩色图，适合验证模型架构。 |
| **CIFAR100** | 100类细粒度自然图像 | CIFAR10的升级版，类别更细，难度更大。 |
| **ImageNet** | 1000类大规模自然图像 | **工业级标准**，高分辨率，需手动下载整理，用于预训练模型。 |
| **Caltech101/256** | 物体识别 | 早期经典数据集，背景较复杂。 |
| **STL10** | 类似CIFAR但分辨率更高 | 96x96像素，包含未标记数据，常用于**半监督/自监督学习**。 |
| **SVHN** | 街道门牌号数字 | 真实场景数字，比MNIST复杂，RGB图像。 |
| **Food101** | 101种食物 | 美食分类，每类1000张图，挑战光照和角度变化。 |
| **Flowers102** | 102种花卉 | 细粒度分类，花朵之间差异微小。 |
| **FGVCAircraft** | 飞机型号 | 极细粒度分类，区分不同型号的飞机。 |
| **StanfordCars** | 汽车型号 | 细粒度分类，区分不同年份和型号的汽车。 |
| **Country211** | 211个国家的地标/风景 | 地理定位相关分类。 |
| **EuroSAT** | 卫星图像土地利用分类 | 遥感领域，分类土地覆盖类型（森林、居民区等）。 |
| **GTSRB** | 德国交通标志 | 自动驾驶相关，识别各种交通路牌。 |
| **SUN397** | 397种场景 | 场景识别（如厨房、海滩、办公室）。 |
| **Places365** | 大规模场景识别 | 专注于环境场景而非物体。 |
| **DTD** | 描述性纹理数据集 | 纹理分类（如编织、斑驳、点状）。 |
| **Omniglot** | 多语言字符 | "小 ImageNet"，包含多种语言的字符，用于少样本学习。 |
| **EMNIST / QMNIST** | MNIST的扩展版 | 包含字母、数字，或修正了MNIST的错误标签。 |
| **KMNIST** | 日文草书数字 | 替代MNIST，难度更高。 |
| **SEMEION** | 手写数字 | 较小众的手写数字集。 |
| **USPS** | 美国邮政手写数字 | 16x16小图，较早的数字集。 |
| **PCAM** | 病理切片分类 | 医疗影像，判断淋巴结切片是否有癌转移。 |
| **Imagenette** | ImageNet的子集 | 精选的10类易区分ImageNet数据，用于快速实验。 |
| **RenderedSST2** | 渲染的文字图像情感分析 | 将文本情感分类转化为图像分类任务。 |

### 目标检测与分割

| 数据集 | 简介 | 特点/用途 |
| :--- | :--- | :--- |
| **CocoDetection** | COCO 目标检测 | **行业标准**，80类物体，包含复杂的遮挡和小物体。 |
| **CocoCaptions** | COCO 图像描述 | 每张图有5句人工描述，用于**图像字幕生成 (Image Captioning)**。 |
| **VOCDetection** | PASCAL VOC 检测 | 早期经典检测数据集，20类物体。 |
| **VOCSegmentation** | PASCAL VOC 分割 | 提供像素级语义分割掩码。 |
| **Cityscapes** | 城市街道场景分割 | **自动驾驶核心**，高分辨率街景，精细的道路/车辆/行人分割。 |
| **SBDataset** | SBD (Semantic Boundaries) | 基于VOC的增强版分割数据，边界更精细。 |
| **INaturalist** | 生物物种识别与检测 | 超大规模动植物数据，细粒度极高。 |
| **OxfordIIITPet** | 宠物品种与分割 | 37种宠物，提供前景分割掩码。 |
| **WIDERFace** | 人脸检测 | 极大规模人脸检测，包含极度拥挤和模糊场景。 |
| **CelebA** | 名人面部属性 | 20万张名人脸，主要用于**人脸属性分析** (如戴眼镜吗？微笑吗？) 和生成。 |
| **LFWPeople / LFWPairs** | 人脸验证 | 用于判断两张脸是否属于同一个人。 |
| **Flickr30k / Flickr8k** | 图像描述 | 类似COCO Captions，用于图文匹配研究。 |

### 视频动作识别
| 数据集 | 简介 | 特点/用途 |
| :--- | :--- | :--- |
| **UCF101** | 101类人类动作 | 经典视频动作数据集，包含体育、日常动作等。 |
| **HMDB51** | 51类人类动作 | 来源广泛（电影、网页），动作更自然。 |
| **Kinetics** | 大规模动作识别 | Google发布，数百类动作，数据量巨大，需自行下载整理。 |

### 立体视觉与光流

| 数据集 | 简介 | 特点/用途 |
| :--- | :--- | :--- |
| **CREStereo / CarlaStereo / ETH3DStereo** | 立体匹配数据集 | 提供左右目图像及视差图 (Disparity Map)，用于训练深度估计模型。 |
| **Kitti / Kitti2012/2015Stereo** | KITTI 自动驾驶立体数据 | 真实道路场景的立体视觉基准。 |
| **Middlebury2014Stereo** | 高精度室内立体匹配 | 实验室环境下的高精度真值。 |
| **FlyingChairs / FlyingThings3D** | 合成光流数据 | 通过3D引擎生成的合成数据，用于预训练光流估计模型。 |
| **Sintel / SintelStereo** | 动画电影光流/立体数据 | 开源动画电影生成的长序列光流/深度数据。 |
| **HD1K** | 高分辨率光流 | 城市驾驶场景的高清光流。 |
| **InStereo2k** | 室内立体匹配 | 大规模室内场景立体数据。 |
| **SceneFlowStereo** | 合成场景流 | 包含物体运动和相机运动的合成数据。 |
| **FallingThingsStereo** | 掉落物体立体数据 | 合成数据，模拟物体掉落过程的深度信息。 |

### 其他

| 类名 | 用途 |
| :--- | :--- |
| **FakeData** | 生成随机噪声数据。用于**调试代码逻辑**（无需下载真实数据，跑通流程即可）。 |
| **PhotoTour** | 局部补丁匹配数据集 (Liberty, NotreDame等)。 |
| **SBU** | 阴影检测数据集 (Shadow Boundaries)。 |
| **CLEVRClassification** | CLEVR 场景的分类任务变体。 |
| **MovingMNIST** | 移动的MNIST数字序列，用于视频预测任务。 |
| **LSUN / LSUNClass** | 大规模场景理解，常用于生成模型 (GAN) 训练。 |


## 加载数据集
```
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# 定义数据预处理 (Transform)
# 通常需要将图片转换为 Tensor 并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 示例：灰度图归一化
])

# 下载并加载训练集
# root: 数据存放根目录
# train: True表示训练集，False表示测试集
# download: True 表示如果本地没有则自动下载
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)
```

# torchvision.models
`torchvision.models` 提供经典的深度学习架构及其预训练权重。

```
from torchvision import models
# 加载带预训练权重的 ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
```
通常模型保存的目录为：`~/.cache/torch/hub/checkpoints/`。

也可以手动下载，本地加载：
```
from torchvision import models

# 查看 ResNet50 所有可用的预训练权重版本
print(list(models.ResNet50_Weights))

# 这里选择了 'IMAGENET1K_V2'，这是 ResNet50 在 ImageNet-1K 数据集上训练的较新版本
weights_enum = models.ResNet50_Weights.IMAGENET1K_V2

# 访问枚举对象的属性，获取该权重文件的远程下载 URL。
print("下载链接:", weights_enum.url)
# 输出示例: https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```
根据 URL 下载模型到本地。

```
import torch
from torchvision import models

# 初始化模型结构 (不加载预训练权重)
model = models.resnet50(weights=None)

# 指定本地权重路径
local_weight_path = "./models/resnet50-11ad3fa6.pth"

# 加载权重
# map_location 用于指定加载到 CPU 还是 GPU
state_dict = torch.load(local_weight_path, map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)

print(f"成功从 {local_weight_path} 加载了权重")
```

# torchvision.transforms
`torchvision.transforms` 是 PyTorch 中用于图像预处理和数据增强的核心模块。它的主要作用是将原始图像（如 PIL Image 或 numpy 数组）转换为模型可以处理的 Tensor 格式，并在训练过程中通过随机变换扩充数据，提高模型的泛化能力。

## Compose (管道)

大多数情况下，你需要将多个变换操作串联起来。transforms.Compose 就是用来做这个的，它像一个管道，图像会依次通过列表中的每一个变换:
```
from torchvision import transforms

# 定义一个变换管道
my_transforms = transforms.Compose([
    transforms.Resize(256),          # 第一步：调整大小
    transforms.CenterCrop(224),      # 第二步：中心裁剪
    transforms.ToTensor(),           # 第三步：转为 Tensor
    transforms.Normalize(...)        # 第四步：标准化
])
```

## 支持的转换类型

### 基础转换
| 变换类 | 功能描述 | 输入 ➔ 输出 | 备注 |
| :--- | :--- | :--- | :--- |
| **`ToTensor`** | 转为 Tensor 并归一化 | PIL/NDArray `[H,W,C]`<br>➔ Tensor `[C,H,W]` | 像素值自动从 `0-255` 缩放到 `0.0-1.0`。**几乎所有管道第一步**。 |
| **`ToPILImage`** | 转为 PIL 图片 | Tensor `[C,H,W]`<br>➔ PIL Image `[H,W,C]` | 用于可视化或在仅支持 PIL 的变换前转换。 |
| **`ConvertImageDtype`** | 转换图像数据类型 | Tensor ➔ Tensor | 将 float/int 类型转换为指定 dtype (如 `torch.float32`) 并缩放数值。 |
| **`Lambda`** | 自定义函数 | 任意 ➔ 任意 | 包裹一个自定义 Python 函数，用于执行未内置的操作。 |


### 几何尺寸与裁剪

| 变换类 | 功能描述 | 关键参数 | 适用阶段 |
| :--- | :--- | :--- | :--- |
| **`Resize`** | 调整图像大小 | `size`: 目标尺寸 | 通用 |
| **`CenterCrop`** | 中心裁剪 | `size`: 裁剪尺寸 | **验证/测试首选** |
| **`RandomCrop`** | 随机裁剪 | `size`, `padding` |  **训练常用** |
| **`RandomResizedCrop`** | 随机缩放后裁剪 | `scale`, `ratio` |  **训练最强增强** (模拟物体远近) |
| **`FiveCrop`** | 切出四角+中心 | `size` | 特殊评估 (需配合 `TenCrop` 使用) |
| **`TenCrop`** | 五裁 + 水平翻转 | `size` | 高精度评估 (推理时常用) |
| **`Pad`** | 填充边框 | `padding`, `fill` | 通用 (保持尺寸或对齐) |
| **`RandomPad`** | 随机填充 | `padding` |  训练增强 |

### 随机几何增强

| 变换类 | 功能描述 | 关键参数 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **`RandomHorizontalFlip`** | 随机水平翻转 | `p=0.5` |  **最常用** (适合大多数物体) |
| **`RandomVerticalFlip`** | 随机垂直翻转 | `p=0.5` |  适合上下对称不敏感的任务 (如卫星图) |
| **`RandomRotation`** | 随机旋转 | `degrees` |  适合方向不敏感任务 (如植物、纹理) |
| **`RandomAffine`** | 随机仿射变换 | `degrees`, `translate`, `scale`, `shear` |  **高级增强** (旋转+平移+缩放+剪切组合) |
| **`RandomPerspective`** | 随机透视变换 | `distortion_scale`, `p` |  模拟不同拍摄视角 |
| **`ElasticTransform`** | 弹性形变 | `alpha`, `sigma` |  医学图像或文本增强常用 |


### 颜色与光度增强

| 变换类 | 功能描述 | 关键参数 | 作用 |
| :--- | :--- | :--- | :--- |
| **`ColorJitter`** | 颜色抖动 | `brightness`, `contrast`, `saturation`, `hue` | 随机调整亮度、对比度、饱和度、色调。 |
| **`RandomGrayscale`** | 随机灰度化 | `p=0.1` | 强迫模型学习形状特征而非颜色。 |
| **`GaussianBlur`** | 高斯模糊 | `kernel_size`, `sigma` | 模拟失焦或低质量图像。 |
| **`RandomInvert`** | 随机反色 | `p=0.5` | 适合特定场景 (如底片、医学影像)。 |
| **`Posterize`** | 色彩量化 | `bits` | 减少颜色位数，模拟复古风格。 |
| **`Solarize`** | 曝光反转 | `threshold` | 高于阈值的像素反色。 |
| **`AdjustSharpness`** | 调整锐度 | `sharpness_factor` | 模拟清晰或模糊。 |
| **`AdjustContrast`** | 调整对比度 | `contrast_factor` | 确定性调整 (也可随机)。 |
| **`AdjustBrightness`** | 调整亮度 | `brightness_factor` | 确定性调整。 |
| **`AdjustSaturation`** | 调整饱和度 | `saturation_factor` | 确定性调整。 |
| **`AdjustHue`** | 调整色相 | `hue_factor` | 确定性调整。 |
| **`RandomEqualize`** | 随机直方图均衡化 | `p=0.5` | 增强对比度分布。 |

### 高级与混合增
| 变换类 | 功能描述 | 特点 |
| :--- | :--- | :--- |
| **`AutoAugment`** | 自动增强策略 | 使用在 ImageNet 上搜索出的最佳策略组合。 |
| **`RandAugment`** | 随机增强 | 简化版 AutoAugment，无需搜索，直接随机应用。 |
| **`TrivialAugmentWide`** | 简单宽幅增强 | 单操作增强，效果极佳且速度快 (ResNet/ViT 新标配)。 |
| **`AugMix`** | 混合增强 | 混合多种增强操作，提高鲁棒性和不确定性估计。 |
| **`Cutout`** | 随机遮挡 | 随机抹去图像中的一部分矩形区域 (需手动实现或使用 `RandomErasing`)。 |
| **`RandomErasing`** | 随机擦除 |  随机用像素值填充矩形区域，模拟遮挡。 |
| **`MixUp` / `CutMix`** | 图像混合 | *注：通常在 DataLoader 的 collate_fn 中实现，不在 transforms 直接调用，但属于同一类思想。* |


### 标准化
必须放在 `ToTensor` 之后，将数据分布转换为标准正态分布。
| 变换类 | 功能描述 | 关键参数 | 备注 |
| :--- | :--- | :--- | :--- |
| **`Normalize`** | 标准化 | `mean`, `std` | 公式：`output = (input - mean) / std`。<br>ImageNet 常用：<br>`mean=[0.485, 0.456, 0.406]`<br>`std=[0.229, 0.224, 0.225]` |

## 自定义 transform

### 基于类的定义
这是最标准的方法。适用于需要保存状态（如随机种子、参数）或逻辑较复杂的变换。

核心规则：
- 定义一个类。
- 在 `__init__` 中初始化参数
- 实现 `__call__(self, img)`  方法，接收图像并返回变换后的图像

```
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        """
        :param mean: 噪声均值
        :param std: 噪声标准差
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 确保输入是 Tensor (如果是 PIL，前面的 ToTensor 应该已经处理了)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Input should be Tensor, got {type(tensor)}")

        # 生成与图像形状相同的噪声
        noise = torch.randn(tensor.size()) * self.std + self.mean

        # 添加噪声并截断到 [0, 1] 范围 (防止像素值溢出)
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0.0, 1.0)

    def __repr__(self):
        # 可选：定义打印格式，方便调试
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

# --- 使用示例 ---
transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.05), # 实例化并使用
])

# 模拟一张图片测试
dummy_img = Image.new('RGB', (224, 224), color='red')
output = transform(dummy_img)
print(f"输出类型: {type(output)}, 形状: {output.shape}")
```


### 基于函数定义
如果你的变换逻辑很简单，不需要保存任何参数，可以直接写一个函数，然后用 `transforms.Lambda` 包裹，或者直接放入 `Compose` (新版 PyTorch 支持直接放函数)。
```
import torchvision.transforms as transforms
import torch

def my_grayscale(tensor):
    """手动将 RGB Tensor 转为灰度"""
    # 简单的加权平均：Y = 0.299R + 0.587G + 0.114B
    if tensor.shape[0] == 3:
        gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        return gray.unsqueeze(0) # 恢复通道维度 [1, H, W]
    return tensor

# --- 使用方式 A: 直接放入 Compose (PyTorch >= 1.10 推荐) ---
transform_a = transforms.Compose([
    transforms.ToTensor(),
    my_grayscale,  # 直接放函数名
])

# --- 使用方式 B: 使用 Lambda 包裹 (兼容旧版本) ---
transform_b = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(my_grayscale),
])
```

### 同时处理图像和标签
在目标检测或分割任务中，变换需要同时作用于图像 (Image) 和 标签 (Target/Boxes/Mask)，并且要保持几何变换的一致性（例如：图裁剪了，框也要跟着裁剪）。
```
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class RandomCropSync:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        # 假设输入 sample 是一个元组: (image, target_dict)
        # target_dict 包含 'boxes': [N, 4]
        image, target = sample

        # 1. 随机生成裁剪参数
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.size)

        # 2. 对图像进行裁剪
        image = F.crop(image, i, j, h, w)

        # 3. 对标签 (Bounding Boxes) 进行同步裁剪
        # 这里只是简单演示逻辑，实际需处理框越界、坐标转换等复杂情况
        if 'boxes' in target:
            boxes = target['boxes']
            # 将 box 坐标从 [x_min, y_min, x_max, y_max] 转换为相对于裁剪区域的坐标
            # ... (此处省略复杂的几何计算代码，实际建议使用 v2 版本)
            pass

        return image, target

# --- 使用示例 ---
# DataLoader 的 collate_fn 之前，transform 会返回 (img, target) 元组
transform = RandomCropSync(size=(224, 224))
# output_img, output_target = transform((input_img, input_target))
```