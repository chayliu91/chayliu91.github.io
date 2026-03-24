---
title: "torchaudio"
date: 2026-03-24
draft: false
categories: ["Pytorch", "数据集和预处理"]
tags: ["Pytorch"]
weight: 4
---

`torchaudio` 是 PyTorch 生态中专门用于音频信号处理和语音深度学习的核心库。它与 `torchvision` 类似，但针对音频数据（波形、频谱）进行了优化，支持 GPU 加速，并能无缝集成到 PyTorch 的 `DataLoader` 和训练循环中。

- `Librosa`: CPU 运行，功能极其丰富，适合数据分析、可视化和研究原型。不支持 GPU，无法直接放入 PyTorch DataLoader 的多进程加速中（容易卡顿）
- `Torchaudio`: GPU 加速，基于 Tensor，原生支持 PyTorch 生态。适合大规模训练、部署和生产环境

建议: 分析数据时用 `Librosa`，训练模型时用 `Torchaudio`。

音频处理常用标准流程:
- 加载 (Load): 读取音频文件 -> 得到波形 (Waveform) 和 采样率 (Sample Rate)
- 预处理 (Preprocess): 重采样 (Resample)、标准化、裁剪
- 特征提取 (Feature Extraction): 将时域波形转换为频域特征 (如 Mel Spectrogram, MFCC)
- 增强 (Augmentation): 添加噪声、混响、时间拉伸等
- 模型输入: 送入神经网络 (CNN, RNN, Transformer)

# torchaudio.datasets
`torchaudio.datasets` 包含了语音处理常用的数据集：

```
import torchaudio.datasets as datasets
import inspect

dataset_classes = [name for name, obj in inspect.getmembers(datasets)
                   if inspect.isclass(obj) and not name.startswith('_')]

print("torchaudio 支持的数据集列表：")
for i, name in enumerate(dataset_classes, 1):
    print(f"{i}. {name}")
```

## 语音识别与合成
| 数据集名称 | 核心用途 | 数据特点/备注 |
| :--- | :--- | :--- |
| **LIBRISPEECH** | 自动语音识别 (ASR) | **行业标准基准**，源自有声书，数据量巨大（约960小时），英文为主。 |
| **COMMONVOICE** | 多语言语音识别 (ASR) | Mozilla 开源，涵盖多种语言、口音和性别，社区贡献数据。 |
| **LJSPEECH** | 单说话人语音合成 (TTS) | 一位女性说话人的高质量录音，TTS 模型的标准训练集。 |
| **LIBRITTS** | 高质语音合成 (TTS) | LibriSpeech 的扩展版，专为合成优化，包含更丰富的韵律和说话人信息。 |
| **TEDLIUM** | 会议/演讲语音识别 | 源自 TED 演讲，包含长段落语音和对应的转录文本。 |
| **CMUARCTIC** | 语音合成 / 语音转换 | 平衡的多说话人英语数据集，常用于基线测试。 |
| **LibriLightLimited** | 低资源语音识别 | LibriLight 的子集，用于研究在极少数据下的模型表现。 |
| **YESNO** | 单元测试 / 教学演示 | 极小数据集（仅 "yes"/"no"），用于快速调试代码逻辑，不可用于训练。 |

## 关键词检测
| 数据集名称 | 核心用途 | 数据特点/备注 |
| :--- | :--- | :--- |
| **SPEECHCOMMANDS** | 关键词检测 (KWS) | Google 发布，含 "Yes", "No", "Up" 等短命令，**入门首选**。 |
| **FluentSpeechCommands** | 自然命令词检测 | 比 Google SpeechCommands 更自然、包含更多背景噪音和语气的指令数据。 |
| **Snips** | 智能助手意图识别 | 包含天气、闹钟、音乐播放等特定场景的自然语言命令。 |

##  音乐与音频分离
| 数据集名称 | 核心用途 | 数据特点/备注 |
| :--- | :--- | :--- |
| **GTZAN** | 音乐流派分类 | 经典入门数据集，含10种流派（如摇滚、爵士），每条30秒。 |
| **MUSDB_HQ** | 音乐源分离 | 高质量分轨数据（人声、鼓、贝斯、其他），音乐分离任务的黄金标准。 |
| **LibriMix** | 语音分离 | 模拟多人同时说话的场景，用于训练模型从混合音中分离出人声。 |

## 情感分析与对话
| 数据集名称 | 核心用途 | 数据特点/备注 |
| :--- | :--- | :--- |
| **IEMOCAP** | **语音情感识别 (SER)** | 包含演员表演的对话，标注了愤怒、快乐、悲伤等情绪，极具价值。 |

## 说话人识别与验证
| 数据集名称 | 核心用途 | 数据特点/备注 |
| :--- | :--- | :--- |
| **VoxCeleb1Identification** | 说话人识别 (多分类) | 名人采访片段，任务是判断音频属于哪位名人。 |
| **VoxCeleb1Verification** | 说话人验证 (二分类) | 同上，但任务是判断两段音频是否来自**同一位**名人。 |
| **VCTK_092** | 说话人识别 / 合成 | 包含109位说话人朗读相同句子的数据，口音多样。 |
| **DR_VCTK** | 说话人自适应 / 合成 | VCTK 的“去混响”版本或特定处理版本，用于高质量说话人建模。 |

## 特殊用途与研究型数据集

| 数据集名称 | 核心用途 | 数据特点/备注 |
| :--- | :--- | :--- |
| **CMUDict** | 文本预处理 / 音素对齐 | **非音频波形数据**。是卡内基梅隆大学的发音词典（单词→音素映射）。 |
| **LibriSpeechBiasing** | 上下文偏置研究 | 包含稀有词汇的样本，用于优化模型对特定关键词的识别能力。 |
| **QUESST14** | 零资源关键词搜索 | 用于研究在没有转录文本的情况下进行关键词检索。 |

# 加载与保存
支持格式: WAV, MP3, FLAC, OGG, SPHERE 等 (依赖后端 ffmpeg 或 sox)：
```
import torchaudio

# 加载音频
waveform, sample_rate = torchaudio.load("audio.wav")
# waveform 形状: [通道数, 时间步数] (例如: [1, 160000] 表示单声道，10秒 @ 16kHz)
# sample_rate: 整数 (例如 16000, 44100)

# 保存音频
torchaudio.save("output.wav", waveform, sample_rate)
```

# 变换管道

这是 `torchaudio` 最强大的部分，类似于 `torchvision.transforms`。你可以将多个变换组合成 `torch.nn.Sequential` 管道。

## 基础波形变换
| 变换类 | 功能 | 常用参数 | 场景 |
| :--- | :--- | :--- | :--- |
| `Resample` | 重采样 | `orig_freq`, `new_freq` | 统一所有音频的采样率 (如统一为 16kHz) |
| `Fade` | 淡入淡出 | `fade_in_len`, `fade_out_len` | 消除音频首尾的爆破音 |
| `Vol` | 音量调整 | `gain`, `gain_type` | 标准化音量大小 |
| `Trim` | 修剪静音 | `top_db` | 切除首尾的静音部分 |

## 频域特征提取

绝大多数语音模型不直接输入波形，而是输入频谱图。

| 变换类 | 功能 | 关键参数 | 输出形状 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| `Spectrogram` | 短时傅里叶变换 (STFT) | `n_fft`, `hop_length`, `win_length` | `[Channel, Freq, Time]` | 将波形转为线性频谱图 |
| `MelSpectrogram` | 梅尔频谱图 | `sample_rate`, `n_mels`, `f_min`, `f_max` | `[Channel, Mel, Time]` | 最常用。模拟人耳听觉特性，语音识别标配 |
| `MFCC` | 梅尔频率倒谱系数 | `n_mfcc`, `melkwargs` | `[Channel, N_MFCC, Time]` | 传统语音识别特征，压缩了频谱信息 |
| `AmplitudeToDB` | 振幅转分贝 | `stimulus`, `top_db` | 同输入 | 将对数刻度转为 dB，符合人耳感知，通常接在 `Spectrogram` 后 |


## 数据增强
用于训练阶段，提高模型鲁棒性。

| 变换类 | 功能 | 作用 |
| :--- | :--- | :--- |
| `AddNoise` (需自定义或 prototype) | 添加背景噪声 | 模拟嘈杂环境 |
| `TimeStretch` | 时间拉伸 | 改变语速但不改变音调 (或反之) |
| `PitchShift` | 变调 | 改变音调但不改变语速 |
| `SpecAugment` (通常手动实现) | 频谱遮挡 | 随机遮挡频谱图的时间轴或频率轴 (Masking)，防止过拟合 |


# Pipelines 与 预训练模型
`torchaudio` 提供了一些高级 API `torchaudio.pipelines`，可以直接加载预训练的模型进行推理，而不仅仅是特征提取。

```
# 示例：使用预训练的 Wav2Vec2 进行语音识别 (伪代码，具体 API 随版本变化)
import torchaudio

# 1. 加载预训练 Bundle
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()

# 2. 准备音频 (需重采样到模型要求的采样率)
waveform, sr = torchaudio.load("audio.wav")
waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

# 3. 推理
with torch.no_grad():
    emission, _ = model(waveform)

# 4. 解码 (CTC Decoder)
# 需要使用 bundle.get_decoder() 或类似方法将 emission 转为文本
```

# 代码模板

## 构建标准的语音识别预处理管道
```
import torch
import torchaudio
from torchaudio import transforms as T

class AudioPreprocessor(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.sample_rate = sample_rate

        # 1. 重采样 (如果输入音频采样率不固定，需在 forward 中动态处理或预先处理)
        # 这里假设输入已经是 16k，或者我们强制重采样
        self.resampler = T.Resample(orig_freq=44100, new_freq=16000)

        # 2. 提取梅尔频谱图
        self.mel_spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,          # 窗口大小 (25ms @ 16k)
            win_length=400,
            hop_length=160,     # 步长 (10ms @ 16k)
            n_mels=n_mels,      # 梅尔滤波器组数量
            f_min=0,
            f_max=8000
        )

        # 3. 转为分贝 (Log Scale)
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, waveform, orig_sr):
        # A. 重采样
        if orig_sr != self.sample_rate:
            waveform = self.resampler(waveform)

        # B. 特征提取
        spec = self.mel_spec(waveform)

        # C. 转 dB
        spec_db = self.amplitude_to_db(spec)

        return spec_db

# --- 使用示例 ---
preprocess = AudioPreprocessor()

# 加载音频
wav, sr = torchaudio.load("example.wav") # wav shape: [1, time]

# 处理
features = preprocess(wav, sr)
print(f"输入形状: {wav.shape}, 输出特征形状: {features.shape}")
# 输出: [1, 80, time_steps] -> 可以直接送入 CNN 或 Transformer
```

## 数据增强 (训练专用)
```
import torch.nn as nn
from torchaudio import transforms as T
import random

class TrainAugmentation(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        # 注意：torchaudio 内置的高级增强较少，常需结合自定义或使用 torchaudio.prototype

    def forward(self, waveform):
        # 1. 随机添加高斯噪声
        if random.random() > 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise

        # 2. 随机音量变化
        gain = random.uniform(0.8, 1.2)
        waveform = waveform * gain

        # 3. 随机裁剪 (模拟不同长度)
        length = waveform.size(1)
        if length > 16000 * 3: # 如果大于3秒
            max_start = length - 16000 * 3
            start = random.randint(0, max_start)
            waveform = waveform[:, start:start + 16000 * 3]

        return waveform

# 组合使用
train_pipeline = nn.Sequential(
    TrainAugmentation(),
    AudioPreprocessor() # 接上面的特征提取器
)
```