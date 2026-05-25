# Noise2Void PyTorch Implementation

Noise2Void - Learning Denoising From Single Noisy Images
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 
# 论文理念

Noise2Void (N2V) 是一种自监督图像去噪方法。它的核心思想是：

- 传统去噪方法需要"噪声图-干净图"配对（有监督学习），这在显微镜/医学影像领域极难获取
- N2V 只需要含噪声的图片即可训练，无需干净 GT
- 原理：从噪声图自身学习，让网络用周围像素预测某个被遮住的像素（盲点），本质是学习"训练数据的期望值 = 干净信号"这一统计规律

适用场景：显微镜图像、CT/MRI 医学影像、天文图像等无法获取真值的领域。


# 项目结构


```shell
Noise2Void-pytorch-implementation/
├── config.yaml          # 训练/推理配置
├── train.py             # 训练主脚本 + loss 函数
├── infer.py             # 推理脚本
├── network.py           # U-Net 模型定义
├── datasets/
│   ├── __init__.py      # 包导出
│   └── dataset.py       # N2V 数据集（随机裁剪 + 盲点 mask）
├── data/                # 输入图像目录
├── checkpoints/         # 保存权重
├── outputs/             # 推理输出
└── results/             # loss 记录
```

# 核心设计

  1. 盲点训练策略

  每一轮取一张原图 → 随机裁剪 patch（训练时采用512×512）→ 选 2% 像素作为盲点 → 每个盲点从其邻域随机取一个像素替换 → 网络用替换后的图预测原图，loss只在盲点位置计算。

  这样网络学到的是：利用周围上下文补全被遮住的像素。给定纯噪声是无法预测的，所以网络只能输出该位置的期望值，也就是去噪后的信号。

  2. U-Net 主干

  Encoder-Bottleneck-Decoder 结构，5 层深度，跳跃连接。上采样用 bilinear + 1×1 conv（避免转置卷积的棋盘效应），输出加 sigmoid 约束到 [0,1]。

  3. Loss

  MSE 仅在盲点位置计算：MSE(pred ⊙ mask, target ⊙ mask)。直觉上就是只关心网络能否从周围上下文恢复被遮住的干净像素。

# 修复
  - CosineAnnealingLR 的 verbose 参数在旧 PyTorch 版本不兼容
  - pixel_mse_loss 中 tensor 解包在不同 PyTorch 版本行为不一致（改为返回 mask 张量彻底绕过问题）
  - 删除 _pick_pixel_pos 等临时性解析代码，数据流更清晰


# pipeline

  训练:
    data/*.png → random_crop(64×64) → blind_spot_mask(2%像素) → U-Net → MSE(masked only) → 反向传播

  推理:
    data/*.png → 全图送入 U-Net → sigmoid → clamp → 输出去噪结果
