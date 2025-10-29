# UNet 图像分割项目

一个基于TensorFlow/Keras实现的U-Net图像分割项目，用于语义分割任务。

## 项目概述

本项目实现了经典的U-Net架构，用于图像语义分割任务。支持Oxford Pets数据集的分割，可以扩展到其他图像分割任务。

## 项目结构

```
UNet/
├── data_processor.py      # 数据预处理和可视化
├── dataset_generator.py   # 数据生成器（OxfordPets数据集）
├── unet_model.py          # UNet模型定义（类封装）
├── main.py               # 主程序入口
├── unet_model.png        # 模型架构图
├── README.md             # 项目说明文档
└── segdata/              # 图像分割数据目录
    ├── images/           # 输入图像（.jpg格式）
    └── annotations/      # 标注图像（.png格式）
```

## 功能特性

### 1. 数据预处理
- **数据加载**：自动加载图像和标注数据
- **数据可视化**：显示原始图像和标注的对比效果
- **数据集划分**：自动划分训练集和验证集

### 2. UNet模型
- **编码器-解码器架构**：标准的U-Net结构
- **跳跃连接**：保留空间信息
- **BatchNormalization**：加速训练收敛
- **可配置参数**：支持自定义图像尺寸、分类数、网络深度等

### 3. 训练和评估
- **数据生成器**：支持批量数据加载
- **模型训练**：完整的训练流程
- **可视化预测**：显示原图、真实mask、预测mask的对比效果

## 安装依赖

```bash
pip install tensorflow matplotlib numpy pillow
```

### 可选依赖（用于模型可视化）
```bash
pip install graphviz
# 还需要安装Graphviz可执行文件：https://graphviz.org/download/
```

## 快速开始

### 1. 准备数据
确保数据目录结构如下：
```
segdata/
├── images/           # 输入图像（.jpg格式）
└── annotations/      # 标注图像（.png格式）
```

### 2. 运行项目
```bash
python main.py
```

### 3. 自定义配置
在`main.py`中修改以下参数：
```python
# 定义模型参数
img_size = (160, 160)     # 图像尺寸
batch_size = 32           # 批次大小
num_classes = 4           # 分类数量
```

## 代码模块说明

### data_processor.py
数据预处理模块，主要功能：
- 加载图像和标注数据
- 显示图像和标注的对比效果

```python
from data_loader import DataLoader

# 创建数据处理器
processor = DataLoader()

# 加载数据
input_paths, target_paths = processor.load_data()

# 显示图像
display_image(0)
```

### dataset_generator.py
数据生成器模块，继承自`keras.utils.Sequence`：
- 批量加载图像数据
- 自动调整图像尺寸
- 支持多线程数据加载

### unet_model.py
UNet模型类，主要方法：
- `__init__()`: 初始化模型参数
- `_build_model()`: 构建模型架构
- `compile()`: 编译模型
- `summary()`: 显示模型摘要
- `get_model()`: 获取Keras模型对象

```python
from unet_model import UNet

# 创建UNet模型
unet = UNet(image_size=(160, 160), num_classes=4)

# 编译模型
unet.compile(optimizer='adam', loss='categorical_crossentropy')

# 显示模型结构
unet.summary()
```

### main.py
主程序入口，包含完整的工作流程：
1. 数据准备和划分
2. 模型创建和编译
3. 模型训练
4. 预测和可视化

## 模型架构

UNet模型采用经典的编码器-解码器结构：

### 编码器（下采样）
- 卷积层 + BatchNormalization + ReLU激活
- 最大池化层进行下采样
- 跳跃连接保留特征信息

### 解码器（上采样）
- 转置卷积进行上采样
- 与编码器特征融合
- 卷积层恢复空间信息

### 输出层
- 1x1卷积输出分割结果
- Softmax激活函数进行多分类

## 训练配置

### 损失函数
- 多分类交叉熵损失：`sparse_categorical_crossentropy`

### 优化器
- RMSprop优化器（默认）
- 支持自定义优化器配置

### 训练参数
```python
# 训练配置示例
unet_model.fit(
    train_data, 
    epochs=10, 
    validation_data=val_data, 
    steps_per_epoch=100,
    validation_steps=20
)
```

## 预测和可视化

项目提供完整的预测可视化功能：

### 四图对比显示
1. **原始图像**：输入图像
2. **真实mask**：标注的ground truth
3. **预测mask**：模型预测结果
4. **叠加效果**：原图与预测mask的叠加

### 调试信息
- 图像和mask的形状信息
- 类别分布统计
- 预测准确性评估

## 扩展和自定义

### 自定义数据集
修改`data_processor.py`中的数据路径：
```python
class DataProcessor:
    def __init__(self, input_dir='your_images/', target_dir='your_annotations/'):
        self.input_dir = input_dir
        self.target_dir = target_dir
```

### 修改模型参数
```python
# 创建自定义UNet模型
unet = UNet(
    image_size=(256, 256),    # 自定义图像尺寸
    num_classes=10,           # 自定义分类数
    features=32,             # 自定义特征通道数
    depth=4                  # 自定义网络深度
)
```

## 常见问题

### 1. Graphviz安装问题
如果模型可视化失败，请确保正确安装Graphviz：
- 安装Python包：`pip install graphviz`
- 下载并安装Graphviz可执行文件
- 将Graphviz的bin目录添加到系统PATH

### 2. 内存不足问题
对于大尺寸图像，可以：
- 减小batch_size
- 降低图像尺寸
- 使用数据生成器减少内存占用

### 3. 训练不收敛
- 检查学习率设置
- 验证数据预处理是否正确
- 确认损失函数适合任务类型

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。