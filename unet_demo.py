import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Cropping2D, Concatenate
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps, Image

# 图片位置
input_dir = "segdata/images/"
# 标注信息位置
target_dir = "segdata/annotations/trimaps/"
# 图像大小设置及类别信息
img_size = (160, 160)
batch_size = 32
num_classes = 4
# 图像的路径
input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
# 目标路径
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

# 显示输入图像
input_img = Image.open(input_img_paths[10])
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(input_img)
plt.title('Input Image')
plt.axis('off')

# 显示目标图像
target_img = PIL.ImageOps.autocontrast(load_img(target_img_paths[10]))
plt.subplot(1, 2, 2)
plt.imshow(target_img)
plt.title('Target Image')
plt.axis('off')

plt.tight_layout()
plt.show()

class OxfordPets(keras.utils.Sequence):
    # 初始化
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        # 批次大小
        self.batch_size = batch_size
        # 图像大小
        self.img_size = img_size
        # 图像的路径
        self.input_img_paths = input_img_paths
        # 目标值路径
        self.target_img_paths = target_img_paths

    # 迭代次数
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    # 获取batch数据
    def __getitem__(self, idx):
        # 获取该批次对应的样本的索引
        i = idx * self.batch_size
        # 获取该批次数据
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        # 构建特征值
        x = np.zeros((batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        # 构建目标值
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)

        return x, y

# 编码部分
# 输入：输入张量，卷积核个数
def downsampling_block(input_tensor, filters):
    # 输入层
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(input_tensor)
    # BN层
    x = BatchNormalization()(x)
    # 激活层
    x = Activation("relu")(x)
    # 卷积层
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    # BN层
    x = BatchNormalization()(x)
    # 激活层
    x = Activation("relu")(x)
    # 返回的是池化后的值和激活未池化的值，激活后未池化的值用于解码部分特征级联
    return MaxPooling2D(pool_size=(2, 2))(x), x

# 解码部分
# 输入：输入张量，特征融合的张量，卷积核个数
def upsampling_block(input_tensor, skip_tensor, filters):
    # 反卷积
    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding="same")(input_tensor)
    # 获取当前特征图的尺寸
    _, x_height, x_width, _ = x.shape
    # 获取要融合的特征图的尺寸
    _, s_height, s_width, _ = skip_tensor.shape
    # 获取特征图的大小差异
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    # 若特征图大小相同不进行裁剪
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    # 若特征图大小不同，使级联时像素大小一致
    else:
        # 获取特征图裁剪后的特征图的大小
        cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
        # 特征图裁剪
        y = Cropping2D(cropping=cropping)(input_tensor)

    # 特征融合
    x = Concatenate()([x, y])
    # 卷积
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    # BN层
    x = BatchNormalization()(x)
    # 激活层
    x = Activation("relu")(x)
    # 卷积层
    x = Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    # BN层
    x = BatchNormalization()(x)
    # 激活层
    x = Activation("relu")(x)
    return x

# 模型构建
# 使用3个深度构建unet网络
def unet(imagesize, classes, features=64, depth=3):
    # 定义输入数据
    inputs = keras.Input(shape=img_size + (3,))
    x = inputs
    # 用来存放进行特征融合的特征图
    skips = []
    # 构建编码部分
    for i in range(depth):
        x, x0 = downsampling_block(x, features)
        skips.append(x0)
        # 下采样过程中，深度增加，特征翻倍，即每次使用翻倍数目的滤波器
        features *= 2

    # 卷积
    x = Conv2D(filters=features, kernel_size=(3, 3), padding="same")(x)
    # BN层
    x = BatchNormalization()(x)
    # 激活
    x = Activation("relu")(x)
    # 卷积
    x = Conv2D(filters=features, kernel_size=(3, 3), padding="same")(x)
    # BN层
    x = BatchNormalization()(x)
    # 激活
    x = Activation("relu")(x)

    # 解码过程
    for i in reversed(range(depth)):
        # 深度增加，特征图通道减半
        features //= 2
        # 上采样
        x = upsampling_block(x, skips[i], features)

    # 卷积
    x = Conv2D(filters=classes, kernel_size=(1, 1), padding="same")(x)
    # 激活
    outputs = Activation("softmax")(x)
    # 模型定义
    model = keras.Model(inputs, outputs)
    return model

model = unet(img_size, 4)
model.summary()

keras.utils.plot_model(model)


# 将数据集划分为训练集和验证集，其中验证集的数量为1000
val_samples = 1000
# 将数据集打乱（图像与标注信息的随机数种子是一样的，才能保证数据的正确性）
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
# 获取训练集数据路径
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
# 获取验证集数据路径
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# 获取训练集
train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
# 模拟验证集
val_gen = OxfordPets(
    batch_size, img_size, val_input_img_paths, val_target_img_paths
)

# 模型编译
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

# 模型训练，epoch设为15
epochs = 1
model.fit(train_gen, epochs=epochs, validation_data=val_gen)

# 模型预测
# 获取验证集数据，并进行预测
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)

# 图像显示
def display_mask(i):
    # 获取到第i个样本的预测结果
    mask = np.argmax(val_preds[i], axis=-1)
    # 纬度调整
    mask = np.expand_dims(mask, axis=-1)
    # 转换为图像，并进行显示
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    return img

i= 10
# 显示输入图像
input_img = Image.open(val_input_img_paths[i])
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(input_img)
plt.title('Input Image')
plt.axis('off')

# 显示目标图像
target_img = display_mask(i)
plt.subplot(1, 2, 2)
plt.imshow(target_img)
plt.title('Target Image')
plt.axis('off')

plt.tight_layout()
plt.show()