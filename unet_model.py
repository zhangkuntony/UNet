import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Cropping2D, Concatenate
from tensorflow.keras.layers import Activation, BatchNormalization

# 输入：输入张量，卷积核个数
def downsampling_block(input_tensor, filters):
    """编码部分的下采样块"""
    # 输入层
    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(input_tensor)
    # BN
    x = BatchNormalization()(x)
    # 激活
    x = Activation('relu')(x)
    # 卷积
    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    # BN
    x = BatchNormalization()(x)
    # 激活
    x = Activation('relu')(x)
    # 返回的是池化后的值和激活未池化的值，激活后未池化的值用于解码部分特征级联
    return MaxPooling2D(pool_size=(2, 2))(x), x

# 输入：输入张量，特征融合的张量，卷积核个数
def upsampling_block(input_tensor, skip_tensor, filters):
    """解码部分的上采样块"""
    # 反卷积
    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
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
        # 获取特征图裁剪后的特征图大小
        cropping = ((h_crop//2, h_crop-h_crop//2), (w_crop//2, w_crop-w_crop//2))
        # 特征图裁剪
        y = Cropping2D(cropping=cropping)(skip_tensor)
    # 特征融合
    x = Concatenate()([x, y])
    # 卷积
    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    # BN
    x = BatchNormalization()(x)
    # 激活层
    x = Activation('relu')(x)
    # 卷积
    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    # BN
    x = BatchNormalization()(x)
    # 激活
    x = Activation('relu')(x)
    return x

# 使用3个深度侯建unet网络
def unet(imagesize, classes, features=64, depth=3):
    # 定义输入数据
    inputs = keras.Input(shape=imagesize + (3,))
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
    x = Conv2D(filters=features, kernel_size=(3, 3), padding='same')(x)
    # BN层
    x = BatchNormalization()(x)
    # 激活
    x = Activation('relu')(x)
    # 卷积
    x = Conv2D(filters=features, kernel_size=(3, 3), padding='same')(x)
    # BN层
    x = BatchNormalization()(x)
    # 激活
    x = Activation('relu')(x)

    # 构建解码部分
    for i in reversed(range(depth)):
        # 深度增加，特征图通道减半
        features //= 2
        x = upsampling_block(x, skips[i], features)

    # 卷积
    x = Conv2D(filters=classes, kernel_size=(1, 1), padding='same')(x)
    # 激活
    outputs = Activation('softmax')(x)
    # 模型定义
    model = keras.Model(inputs, outputs)

    # 查看模型结构
    model.summary()
    # 模型可视化
    keras.utils.plot_model(model)

    return model

