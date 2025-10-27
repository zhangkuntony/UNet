import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, Cropping2D, Concatenate
from tensorflow.keras.layers import Activation, BatchNormalization

class UNet:
    def __init__(self, image_size, num_classes, features=64, depth=3):
        """
        UNet模型类
        
        Args:
            image_size: 输入图像尺寸 (height, width)
            num_classes: 分类数量
            features: 初始特征通道数
            depth: 网络深度
        """
        self.image_size = image_size
        self.num_classes = num_classes
        self.features = features
        self.depth = depth
        self.model = self.unet()
    
    def _downsampling_block(self, input_tensor, filters):
        """编码部分的下采样块"""
        # 卷积
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
        # 返回
        return MaxPooling2D(pool_size=(2, 2))(x), x

    def _upsampling_block(self, input_tensor, skip_tensor, filters):
        """解码部分的上采样块"""
        # 反卷积
        x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        # 尺寸
        _, x_height, x_width, _ = x.shape
        _, s_height, s_width, _ = skip_tensor.shape
        # 计算差异
        h_crop = s_height - x_height
        w_crop = s_width - x_width
        # 判断是否进行裁剪
        if h_crop == 0 and w_crop == 0:
            y = skip_tensor
        else:
            # 获取裁剪的大小
            cropping = ((h_crop//2, h_crop-h_crop//2), (w_crop//2, w_crop-w_crop//2))
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

    def unet(self):
        """构建UNet模型"""
        # 定义输入
        inputs = keras.Input(shape=(self.image_size + (3,)))
        x = inputs
        
        # 构建编码部分
        skips = []
        features = self.features
        
        for i in range(self.depth):
            x, x0 = self._downsampling_block(x, features)
            skips.append(x0)
            features *= 2

        # 底部卷积
        x = Conv2D(filters=features, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=features, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 构建解码部分
        for i in reversed(range(self.depth)):
            features //= 2
            x = self._upsampling_block(x, skips[i], features)
            
        # 输出层
        x = Conv2D(filters=self.num_classes, kernel_size=(1, 1), padding='same')(x)
        outputs = Activation('softmax')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs)
    
    # def summary(self):
    #     """显示模型摘要"""
    #     return self.model.summary()
    #
    # def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    #     """编译模型"""
    #     self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #
    # def get_model(self):
    #     """获取Keras模型对象"""
    #     return self.model

