from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

class OxfordPets(keras.utils.Sequence):
    # 初始化
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        # 批次大小
        super().__init__()
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
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        # 构建特征值
        x = np.zeros((self.batch_size,)+self.img_size+(3,),dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = np.array(img) / 255.0  # 转换为numpy数组并归一化到[0,1]

        # 构建目标值
        y = np.zeros((self.batch_size,)+self.img_size+(1,),dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img_array = np.array(img)
            
            # 你的mask数据包含像素值[1, 2, 3]，需要转换为类别索引
            # 注意：像素值1对应类别1，像素值2对应类别2，像素值3对应类别3
            # 背景类别为0
            mask = img_array.copy()
            # 由于像素值已经是1,2,3，我们只需要确保它们是正确的整数类别
            # 不需要额外的映射，因为sparse_categorical_crossentropy期望整数标签
            
            y[j] = np.expand_dims(mask, 2)

        return x, y