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
    def __getitem__(self, idx, batch_size=32):
        # 获取该批次对应的样本的索引
        i = idx * self.batch_size
        # 获取该批次数据
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        # 构建特征值
        x = np.zeros((batch_size,)+self.img_size+(3,),dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        # 构建目标值
        y = np.zeros((batch_size,)+self.img_size+(1,),dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)

        return x, y