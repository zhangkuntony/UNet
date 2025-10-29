from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

class OxfordPets(keras.utils.Sequence):
    # 在__init__方法中指定batch_size,img_size,input_img_paths,target_img_paths
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        # 计算迭代次数
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        获取每一个batch数据
        """
        i = idx * self.batch_size
        # 获取输入的图像数据
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        # 获取标签数据
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        # 构建特征值数据：获取图像数据中每个像素的数据存储在x中
        x = np.zeros((self.batch_size,)+self.img_size+(3,),dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        # 构建目标值数据：获取标注图像中每个像素中的数据存在y中
        y = np.zeros((self.batch_size,)+self.img_size+(1,),dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img,2)

        return x, y