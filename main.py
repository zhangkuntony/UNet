import PIL.ImageOps
import numpy as np
import random
import tensorflow.keras as keras

import data_loader
from unet_model import unet
from oxford_pets import OxfordPets

def dataset_split(batch_size, img_size):
    input_img_path, target_img_path = data_loader.load_data()

    # 将数据集划分为训练集和验证集，其中验证集的数量设为1000
    val_samples = 1000

    # 将数据集随机打乱(图像与标注信息的随机数种子是一样的，才能保证数据的正确性)
    random.Random(1337).shuffle(input_img_path)
    random.Random(1337).shuffle(target_img_path)

    # 获取训练集数据路径
    train_input_img_paths = input_img_path[:-val_samples]
    train_target_img_paths = target_img_path[:-val_samples]
    # 获取验证集数据路径
    val_input_img_paths = input_img_path[-val_samples:]
    val_target_img_paths = target_img_path[-val_samples:]

    # 获取训练集
    train_gen = OxfordPets(
        batch_size, img_size, train_input_img_paths, train_target_img_paths
    )
    # 模型验证集
    val_gen = OxfordPets(
        batch_size, img_size, val_input_img_paths, val_target_img_paths
    )

    return val_input_img_paths, val_target_img_paths, train_gen, val_gen

# 模型编译及训练
def model_complie_train(train_data, val_data):
    # 创建UNet模型实例
    model = unet(img_size, num_classes)

    # 模型编译
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    # 模型训练, epochs设为5
    epochs = 1
    model.fit(train_data, epochs=epochs, validation_data=val_data)

    return model

def predict(unet_model, val_gen):
    # 获取预测结果
    val_pred = unet_model.predict(val_gen)
    return val_pred

def mask_predict_img(i, val_pred):
    # 获取到第i个样本的预测结果
    mask = np.argmax(val_pred[i], axis=-1)
    # 纬度调整
    mask = np.expand_dims(mask, axis=-1)
    # 转换为图像，并进行显示
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    return img


if __name__ == '__main__':
    # 定义模型参数
    img_size = (160, 160)
    batch_size = 32
    num_classes = 4

    # 划分数据集
    val_input_img_paths, val_target_img_paths, train_data, val_data = dataset_split(batch_size, img_size)

    # 模型编译&训练
    unet = model_complie_train(train_data, val_data)
    
    # 模型预测
    predict_images = predict(unet, val_data)

    # 显示原图，目标图，和预测图
    index = 10
    data_loader.display_images_vs_predict(val_input_img_paths[index], val_target_img_paths[index], predict_images[index])
    
