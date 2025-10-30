import PIL.ImageOps
import numpy as np
import random
import tensorflow.keras as keras

import data_loader
from unet_model import unet
from oxford_pets import OxfordPets

def dataset_split(dataset_batch_size, dataset_img_size):
    input_img_path, target_img_path = data_loader.load_data()

    # 将数据集划分为训练集和验证集，其中验证集的数量设为1000
    val_samples = 1000

    # 将数据集随机打乱(图像与标注信息的随机数种子是一样的，才能保证数据的正确性)
    random.Random(1337).shuffle(input_img_path)
    random.Random(1337).shuffle(target_img_path)

    # 获取训练集数据路径
    dataset_train_input_img_paths = input_img_path[:-val_samples]
    dataset_train_target_img_paths = target_img_path[:-val_samples]
    # 获取验证集数据路径
    dataset_validate_input_img_paths = input_img_path[-val_samples:]
    dataset_validate_target_img_paths = target_img_path[-val_samples:]

    # 获取训练集
    train_gen = OxfordPets(
        dataset_batch_size, dataset_img_size, dataset_train_input_img_paths, dataset_train_target_img_paths
    )
    # 模型验证集
    val_gen = OxfordPets(
        dataset_batch_size, dataset_img_size, dataset_validate_input_img_paths, dataset_validate_target_img_paths
    )

    return dataset_validate_input_img_paths, dataset_validate_target_img_paths, train_gen, val_gen

# 模型编译及训练
def model_complie_train(model_train_data, model_validate_data, save_model=True):
    # 创建UNet模型实例
    model = unet(img_size, num_classes)

    # 模型编译
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    # 模型训练, epochs设为5
    epochs = 10
    model.fit(model_train_data, epochs=epochs, validation_data=model_validate_data)

    # 保存模型
    if save_model:
        model.save('unet_model.h5')
        print("模型已保存为 'unet_model.h5'")

    return model

# 加载已保存的模型
def load_saved_model():
    try:
        model = keras.models.load_model('unet_model.h5')
        print("成功加载已保存的模型")
        return model
    except (OSError, IOError) as e:
        print(f"模型文件读取失败: {e}")
        return None
    except ImportError as e:
        print(f"模型依赖缺失: {e}")
        return None
    except ValueError as e:
        print(f"模型文件格式错误: {e}")
        return None

def predict(model, val_gen):
    # 获取预测结果
    val_pred = model.predict(val_gen)
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

    # 尝试加载已保存的模型，如果不存在则重新训练
    unet_model = load_saved_model()
    if unet_model is None:
        print("开始训练新模型...")
        unet_model = model_complie_train(train_data, val_data, save_model=True)
    else:
        print("使用已保存的模型进行预测")
    
    # 模型预测
    predict_images = predict(unet_model, val_data)

    # 显示原图，目标图，和预测图
    index = 10
    predict_img = mask_predict_img(index, predict_images)
    data_loader.display_images_vs_predict(val_input_img_paths[index], val_target_img_paths[index], predict_img)
    
    # 保存预测结果示例
    print(f"预测完成！模型已保存为 'unet_model.h5'")
    print("下次运行将直接使用保存的模型，无需重新训练")
