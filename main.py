import numpy as np
import random
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import data_processor
from dataset_generator import OxfordPets
from unet_model import UNet


def dataset_split(batch_size, img_size):
    input_img_path, target_img_path = data_processor.DataProcessor().load_data()

    # 验证集数量
    val_samples = 1000

    # 随机打乱
    random.Random(1337).shuffle(input_img_path)
    random.Random(1337).shuffle(target_img_path)

    # 划分数据集
    # 训练集
    train_input_img_paths = input_img_path[:-val_samples]
    train_target_img_paths = target_img_path[:-val_samples]
    # 验证集
    val_input_img_paths = input_img_path[-val_samples:]
    val_target_img_paths = target_img_path[-val_samples:]

    # 数据集获取
    train_gen = OxfordPets(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    return train_gen, val_gen

def model_complie_train(train_data, val_data):
    # 创建UNet模型实例
    unet_model = UNet(image_size=img_size, num_classes=num_classes)
    unet_model = unet_model.unet()

    # 显示模型摘要
    unet_model.summary()

    # 显示模型架构
    keras.utils.plot_model(unet_model, to_file='unet_model.png', show_shapes=True, show_layer_names=True)

    print("UNet模型创建成功！")

    # 模型编译
    unet_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    # 模型训练
    unet_model.fit(train_data, epochs=1, validation_data=val_data)

    return unet_model

def predict(unet_model, index, val_gen):
    # 获取预测结果
    val_pred = unet_model.predict(val_gen)
    pred = val_pred[index]
    mask = np.argmax(pred, axis=-1)
    
    # 直接从验证集数据中获取对应的图像
    # 获取验证集的batch索引和图像索引
    batch_idx = index // val_gen.batch_size
    img_idx = index % val_gen.batch_size
    
    # 获取对应的batch数据
    val_batch = val_gen[batch_idx]
    original_img = val_batch[0][img_idx]  # 输入图像
    true_mask = val_batch[1][img_idx]     # 真实mask
    
    # 将图像数据转换为可显示的格式
    original_img_display = (original_img * 255).astype(np.uint8)
    true_mask_display = np.squeeze(true_mask, axis=-1)
    
    # 创建对比图
    plt.figure(figsize=(10, 10))
    
    # 显示原图
    plt.subplot(2, 2, 1)
    plt.imshow(original_img_display)
    plt.title('Original Image (from val_data)')
    plt.axis('off')
    
    # 显示真实mask
    plt.subplot(2, 2, 2)
    plt.imshow(true_mask_display, cmap='viridis')
    plt.title('True Mask')
    plt.axis('off')
    
    # 显示预测mask
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='viridis')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # 显示叠加效果
    plt.subplot(2, 2, 4)
    plt.imshow(original_img_display)
    plt.imshow(mask, cmap='viridis', alpha=0.5)  # 半透明叠加
    plt.title('Overlay Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"预测完成！显示第 {index} 张图像的对比效果")
    print(f"原始图像形状: {original_img.shape}")
    print(f"预测Mask形状: {mask.shape}")
    print(f"真实Mask形状: {true_mask.shape}")
    print(f"预测Mask中类别分布: {np.unique(mask, return_counts=True)}")
    print(f"真实Mask中类别分布: {np.unique(true_mask_display, return_counts=True)}")

if __name__ == '__main__':
    # 定义模型参数
    img_size = (160, 160)
    batch_size = 32
    num_classes = 4

    # 划分数据集
    train_data, val_data = dataset_split(batch_size, img_size)

    # 模型编译&训练
    unet = model_complie_train(train_data, val_data)

    # 获取验证集图像路径用于预测
    input_img_path, target_img_path = data_processor.DataProcessor().load_data()
    val_input_img_paths = input_img_path[-1000:]  # 验证集图像路径
    
    # 模型预测
    predict(unet, 10, val_data)
    
