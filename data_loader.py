import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageOps

input_dir = "segdata/images/"
target_dir = "segdata/annotations/trimaps/"

def display_image(input_img, target_img):
    # 显示输入图像
    input_img = Image.open(input_img)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(input_img)
    plt.title('Input Image')
    plt.axis('off')

    # 显示目标图像
    target_img = ImageOps.autocontrast(load_img(target_img))
    plt.subplot(1, 2, 2)
    plt.imshow(target_img)
    plt.title('Target Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def display_images_vs_predict(input_img, target_img, predict_img):
    # 显示输入图像
    input_img = Image.open(input_img)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(input_img)
    plt.title('Input Image')
    plt.axis('off')

    # 显示目标图像
    target_img = ImageOps.autocontrast(load_img(target_img))
    plt.subplot(1, 3, 2)
    plt.imshow(target_img)
    plt.title('Target Image')
    plt.axis('off')

    # 显示预测图像
    predict_img = ImageOps.autocontrast(predict_img)
    plt.subplot(1, 3, 3)
    plt.imshow(predict_img)
    plt.title('Predict Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def load_data():
    input_img_path = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith('.jpg')
        ]
    )

    target_img_path = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith('.png') and not fname.startswith('.')
        ]
    )

    return input_img_path, target_img_path

if __name__ == '__main__':
    input_path, target_path = load_data()
    index = 10
    display_image(input_path[index], target_path[index])
