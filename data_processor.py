import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageOps

class DataProcessor:
    def __init__(self, input_dir='segdata/images/', target_dir='segdata/annotations/trimaps/'):
        # 图像位置
        self.input_dir = input_dir
        # 标注位置
        self.target_dir = target_dir

    def load_data(self):
        input_img_path = sorted([os.path.join(self.input_dir, fname)
                                 for fname in os.listdir(self.input_dir) if fname.endswith('.jpg')])

        print('Number of images:', len(input_img_path))
        print('First 5 images:', input_img_path[:5])

        target_img_path = sorted([os.path.join(self.target_dir, fname)
                                  for fname in os.listdir(self.target_dir) if fname.endswith('.png') and not fname.startswith('.')])

        print('Number of target files:', len(target_img_path))
        print('First 5 target files:', target_img_path[:5])

        return input_img_path, target_img_path

    def display_image(self, index):
        input_img_path, target_img_path = self.load_data()
        
        # 显示输入图像
        input_img = Image.open(input_img_path[index])
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.title('Input Image')
        plt.axis('off')
        
        # 显示目标图像
        target_img = ImageOps.autocontrast(load_img(target_img_path[index]))
        plt.subplot(1, 2, 2)
        plt.imshow(target_img)
        plt.title('Target Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    processor = DataProcessor()
    processor.display_image(0)
