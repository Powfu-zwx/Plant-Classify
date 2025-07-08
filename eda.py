import os

train_dir = 'data/train'
class_counts = {
    folder: len(os.listdir(os.path.join(train_dir, folder)))
    for folder in os.listdir(train_dir)
}
print("类别样本数：", class_counts)


import matplotlib.pyplot as plt
from PIL import Image
import os

def show_sample_images(base_path, class_name, n=8):
    files = os.listdir(os.path.join(base_path, class_name))[:n]
    plt.figure(figsize=(12, 6))
    for i, file in enumerate(files):
        img_path = os.path.join(base_path, class_name, file)
        image = Image.open(img_path)
        plt.subplot(2, 4, i+1)
        plt.imshow(image)
        plt.title(class_name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# 示例调用：显示 Sugar beet 类别的图像样本
show_sample_images("data/train", "Sugar beet")

import matplotlib.pyplot as plt

# 可视化类别分布柱状图
def plot_class_distribution(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in Training Set')
    plt.tight_layout()
    plt.show()

plot_class_distribution(class_counts)
