import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import PlantDataset  # 记得写好这个类

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.image_names = [os.path.basename(p) for p in self.image_paths]
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image

def load_test_images(test_dir, transform, batch_size=32):
    dataset = TestDataset(test_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataset.image_names

# 加载图像路径与标签
def load_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(class_name)

    return image_paths, labels

# 标签编码器
def encode_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return labels_encoded, le

# 划分数据 + 构造 DataLoader
def get_dataloaders(image_paths, labels_encoded, train_transform, val_transform, batch_size=32):
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
    )

    train_dataset = PlantDataset(X_train, y_train, transform=train_transform)
    val_dataset = PlantDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader