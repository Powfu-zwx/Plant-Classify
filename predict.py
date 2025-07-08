import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from model import build_model
from data_utils import encode_labels
from transforms import val_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 加载模型 =====
num_classes = 12
model = build_model(num_classes).to(device)
model.load_state_dict(torch.load("best_model_resnet18.pth"))
model.eval()

# ===== 构建标签映射（与训练保持一致）=====
# 重新读取训练集 label
image_paths, labels = [], []
train_dir = "data/train"
for cls in os.listdir(train_dir):
    for fname in os.listdir(os.path.join(train_dir, cls)):
        image_paths.append(os.path.join(train_dir, cls, fname))
        labels.append(cls)

_, label_encoder = encode_labels(labels)
idx_to_class = {i: c for i, c in enumerate(label_encoder.classes_)}

# ===== 预测 =====
test_dir = "data/test"
results = []

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    image = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred_idx = output.argmax(dim=1).item()
        pred_label = idx_to_class[pred_idx]
        results.append((img_name, pred_label))

# ===== 保存 submission.csv =====
df = pd.DataFrame(results, columns=["file", "species"])
df.sort_values("file", inplace=True)  # 按文件名排序（Kaggle规范）
df.to_csv("submission.csv", index=False)
print("已生成 submission.csv，可上传至 Kaggle 评测")
