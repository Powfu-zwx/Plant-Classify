import torch
import torch.nn.functional as F
from torchvision import models
from data_utils import load_test_images
from transforms import test_transform
import pandas as pd
import os
import pickle

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 label_encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 加载测试数据
test_loader, image_names = load_test_images("data/test", test_transform, batch_size=32)

# 加载模型函数
def load_model(model_name, weights_path, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    else:
        raise ValueError("Unsupported model name")

    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

num_classes = len(label_encoder.classes_)

# 加载两个模型
model18 = load_model("resnet18", "best_model_resnet18.pth", num_classes)
model34 = load_model("resnet34", "best_model_resnet34.pth", num_classes)

# 开始融合预测
all_preds = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)

        outputs18 = F.softmax(model18(images), dim=1)
        outputs34 = F.softmax(model34(images), dim=1)

        # soft voting 融合（等权重平均）
        outputs = (outputs18 + outputs34) / 2

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

# 反编码为标签名
labels = label_encoder.inverse_transform(all_preds)

# 写入 submission 文件
submission = pd.DataFrame({
    "file": image_names,
    "species": labels
})
submission.to_csv("submission_ensemble.csv", index=False)

print("融合完成，已保存为 submission_ensemble.csv")
