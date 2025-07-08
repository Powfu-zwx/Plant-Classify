import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import pickle
from model import build_model

# 1. 参数和路径配置
test_dir = "data/test"  # 测试集文件夹路径
model_path_resnet18 = "best_model_resnet18.pth"
model_path_resnet34 = "best_model_resnet34.pth"
label_encoder_path = "label_encoder.pkl"
num_classes = 12  # 根据你的数据调整

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载LabelEncoder
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# 3. 定义数据预处理（与训练时保持一致）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 4. 加载模型函数
def load_trained_model(model_path, backbone):
    model = build_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 5. 载入两个模型
model18 = load_trained_model(model_path_resnet18, backbone='resnet18')
model34 = load_trained_model(model_path_resnet34, backbone='resnet34')

# 6. 预测所有测试图像
test_filenames = sorted(os.listdir(test_dir))
predictions = []

with torch.no_grad():
    for fname in test_filenames:
        img_path = os.path.join(test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = test_transform(img).unsqueeze(0).to(device)  # 增加batch维度

        out18 = model18(img_tensor)
        out34 = model34(img_tensor)

        # 简单平均融合
        output = (out18 + out34) / 2
        pred = output.argmax(dim=1).item()
        predictions.append(pred)

# 7. 反编码数字标签为类别名称
pred_labels = label_encoder.inverse_transform(predictions)

# 8. 生成提交文件
submission = pd.DataFrame({
    "file": test_filenames,
    "species": pred_labels
})

submission.to_csv("submission.csv", index=False)
print("submission.csv 文件已生成，准备上传Kaggle！")
