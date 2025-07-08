import torch
import torch.nn as nn
from model import build_model
from data_utils import load_image_paths_and_labels, encode_labels, get_dataloaders, load_test_images
from transforms import train_transform, val_transform, test_transform
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 数据加载 =====
image_paths, labels = load_image_paths_and_labels("data/train")
labels_encoded, label_encoder = encode_labels(labels)
num_classes = len(label_encoder.classes_)

train_loader, val_loader = get_dataloaders(
    image_paths, labels_encoded, train_transform, val_transform, batch_size=32
)

# ===== 模型初始化 =====
model = build_model(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

loss_list = []
train_loss_per_epoch = []
train_acc_per_epoch = []
best_acc = 0.0

# ===== 训练循环 =====
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss_list.append(loss.item())
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    acc = correct / total
    train_loss_per_epoch.append(avg_loss)
    train_acc_per_epoch.append(acc)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {acc:.4f}")
    scheduler.step()

    # ===== 验证 =====
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Acc: {val_acc:.4f}")
    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model_resnet34.pth")
        print(f"[Saved best model @ Epoch {epoch+1} with acc: {val_acc:.4f}]")

# 保存训练日志
with open("log_resnet34_adagrad.pkl", "wb") as f:
    pickle.dump({"loss": train_loss_per_epoch, "acc": train_acc_per_epoch}, f)

# ===== 生成 submission.csv =====
# 加载测试图像
test_loader, test_image_names = load_test_images("data/test", test_transform, batch_size=32)

# 加载 best model
model.load_state_dict(torch.load("best_model_resnet34.pth"))
model.eval()

all_preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())

# 将预测的编码转回标签
predicted_labels = label_encoder.inverse_transform(all_preds)

# 生成 submission.csv
submission = pd.DataFrame({
    "file": test_image_names,
    "species": predicted_labels
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission.csv 已生成")