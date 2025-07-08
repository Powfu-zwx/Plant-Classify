import torch
import torch.nn as nn
from torchvision import models
from data_utils import load_image_paths_and_labels, encode_labels, get_dataloaders
from transforms import train_transform, val_transform
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 数据加载 =====
image_paths, labels = load_image_paths_and_labels("data/train")
labels_encoded, label_encoder = encode_labels(labels)
num_classes = len(label_encoder.classes_)

# 保存 label_encoder 以便 submission 使用
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

train_loader, val_loader = get_dataloaders(
    image_paths, labels_encoded, train_transform, val_transform, batch_size=32
)

# ===== 构建 ResNet18 模型 =====
model = models.resnet18(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_classes)
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# ===== 训练循环 =====
best_val_acc = 0.0
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
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {acc:.4f}")

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
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_resnet18.pth")
        print("保存当前最佳模型")

