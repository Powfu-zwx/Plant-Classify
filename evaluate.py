import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from model import build_model
from data_utils import load_image_paths_and_labels, encode_labels, get_dataloaders
from transforms import val_transform
from PIL import Image

# ===== 设置 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model_resnet34.pth"  # 修改为你保存的模型路径
batch_size = 32

# ===== 加载数据 =====
image_paths, labels = load_image_paths_and_labels("data/train")
labels_encoded, label_encoder = encode_labels(labels)
_, val_loader = get_dataloaders(image_paths, labels_encoded, val_transform, val_transform, batch_size=batch_size)
val_dataset = val_loader.dataset

# ===== 加载模型 =====
num_classes = len(label_encoder.classes_)
model = build_model(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== TTA 变换设置 =====
tta_transforms = [
    T.Compose([T.Resize((224, 224))]),
    T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0)]),
    T.Compose([T.Resize((224, 224)), T.RandomVerticalFlip(p=1.0)]),
    T.Compose([T.Resize((224, 224)), T.RandomRotation(15)]),
    T.Compose([T.Resize((224, 224)), T.ColorJitter(brightness=0.2, contrast=0.2)])
]

# ===== TTA 预测函数 =====
def predict_tta(model, dataset, device):
    all_preds = []
    true_labels = []
    with torch.no_grad():
        for img, label in dataset:
            img_preds = []
            for t in tta_transforms:
                transformed = t(img).unsqueeze(0).to(device)
                out = model(transformed)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                img_preds.append(prob)
            avg_prob = np.mean(img_preds, axis=0)
            all_preds.append(avg_prob.argmax())
            true_labels.append(label)
    return np.array(all_preds), np.array(true_labels)

# ===== 混淆矩阵绘制函数 =====
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ===== 错误样本可视化函数 =====
def show_misclassified(model, dataset, label_encoder, device):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image, true_label = dataset[i]
            input_tensor = tta_transforms[0](image).unsqueeze(0).to(device)
            pred_label = model(input_tensor).argmax(dim=1).item()
            if pred_label != true_label:
                misclassified.append((image, true_label, pred_label))
            if len(misclassified) == 9:
                break

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i, (img, true, pred) in enumerate(misclassified):
        ax = axs[i // 3, i % 3]
        ax.imshow(np.transpose(np.array(img), (1, 2, 0)))
        ax.set_title(f"T:{label_encoder.classes_[true]}, P:{label_encoder.classes_[pred]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# ===== 主流程 =====
if __name__ == "__main__":
    print("🔍 Running TTA Evaluation...")
    preds, true = predict_tta(model, val_dataset, device)
    acc = accuracy_score(true, preds)
    print(f"TTA Validation Accuracy: {acc:.4f}")

    print("Plotting confusion matrix...")
    plot_confusion_matrix(true, preds, class_names=label_encoder.classes_)

    print("Showing misclassified samples...")
    show_misclassified(model, val_dataset, label_encoder, device)
