import torch.nn as nn
from torchvision import models

def build_model(num_classes, backbone='resnet18', pretrained=True):
    """
    构建基于ResNet的分类模型

    参数:
        num_classes (int): 分类类别数
        backbone (str): 可选 'resnet18' 或 'resnet34'
        pretrained (bool): 是否加载ImageNet预训练权重

    返回:
        model (nn.Module): 构建好的模型
    """
    if backbone == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
