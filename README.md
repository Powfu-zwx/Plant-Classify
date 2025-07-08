# 🌱 Plant Seedlings Classification with ResNet18 & ResNet34 Ensemble

本项目基于 Kaggle 的 [Plant Seedlings Classification](https://www.kaggle.com/competitions/plant-seedlings-classification) 比赛，通过训练深度卷积神经网络模型，对 12 类植物幼苗图像进行分类，最终融合 ResNet18 和 ResNet34 模型，达到 **0.96347** 的提交准确率。

## 🚀 项目亮点

数据预处理与增强

实现包括随机裁剪、旋转、水平翻转、颜色抖动等多种数据增强策略，提升模型泛化能力。

设计训练/验证集划分方案，保证类别分布均衡。

模型设计与训练

基于PyTorch构建ResNet18和ResNet34骨干网络，采用预训练权重，进行迁移学习。

设计冻结与解冻策略，逐层微调，显著提升模型表现。

实现多种优化器（SGD、Momentum、AdaGrad、Adam）对比，最终选用AdaGrad。

集成学习率调度器（ReduceLROnPlateau），自动调节学习率，提升收敛速度和稳定性。

模型融合与性能提升

实现ResNet18与ResNet34模型输出概率平均融合，提升分类准确率。
实现测试时增强（TTA），通过多次增强预测概率平均，进一步提高测试准确率。

模型评估与错误分析

绘制混淆矩阵，定位模型易混淆类别，指导后续针对性优化。
收集错误分类样本，分析失败案例，制定数据增强及模型调整策略。

提交文件自动生成

设计推理脚本，实现批量预测与结果反编码，自动生成符合Kaggle格式的提交文件。
完成多次Kaggle提交，准确率稳定达到95%以上。

代码模块化与工程化

数据加载、预处理、模型定义、训练、推理分离成独立模块，便于维护和复用。
详细注释与README文档，确保项目易于他人理解和复现。


## 🧠 模型结构

- `ResNet18` 和 `ResNet34`，预训练自 ImageNet
- 解冻 layer4 + fc 层进行微调
- 最终预测结果为两个模型 softmax 概率加权平均


## 📁 项目结构说明

plant_classifier/

├── data/ # 存放训练/测试图像（本地）Kaggle本地可下载，这里不上传了

│ └── train/ # 12 个类别文件夹

├── model.py # 构建模型结构（支持 resnet18/resnet34）

├── train_resnet18.py # 训练 ResNet18 主逻辑

├── train_resnet34.py # 训练 ResNet34 主逻辑

├── ensemble.py # 融合两个模型生成提交文件

├── evaluate.py # TTA、混淆矩阵、错误分析等可视化

├── data_utils.py # 加载数据集、标签编码、DataLoader 构造

├── transforms.py # 定义数据增强与图像预处理

├── submission.csv # 提交文件样例

├── best_model_resnet18.pth # ResNet18 最优模型权重

├── best_model_resnet34.pth # ResNet34 最优模型权重

└── README.md

## 📦 依赖环境
bash

pip install -r requirements.txt：

torch>=2.0.0

torchvision>=0.15.0

scikit-learn

matplotlib

seaborn

numpy

pillow

🏁 快速开始

🔧 1. 训练模型
python train_resnet18.py

python train_resnet34.py

🤝 2. 模型融合

python ensemble.py

会生成 submission.csv 文件用于提交。

📊 3. 评估模型

python evaluate.py

展示：

TTA后验证集准确率

混淆矩阵图

前9个错误分类样本

📌 结果成绩
模型	验证集准确率	Kaggle提交分数

ResNet18	~91.7%	0.95969

ResNet34	~91.9%	0.95895

融合模型	✅ 93.6%	✅ 0.96347


📬 联系我
如有任何交流合作或疑问，欢迎联系我：
📮 Email：1011046478@qq.com
