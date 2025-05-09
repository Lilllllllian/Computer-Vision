# 项目概述

## 任务一：Caltech-101图像分类

### 项目背景与目标
Caltech-101数据集是一个广泛用于图像分类任务的数据集，包含101个类别以及一个背景类别，共计约9,146张图像。本任务的目标是利用在ImageNet上预训练的CNN模型（如ResNet-18），通过微调（fine-tuning）方法，调整网络结构以适应Caltech-101数据集的分类需求，从而实现高效的图像分类。

### 方法与步骤
- **数据集准备**：按照Caltech-101数据集的标准划分训练集和测试集。
- **模型调整**：选择ResNet-18作为基础模型，将其输出层大小修改为101，以匹配Caltech-101数据集的类别数量。使用ImageNet预训练参数初始化其余层。
- **微调训练**：对新的输出层从零开始训练，同时以较小的学习率微调其余层的参数。
- **超参数调优**：通过调整学习率、训练步数等超参数，优化模型性能。
- **对比实验**：与仅使用Caltech-101数据集从随机初始化的网络参数开始训练的模型进行对比，验证预训练模型的优势。
- **可视化**：利用TensorBoard可视化训练过程中的损失曲线和验证集准确率变化。

### 实验结果与结论
通过微调预训练的ResNet-18模型，在Caltech-101数据集上取得了显著的分类效果提升。与随机初始化模型相比，预训练模型在验证集和测试集上均表现出更高的准确率和更低的损失，验证了迁移学习的有效性。

### 代码与模型权重
本项目的代码已提交至GitHub仓库[^1]。训练好的模型权重文件已上传至Google Drive，包括预训练最佳模型链接[^2]和随机初始化最佳模型链接[^3]供下载使用。

[^1]: https://github.com/Lilllllllian/Computer-Vision/tree/main
[^2]: https://drive.google.com/file/d/19O-4N0eaNmykoC1IrbHPbtczzjAqlN4Q/view?usp=drive_link
[^3]: https://drive.google.com/file/d/1EjNQ3ztvaKSL9QGn55jHhnzDs-OtBbeN/view?usp=drive_link
