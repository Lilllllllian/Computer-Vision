# 项目概述
## Requirements
本项目需要 Python >= 3.8。所需的包详见 requirements.txt 文件。可以使用以下命令安装它们：
```
pip install -r requirements.txt
```

## Report
- `Computer_Vision_Midterm.pdf`：完整的实验报告。

## 环境配置

- **Software Environment**
  - PyTorch 2.1.0
  - Python 3.10 (Ubuntu 22.04 LTS)
  - CUDA 12.1

- **Hardware Configuration**
  - **GPU**: NVIDIA RTX 4090 (24GB VRAM) × 1
  - **CPU**: 16 vCPU Intel® Xeon® Gold 6430

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

### 操作说明

**数据集路径**：`/main/caltech-101`  
**划分后的数据集路径**：`/main/caltech_101_split_3way`  
**相关图片保存路径**：`/root/figs`  
**模型最佳权重路径**：`/main/model`（同时已上传至 [Google Drive](https://drive.google.com/file/d/19O-4N0eaNmykoC1IrbHPbtczzjAqlN4Q/view?usp=sharing)）  
**运行记录保存路径**：`/main/runs`

完整操作流程文件：
- `TASK1_pipeline.ipynb`：包含数据处理、超参数调优、训练、测试模型的完整流程。直接克隆仓库到本地后即可运行（注意：由于项目一在 Kaggle 上运行，实际运行时可能需要调整路径）。
- `save-best-model.ipynb`：保存模型相关操作。
- `plot_results.ipynb`：可视化训练过程相关操作。

### 实验结果与结论

通过微调预训练的ResNet-18模型，在Caltech-101数据集上取得了显著的分类效果提升。与随机初始化模型相比，预训练模型在验证集和测试集上均表现出更高的准确率和更低的损失，验证了迁移学习的有效性。

### 代码与模型权重

本项目的代码已提交至GitHub仓库[^1]。训练好的模型权重文件已上传至Google Drive，包括预训练最佳模型链接[^2]和随机初始化最佳模型链接[^3]供下载使用。

## 任务二：VOC数据集上的目标检测

### 项目背景与目标

VOC数据集是一个用于目标检测和实例分割的经典数据集。本任务旨在使用MMDetection框架，训练并测试Mask R-CNN和Sparse R-CNN两个模型，比较它们在VOC数据集上的目标检测与实例分割性能。

### 方法与步骤

- **环境搭建**：配置MMDetection框架所需的环境和依赖。
- **数据集准备**：将VOC数据集转换为MMDetection支持的格式（这里为方便进行实例分割转为coco格式）；根据train.txt和val.txt的信息对原始数据集进行划分。
- **模型训练**：分别训练Mask R-CNN和Sparse R-CNN模型，记录训练过程中的损失和mAP指标；通过TensorBoard可视化训练过程中的损失曲线和验证集上的mAP曲线，分析模型性能。
- **结果可视化**：挑选测试集中的图像，对比两个模型生成的proposal box和最终预测结果；同时，对三个不在VOC数据集中的图像进行检测，可视化比较两个模型的检测效果。

### 操作说明
**数据集路径**：`/main/VOCdataset`  
**参考 JSON 文件**：由于大小限制，仅上传了部分处理为 COCO 格式的 JSON 文件。完整数据集可从 [Google Drive](https://drive.google.com/file/d/1eaVaLZrxSiniHDmCLA3xQPmgUW8hwWYK/view?usp=sharing) 下载。  
**模型架构图片路径**：`/root/figs2`  
**测试结果保存路径**：`/main/results`  
**测试图片样例路径**：`/main/test_images`  
**训练记录及可视化路径**：`/main/vis`（包含 loss 和 mAP 变化记录及 TensorBoard 可视化结果）

完整操作流程文件：
- `MASK R-CNN.ipynb` 和 `SPARSE R-CNN.ipynb`：包含数据处理、训练、测试模型的完整流程。直接克隆仓库到本地后即可运行（注意：由于项目二在 AutoDL 平台上运行，实际运行时可能需要调整路径）。
- `SparseInst.ipynb`：提供如何在 Detectron2 框架下使用 SparseInst 模型进行实例分割的示例。

### 代码与模型权重

本项目的代码已提交至GitHub仓库[^4]。训练好的模型权重文件已上传至Google Drive，包括：

**maskrcnn模型链接**：

1. 最终得到的最优模型（42轮训练后，存在轻微过拟合）[^5]
2. 训练30轮得到的模型，mAP指标已较好[^6]

**sparsercnn模型链接**：

1. 最终得到的最优模型（80轮训练后，但仍未完全收敛）[^7]

另外，由于mmdet框架下不好直接使用sparsercnn做分割，找到了一个由sparsercnn改进而来的分割模型**SparseInst**，在detectron2框架下利用coco数据集上训练好的模型权重对部分图片进行了测试仅供参考。模型权重下载链接为[^8]。

[^1]: https://github.com/Lilllllllian/Computer-Vision/tree/main
[^2]: https://drive.google.com/file/d/19O-4N0eaNmykoC1IrbHPbtczzjAqlN4Q/view?usp=sharing
[^3]: https://drive.google.com/file/d/1EjNQ3ztvaKSL9QGn55jHhnzDs-OtBbeN/view?usp=sharing
[^4]: https://github.com/Lilllllllian/Computer-Vision/tree/main
[^5]: https://drive.google.com/file/d/1yVarIo11Qxx7yvuCBnCdcjn_yIe_GAeX/view?usp=sharing
[^6]: https://drive.google.com/file/d/1JAuOrR6ZOMF6eJtIHobFFiiy6yS1KrxY/view?usp=sharing
[^7]: https://drive.google.com/file/d/17YVkl4ALWOhb1SFXdOe-teBhPlhrq_j9/view?usp=sharing
[^8]: https://drive.usercontent.google.com/download?id=1MK8rO3qtA7vN9KVSBdp0VvZHCNq8-bvz&export=download&authuser=0
