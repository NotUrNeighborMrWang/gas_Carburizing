# Improved design method for gas carburizing process through data-driven and physical information

## 📑 项目简介

本项目对应论文 **《Improved design method for gas carburizing process through data-driven and physical information》**，针对传统气体渗碳工艺设计依赖试错、数值模拟计算成本高、纯数据驱动模型泛化性差等痛点，提出数据与物理信息双驱动的仿真方案，将物理定律嵌入深度学习框架，解决渗碳过程中碳元素扩散仿真的精度与效率平衡难题。

**核心创新点**：融合菲克第二定律构建PINN模型，打破神经网络对海量数据的依赖；相比传统数值求解器，计算速度提升数个数量级，碳浓度预测偏差仅0.008%。

## ✨ 核心功能

- **碳浓度分布精准预测**：仿真强渗、扩散阶段的碳元素迁移规律，输出沿层深的碳浓度分布曲线

- **渗碳工艺参数优化**：智能求解最优强渗时间、扩散时间与强渗扩散比，适配20CrMo、Cr13系列等典型渗碳钢

- **超快仿真推理**：模型训练完成后，单样本推理时间仅毫秒级，远优于Deform等传统数值仿真软件

- **结果可视化**：自动生成碳浓度分布、硬度梯度对比图，直观验证仿真精度

- **低数据量适配**：物理约束加持，小样本数据即可实现高精度训练，降低工业场景数据获取门槛

## 🔧 环境依赖

本项目基于Python深度学习框架开发，需安装以下依赖库：

```bash
# 基础科学计算库
pip install numpy pandas matplotlib

# 深度学习框架
pip install torch torchvision

# PINN专用求解库
pip install deepxde

# 其他辅助库
pip install scipy opencv-python

```

## 📊 实验验证结果

以典型渗碳钢**20CrMo**为验证对象，目标有效渗碳层深度1.0±0.1mm、表面碳浓度0.70%：

- 最优工艺：强渗时间172min，扩散时间98min，强渗扩散比1.75:1

- 实测有效渗碳层深度0.9mm，偏差处于误差允许范围内

- 表面最高硬度695.3HV，符合工业渗碳质量标准

- 计算效率：推理时间较Deform软件提升约6个数量级

## 🔍 关键参数说明

|参数名称|含义|默认值/推荐值|
|---|---|---|
|hidden_layers|PINN隐藏层层数|3（精度与算力最优平衡）|
|neurons_per_layer|每层神经元数量|100|
|activation|激活函数|tanh|
|optimizer|优化器|L-BFGS|
|target_c_surface|目标表面碳浓度|0.70%|

## 📄 引用说明

如果本项目对您的研究有帮助，请引用相关论文：

```bibtex
@article{Wang2025Carburizing,
  title={Improved design method for gas carburizing process through data-driven and physical information},
  author={Xuefei Wang, Chunyang Luo, Di Jiang, Haojie Wang, Zhaodong Wang},
  journal={Computational Materials Science},
  volume={247},
  pages={113507},
  year={2025},
  doi={10.1016/j.commatsci.2024.113507}
}

```
