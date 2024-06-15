# README

## 项目简介

本项目旨在使用长短期记忆网络（LSTM）预测糖尿病患者的血糖水平。我们通过训练一个多层LSTM模型来捕捉时间序列数据中的长时间依赖关系，并通过模型预测未来的血糖水平。

## 环境配置

### 硬件要求

- CPU 或 GPU（推荐使用支持CUDA的NVIDIA GPU以加快训练速度）
- 至少16GB内存

### 软件要求

- 操作系统：Windows, macOS, 或 Linux
- Python 版本：3.8或以上

### 依赖库

- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- OpenPyXL (用于处理Excel文件)

### 安装依赖

可以使用 `pip` 安装所有依赖库：

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib openpyxl
```

### 文件夹结构

project_root/
├── LSTM.py
├── predict.py
├── blood_glucose_prediction_model.h5
├── datasets_new/
│   ├── summary/
│   │   ├── Shanghai_T1DM_Summary0.xlsx
│   │   └── Shanghai_T2DM_Summary0.xlsx
│   ├── Shanghai_T1DM/
│   │   └── <患者数据文件1>.csv
│   │   └── <患者数据文件2>.csv
│   │   └── ...
│   └── Shanghai_T2DM/
│       └── <患者数据文件1>.csv
│       └── <患者数据文件2>.csv
│       └── ...


### 运行步骤

```bash
python LSTM.py
python predict.py
```