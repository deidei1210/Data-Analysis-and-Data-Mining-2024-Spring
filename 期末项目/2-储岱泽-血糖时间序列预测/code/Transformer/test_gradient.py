import torch
from dataset import getLoader, getT1Loader, getT2Loader
from TransformerModel import TransformerModel
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model = TransformerModel(input_dim=15, output_dim=1)
model.load_state_dict(
    torch.load("models/best_model_20240614-113015.pth")
)  # 将 <your_timestamp> 替换为实际的时间戳
model = model.to(device)

# 定义损失函数
criterion = nn.L1Loss()  # 使用MAE
mse_criterion = nn.MSELoss()  # 使用MSE

# 获取数据加载器
train_loader, val_loader = getLoader()


# 计算梯度贡献度
def calculate_gradients(model, inputs, targets):
    model.eval()
    inputs = inputs.to(device).requires_grad_(True)
    targets = targets.to(device)

    outputs = model(inputs, torch.zeros((inputs.size(0), 1, 1), device=device))
    loss = criterion(outputs[:, -1, :], targets[:, -1])
    model.zero_grad()
    loss.backward()

    gradients = inputs.grad.data.cpu().numpy()
    return gradients


# 初始化存储梯度贡献度的数组
gradient_contributions = None

# 计算验证集上的梯度贡献度
with torch.no_grad():
    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        gradients = calculate_gradients(model, src, tgt)

        if gradient_contributions is None:
            gradient_contributions = gradients
        else:
            gradient_contributions += gradients

# 平均梯度贡献度
gradient_contributions /= len(val_loader)

# 可视化梯度贡献度
time_series_features = [
    "CGM (mg / dl)",
    "Insulin dose - s.c.",
    "CSII - bolus insulin (Novolin R, IU)",
    "Carbohydrate/g",
]
static_features = [
    "type",
    "patient_id",
    "Age (years)",
    "Weight (kg)",
    "BMI (kg/m2)",
    "Duration of Diabetes (years)",
    "HbA1c (mmol/mol)",
    "Fasting Plasma Glucose (mg/dl)",
    "2-hour Postprandial C-peptide (nmol/L)",
    "Fasting C-peptide (nmol/L)",
    "Glycated Albumin (%)",
    "Acute Diabetic Complications",
    "Diabetic Macrovascular Complications",
    "Diabetic Microvascular Complications",
    "Comorbidities",
    "Hypoglycemic Agents",
    "Other Agents",
]

# 对时间序列特征和静态特征分别计算梯度贡献度的均值
avg_time_series_gradients = np.mean(gradient_contributions, axis=(0, 2))
avg_static_gradients = np.mean(gradient_contributions, axis=(0, 1))

# 绘制时间序列特征的梯度贡献度
plt.figure(figsize=(12, 6))
plt.bar(range(len(time_series_features)), avg_time_series_gradients, align="center")
plt.xticks(range(len(time_series_features)), time_series_features, rotation=90)
plt.xlabel("时间序列特征")
plt.ylabel("梯度贡献度")
plt.title("时间序列特征的梯度贡献度")
plt.show()

# 绘制静态特征的梯度贡献度
plt.figure(figsize=(12, 6))
plt.bar(range(len(static_features)), avg_static_gradients, align="center")
plt.xticks(range(len(static_features)), static_features, rotation=90)
plt.xlabel("静态特征")
plt.ylabel("梯度贡献度")
plt.title("静态特征的梯度贡献度")
plt.show()
