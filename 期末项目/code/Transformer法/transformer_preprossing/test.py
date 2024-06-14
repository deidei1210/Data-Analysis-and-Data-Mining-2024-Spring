import torch
from dataset import getLoader, getT1Loader, getT2Loader
from TransformerModel import TransformerModel
import torch.nn as nn

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

# 获取数据加载器
train_loader, val_loader = getLoader()

# 在验证集上进行预测并分别计算每个时间点的MAE
model.eval()
total_losses = [0, 0, 0, 0]
counts = [0, 0, 0, 0]

with torch.no_grad():
    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = torch.zeros((tgt.size(0), 1, 1), device=src.device)
        predictions = []
        for _ in range(tgt.size(1)):
            output = model(src, tgt_input)
            next_value = output[:, -1:, :]
            predictions.append(next_value)
            tgt_input = torch.cat((tgt_input, next_value), dim=1)
        predictions = torch.cat(predictions, dim=1)

        # 调整 predictions 的形状
        predictions = predictions.squeeze(-1)  # 从 [32, 4, 1] 变为 [32, 4]

        # 计算每个时间点的MAE
        for t in range(tgt.size(1)):
            loss = criterion(predictions[:, t], tgt[:, t])
            total_losses[t] += loss.item() * tgt.size(0)
            counts[t] += tgt.size(0)

# 平均MAE
avg_maes = [total_losses[t] / counts[t] for t in range(4)]
for t, mae in enumerate(avg_maes):
    print(f"Average MAE at time point {t+1}: {mae}")
