import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import getLoader
from TransformerModel import TransformerModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 导入tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 获取当前时间戳
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 创建一个SummaryWriter实例，使用时间戳作为日志目录的名称
writer = SummaryWriter(f"runs/train_logs_{timestamp}")


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=20,
):
    model = model.to(device)
    criterion = criterion.to(device)

    best_val_loss = float("inf")  # 用于保存最佳验证损失

    step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        print("Size of the loader: ", len(train_loader))
        for src, tgt in tqdm(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            tgt_input = torch.zeros((tgt.size(0), 1, 1), device=src.device)
            loss = 0
            for t in range(tgt.size(1)):
                output = model(src, tgt_input)
                loss += criterion(output[:, -1, :], tgt[:, t : t + 1])
                tgt_input = torch.cat((tgt_input, output[:, -1:, :]), dim=1)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            writer.add_scalar("Training Loss", loss.item(), step)
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")
        # 验证集上的损失
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                tgt_input = torch.zeros((tgt.size(0), 1, 1), device=src.device)
                loss = 0
                for t in range(tgt.size(1)):
                    output = model(src, tgt_input)
                    loss += criterion(output[:, -1, :], tgt[:, t : t + 1])
                    tgt_input = torch.cat((tgt_input, output[:, -1:, :]), dim=1)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)  # 记录验证损失

        # 如果模型在验证集上表现更好，保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"models/best_model_{timestamp}.pth")
            print("Model Saved")


# 假设getLoader返回的是训练和验证数据加载器
train_loader, val_loader = getLoader()
# 获取一个数据批次
data, target = next(iter(train_loader))
data = data.to(device)
target = target.to(device)


model = TransformerModel(input_dim=15, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)


train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=20,
)

writer.close()  # 关闭SummaryWriter
