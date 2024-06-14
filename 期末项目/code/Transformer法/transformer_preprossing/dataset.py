import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


class CGMDataset(Dataset):
    def __init__(self, file_paths=None, input_window_size=4, output_window_size=4):
        self.data = []
        self.targets = []
        if file_paths:
            for file_path in file_paths:
                X, y = self.process_file(
                    file_path, input_window_size, output_window_size
                )
                self.data.append(X)
                self.targets.append(y)

            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)

            # 检查X_mixed中是否有NaN值
        if np.isnan(self.data).any():
            print("X_mixed contains NaN values")

        # 检查y_mixed中是否有NaN值
        if np.isnan(self.targets).any():
            print("y_mixed contains NaN values")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            self.targets[idx], dtype=torch.float32
        )

    @staticmethod
    def process_file(file_path, input_window_size=4, output_window_size=4):
        data = pd.read_csv(file_path)

        # 删除包含NaN值的行
        data = data.dropna()

        # 获取所有特征，删除Date列
        features = data.drop(columns=["Date"]).values
        targets = data["CGM (mg / dl)"].values

        X = []
        y = []

        for i in range(len(data) - input_window_size - output_window_size):
            X.append(features[i : i + input_window_size])  # 保持时间窗口内的特征维度
            y.append(
                targets[
                    i + input_window_size : i + input_window_size + output_window_size
                ]
            )  # 目标是下四个时间点的血糖值

        return np.array(X), np.array(y)

    @staticmethod
    def upsample(X, y, factor):
        indices = np.random.choice(len(X), size=int(len(X) * factor), replace=True)
        return X[indices], y[indices]

    @staticmethod
    def downsample(X, y, factor):
        indices = np.random.choice(len(X), size=int(len(X) * factor), replace=False)
        return X[indices], y[indices]


def getLoader():
    # 文件夹路径
    directory_paths = {"T1DM": "dataset/T1DM", "T2DM": "dataset/T2DM"}

    # 获取文件路径
    file_paths_T1DM = [
        os.path.join(directory_paths["T1DM"], filename)
        for filename in os.listdir(directory_paths["T1DM"])
        if filename.endswith(".csv")
    ]
    file_paths_T2DM = [
        os.path.join(directory_paths["T2DM"], filename)
        for filename in os.listdir(directory_paths["T2DM"])
        if filename.endswith(".csv")
    ]

    # 创建单独的数据集，不进行采样处理
    input_window_size = 8  # 可调节的输入窗口
    output_window_size = 4  # 固定的输出窗口

    dataset_T1DM = CGMDataset(
        file_paths_T1DM,
        input_window_size=input_window_size,
        output_window_size=output_window_size,
    )
    dataset_T2DM = CGMDataset(
        file_paths_T2DM,
        input_window_size=input_window_size,
        output_window_size=output_window_size,
    )

    # 对T1DM和T2DM分别进行上采样和下采样
    upsample_factor_T1DM = 2.5
    downsample_factor_T2DM = 0.4

    X_T1DM, y_T1DM = dataset_T1DM.data, dataset_T1DM.targets
    X_T2DM, y_T2DM = dataset_T2DM.data, dataset_T2DM.targets

    # 检查X_mixed中是否有NaN值
    if np.isnan(X_T1DM).any():
        print("X_mixed contains NaN values")

    # 检查y_mixed中是否有NaN值
    if np.isnan(X_T2DM).any():
        print("y_mixed contains NaN values")

    X_T1DM, y_T1DM = CGMDataset.upsample(X_T1DM, y_T1DM, upsample_factor_T1DM)
    X_T2DM, y_T2DM = CGMDataset.downsample(X_T2DM, y_T2DM, downsample_factor_T2DM)

    # 检查X_mixed中是否有NaN值
    if np.isnan(X_T1DM).any():
        print("X_mixed contains NaN values")

    # 检查y_mixed中是否有NaN值
    if np.isnan(X_T2DM).any():
        print("y_mixed contains NaN values")

    # 混合数据集
    X_mixed = np.concatenate([X_T1DM, X_T2DM], axis=0)
    y_mixed = np.concatenate([y_T1DM, y_T2DM], axis=0)

    # 检查X_mixed中是否有NaN值
    if np.isnan(X_mixed).any():
        print("X_mixed contains NaN values")

    # 检查y_mixed中是否有NaN值
    if np.isnan(y_mixed).any():
        print("y_mixed contains NaN values")

    # 创建StandardScaler对象
    scaler_X = StandardScaler()

    # 将三维数据转换为二维
    X_mixed_2D = X_mixed.reshape(-1, X_mixed.shape[-1])

    # 使用X_mixed来拟合scaler，然后对X_mixed进行转换
    X_mixed_2D = scaler_X.fit_transform(X_mixed_2D)

    # 将二维数据转回三维
    X_mixed = X_mixed_2D.reshape(X_mixed.shape)

    dataset_mixed = CGMDataset()
    dataset_mixed.data = X_mixed
    dataset_mixed.targets = y_mixed

    # 将数据集划分为训练集和验证集
    train_size = int(0.9 * len(dataset_mixed))
    val_size = len(dataset_mixed) - train_size
    train_dataset, val_dataset = random_split(dataset_mixed, [train_size, val_size])

    print(f"Mixed dataset size: {len(dataset_mixed)}")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def getT1Loader():
    # 文件夹路径
    directory_paths = {"T1DM": "dataset/T1DM", "T2DM": "dataset/T2DM"}

    # 获取文件路径
    file_paths_T1DM = [
        os.path.join(directory_paths["T1DM"], filename)
        for filename in os.listdir(directory_paths["T1DM"])
        if filename.endswith(".csv")
    ]

    # 创建单独的数据集，不进行采样处理
    input_window_size = 4  # 可调节的输入窗口
    output_window_size = 4  # 固定的输出窗口

    dataset_T1DM = CGMDataset(
        file_paths_T1DM,
        input_window_size=input_window_size,
        output_window_size=output_window_size,
    )

    # 创建StandardScaler对象
    scaler_X = StandardScaler()

    # 将三维数据转换为二维
    X_2D = dataset_T1DM.data.reshape(-1, dataset_T1DM.data.shape[-1])

    # 使用X_mixed来拟合scaler，然后对X_mixed进行转换
    X_2D = scaler_X.fit_transform(X_2D)

    # 将二维数据转回三维
    dataset_T1DM.data = X_2D.reshape(dataset_T1DM.data.shape)

    X_T1DM, y_T1DM = dataset_T1DM.data, dataset_T1DM.targets

    # 检查X_mixed中是否有NaN值
    if np.isnan(X_T1DM).any():
        print("X_mixed contains NaN values")

    # 检查y_mixed中是否有NaN值
    if np.isnan(y_T1DM).any():
        print("y_mixed contains NaN values")

    # 将数据集划分为训练集和验证集
    train_size = int(0.9 * len(dataset_T1DM))
    val_size = len(dataset_T1DM) - train_size
    train_dataset, val_dataset = random_split(dataset_T1DM, [train_size, val_size])

    print(f"Mixed dataset size: {len(dataset_T1DM)}")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def getT2Loader():
    # 文件夹路径
    directory_paths = {"T1DM": "dataset/T1DM", "T2DM": "dataset/T2DM"}

    # 获取文件路径
    file_paths_T2DM = [
        os.path.join(directory_paths["T2DM"], filename)
        for filename in os.listdir(directory_paths["T2DM"])
        if filename.endswith(".csv")
    ]

    # 创建单独的数据集，不进行采样处理
    input_window_size = 4  # 可调节的输入窗口
    output_window_size = 4  # 固定的输出窗口

    dataset_T2DM = CGMDataset(
        file_paths_T2DM,
        input_window_size=input_window_size,
        output_window_size=output_window_size,
    )

    # 创建StandardScaler对象
    scaler_X = StandardScaler()

    # 将三维数据转换为二维
    X_2D = dataset_T2DM.data.reshape(-1, dataset_T2DM.data.shape[-1])

    # 使用X_mixed来拟合scaler，然后对X_mixed进行转换
    X_2D = scaler_X.fit_transform(X_2D)

    # 将二维数据转回三维
    dataset_T2DM.data = X_2D.reshape(dataset_T2DM.data.shape)

    X_T1DM, y_T1DM = dataset_T2DM.data, dataset_T2DM.targets

    # 检查X_mixed中是否有NaN值
    if np.isnan(X_T1DM).any():
        print("X_mixed contains NaN values")

    # 检查y_mixed中是否有NaN值
    if np.isnan(y_T1DM).any():
        print("y_mixed contains NaN values")

    # 将数据集划分为训练集和验证集
    train_size = int(0.9 * len(dataset_T2DM))
    val_size = len(dataset_T2DM) - train_size
    train_dataset, val_dataset = random_split(dataset_T2DM, [train_size, val_size])

    print(f"Mixed dataset size: {len(dataset_T2DM)}")

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


# if __name__ == "__main__":


#     # 打印数据集大小
#     print(f"T1DM dataset size: {len(dataset_T1DM)}")
#     print(f"T2DM dataset size: {len(dataset_T2DM)}")
#     print(f"Mixed dataset size: {len(dataset_mixed)}")

#     # 打印混合数据集的一条数据
#     sample_X, sample_y = dataset_T1DM[0]
#     print(f"Sample X: {sample_X}")
#     print(f"Sample y: {sample_y}")

#     # 计算并显示各数据集的比重
#     total_data_count = len(dataset_mixed)
#     t1dm_proportion = len(X_T1DM) / total_data_count
#     t2dm_proportion = len(X_T2DM) / total_data_count
#     print(f"T1DM data proportion: {t1dm_proportion:.2%}")
#     print(f"T2DM data proportion: {t2dm_proportion:.2%}")
