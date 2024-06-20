import numpy as np  # 导入numpy库，用于矩阵运算和处理
import matplotlib.pyplot as plt

class DBSCAN:
    # 初始化函数，设置eps邻域半径和minPts最小点数
    def __init__(self, eps, min_pts):
        self.eps = eps  # 邻域半径，两个点成为邻居的最大距离
        self.min_pts = min_pts  # 一个点成为“核心点”所需的最小邻居数目

    # 主函数，用于拟合数据并预测每个点的聚类标签
    def fit_predict(self, X):
        labels = [0] * len(X)  # 初始化所有点的标签为0
        cluster_id = 0  # 初始化聚类ID

        # 对每个点进行迭代
        for i in range(len(X)):
            if labels[i] != 0:  # 如果点已经被标记，则跳过
                continue

            # 获取点i的邻域点
            neighbors = self.region_query(X, i)
            if len(neighbors) < self.min_pts:  # 如果邻域点数少于minPts，则为噪声点
                labels[i] = -1  # 标记为-1
            else:  # 否则，将该点作为新聚类的核心点
                cluster_id += 1  # 聚类ID自增
                self.expand_cluster(X, labels, i, neighbors, cluster_id)  # 扩展该核心点的聚类

        return labels  # 返回所有点的聚类标签

    # 递归函数，用于扩展以核心点为核心的聚类
    def expand_cluster(self, X, labels, core_idx, neighbors, cluster_id):
        labels[core_idx] = cluster_id  # 将核心点标记为当前聚类ID

        i = 0  # 初始化索引
        # 当存在待处理的邻居点时
        while i < len(neighbors):
            idx = neighbors[i]  # 取出当前邻居点的索引
            if labels[idx] == -1:  # 如果邻居是噪声点，则将其标记为当前聚类ID
                labels[idx] = cluster_id
            elif labels[idx] == 0:  # 如果邻居尚未分配到任何聚类
                labels[idx] = cluster_id  # 将其标记为当前聚类ID
                new_neighbors = self.region_query(X, idx)  # 获取邻居点的邻域点
                if len(new_neighbors) >= self.min_pts:  # 如果邻居点的邻域点数足够
                    neighbors.extend(new_neighbors)  # 将这些邻域点添加到待处理列表
            i += 1  # 移动到下一个邻居点

    # 寻找给定点的邻域点
    def region_query(self, X, idx):
        neighbors = []  # 初始化邻域点列表
        for i in range(len(X)):  # 遍历所有点
            if np.linalg.norm(X[idx] - X[i]) < self.eps:  # 如果两点之间的欧氏距离小于eps
                neighbors.append(i)  # 将点i添加到邻域点列表
        return neighbors  # 返回邻域点列表

# 示例用法
if __name__ == "__main__":
    # 创建一个示例数据集
    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

    # 初始化DBSCAN模型并拟合数据
    dbscan = DBSCAN(eps=3, min_pts=2)
    labels = dbscan.fit_predict(X)

    # 打印聚类结果
    print("Cluster labels:", labels)

    # 提取每个聚类的点
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(X[i])

    # 绘制聚类结果
    plt.figure(figsize=(8, 6))
    for label, points in clusters.items():
        if label == -1:
            plt.scatter([p[0] for p in points], [p[1] for p in points], c='gray', label='Noise')
        else:
            plt.scatter([p[0] for p in points], [p[1] for p in points], label=f'Cluster {label}')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering Result')
    plt.legend()
    plt.show()