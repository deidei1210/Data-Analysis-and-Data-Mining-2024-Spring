import numpy as np
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters  # 期望分成的簇的个数
        self.linkage = linkage        # 链接方式选择
        self.cluster_points = []  # 初始化簇
        self.distances_dict = {}  # 初始化距离矩阵字典
    
    def fit(self, X):
        n_samples, _ = X.shape
        self.labels_ = np.zeros(n_samples)
        
        distances = np.zeros((n_samples, n_samples))
        self.cluster_points = [[i] for i in range(n_samples)]  # 每个点为一个簇
        # 初始化簇和距离矩阵字典
        # print(self.cluster_points)

        #计算距离矩阵字典
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i][j] = self.calculate_distance(X[i], X[j])
        # print(distances)

        #进行聚类
        for _ in range(n_samples - self.n_clusters):

            #先寻找距离最短的两个簇
            min_distance = np.inf
            for b in range(n_samples):
                for c in range(b+1, n_samples):
                    if distances[b][c] < min_distance:
                        min_distance = distances[b][c]
                        min_i = b  #将距离最短的这两个簇给记下来
                        min_j = c
            # 先对存储簇的ID的数组给处理了
            self.cluster_points[min_i].extend(self.cluster_points[min_j])  # 将第min_j行合并到上面的min_i
            del self.cluster_points[min_j]  # 然后将min_j行删掉

            # print(self.cluster_points)
            
            # 更新距离字典
            for i in range(n_samples):
                if i != min_i and i != min_j:
                    if self.linkage == 'single':
                        distances[min_i, i] = min(distances[min_i, i], distances[min_j, i])
                    elif self.linkage == 'complete':
                        distances[min_i, i] = max(distances[min_i, i], distances[min_j, i])
                    elif self.linkage == 'average':
                        distances[min_i, i] = (distances[min_i, i] + distances[min_j, i]) / 2            
            # print(distances)

            distances = np.delete(distances, min_j, axis=0)
            distances = np.delete(distances, min_j, axis=1)
            # print(distances)

            n_samples -= 1  # 更新样本数量
        
        # 根据簇的合并情况得到最终的标签
        self.labels_ = self.get_labels(self.distances_dict)

        return self.cluster_points

    # 运用距离公式计算距离   
    def calculate_distance(self, x1, x2):
        if self.linkage == 'single':
            return np.linalg.norm(x1 - x2, ord=1)  # 曼哈顿距离
        elif self.linkage == 'complete':
            return np.linalg.norm(x1 - x2, ord=np.inf)  # 切比雪夫距离
        elif self.linkage == 'average':
            return np.linalg.norm(x1 - x2)  #欧氏距离
    
    def get_labels(self, distances_dict):
        n_samples = len(distances_dict)
        labels = np.zeros(n_samples)
        current_label = 0
        cluster_dict = {}
        
        for i, distances in distances_dict.items():
            if i not in cluster_dict:
                cluster_dict[i] = current_label
                current_label += 1
            
            for j, dist in distances.items():
                if dist == 0:
                    if j not in cluster_dict:
                        cluster_dict[j] = cluster_dict[i]
                    else:
                        for k, v in cluster_dict.items():
                            if v == cluster_dict[j]:
                                cluster_dict[k] = cluster_dict[i]
                    break
        
        for i in range(n_samples):
            labels[i] = cluster_dict[i]

        return labels.astype(int)

    # 聚类结果可视化
    def plot_clusters(self,X):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        fig, ax = plt.subplots()
    
        for i, cluster in enumerate(self.cluster_points):
            cluster_color = colors[i % len(colors)]
            cluster_data = X[cluster]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], c=cluster_color, label=f'Cluster {i}')

        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Clustered Data')
        plt.show()

# 测试算法
X_t = np.array([[1, 2], [1, 3], [2, 2], [8, 7], [8, 8], [7, 7]])
model_single = HierarchicalClustering(n_clusters=2, linkage='single')
model_complete = HierarchicalClustering(n_clusters=2, linkage='complete')
model_average = HierarchicalClustering(n_clusters=2, linkage='average')

cluster1 = model_single.fit(X_t)
cluster2 = model_complete.fit(X_t)
cluster3 = model_average.fit(X_t)

print(cluster1)
print(cluster2)
print(cluster3)

model_single.plot_clusters(X_t)
model_complete.plot_clusters(X_t)
model_average.plot_clusters(X_t)
