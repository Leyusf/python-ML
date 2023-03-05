import random
import numpy as np


class KMeans:
    def __init__(self, k, e=1e-4):
        self.data = None
        self.capacity = None
        self.k = k
        # 保存聚类中心的索引和类样本的索引
        self.centers = []
        self.clusters = []
        self.e = e

    def fit(self, X):
        self.data = X
        self.capacity = len(X)
        self.__pick_start_point()
        changed = True
        while changed:
            self.clusters = []
            for i in range(self.k):
                self.clusters.append([])
            for i in range(self.capacity):
                min_distance = np.inf
                center = -1
                # 寻找簇
                for j in range(self.k):
                    distance = self.__distance(self.data[i], self.centers[j])
                    if min_distance > distance:
                        min_distance = distance
                        center = j
                # 加入簇
                self.clusters[center].append(self.data[i])
            newCenters = []
            for cluster in self.clusters:
                newCenters.append(self.__calCenter(cluster).tolist())
            # if (np.array(newCenters) == self.centers).all():
            #     changed = False
            if np.linalg.norm(np.array(newCenters) - self.centers) < self.e:
                changed = False
            else:
                self.centers = np.array(newCenters)

    # 随机选择初始的簇心
    def __pick_start_point(self):
        # 随机确定初始簇心
        samples = random.sample([i for i in range(len(self.data))], self.k)
        for sample in samples:
            self.centers.append(self.data[sample])

    def __distance(self, data, center):
        diff = data - center
        return np.sum(np.square(diff), axis=-1)

    def __calCenter(self, cluster):
        # 计算该簇的中心
        cluster = np.array(cluster)
        if cluster.shape[0] == 0:
            return False
        return np.mean(cluster.T, axis=-1)

    def predict(self, X):
        pred = []
        for i in range(len(X)):
            min_dis = np.inf
            c = -1
            for j in range(self.k):
                dis = self.__distance(X[i], self.centers[j])
                if dis < min_dis:
                    min_dis = dis
                    c = j
            pred.append(c)
        return pred





