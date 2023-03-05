import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans as stdKM
from sklearn.datasets import load_iris
import KMeans


def plotKmeans(pred, title):
    xdata = []
    ydata = []
    for Cluster in pred:
        xsubdata = []
        ysubdata = []
        for point in Cluster:
            xsubdata.append(point[0])
            ysubdata.append(point[1])
        xdata.append(xsubdata)
        ydata.append(ysubdata)

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(xdata)):
        for j in range(len(xdata[i])):
            x = np.array([xdata[i][j]])
            y = np.array([ydata[i][j]])
            plt.plot(x, y,
                     color=colors[i],  # 全部点设置为红色
                     marker='o',  # 点的形状为圆点
                     ms=7,
                     linestyle='-')
    plt.grid(True)
    plt.title(title)
    plt.show()


def findClass(k, pred, data):
    clusters = [[] for i in range(k)]
    for i in range(len(pred)):
        clusters[pred[i]].append(data[i])
    return clusters


iris = load_iris()
X = iris.data  # data
Y = iris.target  # label
Km = KMeans.KMeans(3)
Km.fit(X)
pred1 = findClass(3, Km.predict(X), X)
plotKmeans(pred1, "my")

kmeans = stdKM(3)
kmeans.fit(X)
pred2 = findClass(3, kmeans.predict(X), X)
plotKmeans(pred2, "sk")
print("Done")