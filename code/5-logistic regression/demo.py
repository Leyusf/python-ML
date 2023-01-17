import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# dataset = pd.read_csv('Social_Network_Ads.csv')
# 预测用户是否购买，0是不购买，1是购买
# X = dataset.iloc[:, [2, 3]].values
# Y = dataset.iloc[:, 4].values

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
data['cancer'] = [dataset.target_names[t] for t in dataset.target]
X = dataset.data
Y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 使用sklearn的逻辑回归模型
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print(classifier.score(X_test, y_test))
# 评估预测
# 使用混淆矩阵
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# 可视化
# from matplotlib.colors import ListedColormap
#
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
#
# plt.title('SK LOGISTIC(Training set)')
# plt.xlabel(' Age')
# plt.ylabel(' Estimated Salary')
# plt.legend()
# plt.show()
#
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
#
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
#
# plt.title('SK LOGISTIC(Test set)')
# plt.xlabel(' Age')
# plt.ylabel(' Estimated Salary')
# plt.legend()
# plt.show()

# 使用自己实现的
from LogisticRefressionImpl import MyLogisticRegressor

LR = MyLogisticRegressor(1500, 0.1)
LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)
print(y_pred)
count_right = sum(y_pred == y_test) * 1.0 / len(y_pred)
print(count_right)
