import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from DecisionTree import DecisionTree

dataset = pd.read_csv("car.csv")
dataset.replace('?', "nan")
# 去掉缺失值
dataset.dropna()

le = LabelEncoder()
# 将非数字标签替换为 0,1,2
for i in dataset.columns:
    dataset[i] = le.fit_transform(dataset[i])

X = dataset[dataset.columns[:-1]].values
y = dataset[dataset.columns[-1]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

myTree = DecisionTree(std="gain_ratio")
myTree.fit(X_train, y_train)
pred1 = myTree.predict(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
pred2 = tree.predict(X_test)

print(y_test)
print(pred1)
print(accuracy_score(y_test, pred1))
print(pred2)
print(accuracy_score(y_test, pred2))
