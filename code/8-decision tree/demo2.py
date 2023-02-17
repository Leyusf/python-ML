import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from DecisionTree import DecisionTree

iris = load_wine()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
pred1 = classifier.predict(X_test)

myTree = DecisionTree(std="gain_ratio")
myTree.fit(X_train, y_train)
pred2 = myTree.predict(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
pred3 = tree.predict(X_test)


print(pred1)
print(accuracy_score(y_test, pred1))
print(pred2)
print(accuracy_score(y_test, pred2))
print(pred3)
print(accuracy_score(y_test, pred3))
