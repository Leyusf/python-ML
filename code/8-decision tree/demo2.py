from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from DecisionTree import DecisionTree
from RandomForest import RandomForest

iris = load_wine()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

myTree = DecisionTree(std="gain_ratio")
myTree.fit(X_train, y_train)
pred1 = myTree.predict(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
pred2 = tree.predict(X_test)

myForest = RandomForest()
myForest.fit(X_train, y_train)
pred3 = tree.predict(X_test)

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
pred4 = tree.predict(X_test)

print(pred1)
print(accuracy_score(y_test, pred1))
print(pred2)
print(accuracy_score(y_test, pred2))
print(pred3)
print(accuracy_score(y_test, pred3))
print(pred4)
print(accuracy_score(y_test, pred4))
