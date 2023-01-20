from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from QDA import QDA
from LDA import LDA

# QDA二分类
data = load_breast_cancer()
X = data["data"]
y = data["target"]
qda = QDA()
qda.fit(X, y)
y_pred = qda.predict(X)
print(y_pred)
count_right = sum(y_pred == y) * 1.0 / len(y_pred)
print(count_right)

# QDA多分类
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.25, random_state=0)
qda = QDA()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)
print(y_pred)
count_right = sum(y_test == y_pred) * 1.0 / len(y_pred)
print(count_right)

# scikit的QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

sqda = QuadraticDiscriminantAnalysis()
sqda.fit(X_train, y_train)
y_pred = qda.predict(X_test)
print(y_pred)
count_right = sum(y_test == y_pred) * 1.0 / len(y_pred)
print(count_right)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# LDA多分类
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.25, random_state=0)
lda = LDA()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(y_pred)
count_right = sum(y_pred == y_test) * 1.0 / len(y_pred)
print(count_right)

# scikit的LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

slda = LinearDiscriminantAnalysis()
slda.fit(X_train, y_train)
y_pred = slda.predict(X_test)
print(y_pred)
count_right = sum(y_pred == y_test) * 1.0 / len(y_pred)
print(count_right)

# NaiveBayes 多分类
from NaiveBayes import NaiveBayes

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.25, random_state=0)
nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(y_pred)
count_right = sum(y_pred == y_test) * 1.0 / len(y_pred)
print(count_right)

# scikit的 NB
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(y_pred)
count_right = sum(y_pred == y_test) * 1.0 / len(y_pred)
print(count_right)
