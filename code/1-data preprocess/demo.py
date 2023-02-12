import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from MultiClassifierImpl import MyMultiClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(y_train)
y_train = label_binarizer.transform(y_train)
y_test = label_binarizer.transform(y_test)

classifier = MyMultiClassifier(10000, penalty="l2", alg="adadelta", lambda_=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred_res = np.zeros(len(y_pred))
for i in range(len(y_pred)):
    y_pred_res[i] = np.argmax(y_pred[i])
y_test_res = np.zeros(len(y_pred))
for i in range(len(y_pred)):
    y_test_res[i] = np.argmax(y_test[i])
print(y_pred_res)
print(y_test_res)
count_right = sum(y_pred_res == y_test_res) * 1.0 / len(y_pred)
print(count_right)

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression(penalty="l2", C=0.5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
print(classifier.score(X_test, y_test))
