
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


iris = load_iris()

x_train = iris.data[:-30] #12개는 train, 30개는 test
y_train = iris.target[:-30]

x_test = iris.data[-30:]
y_test = iris.target[-30:]

rfc = RandomForestClassifier(n_estimators=10)

rfc.fit(x_train, y_train)
prediction = rfc.predict(x_test)
# print(prediction==y_test)

# print(rfc.score(x_test, y_test)) #테스트 데이터에 대한 예측결과 정확도

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# print("Accuracy is :", accuracy_score(prediction, y_test))
# print("=================================================")
# print(classification_report(prediction, y_test))

from sklearn.model_selection import train_test_split

x = iris.data
y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2)
# print(y_test) #스플릿하지 않은 데이터
# print(Y_test) #랜덤으로 스플릿해서 뽑은 데이터

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, Y_train)
prediction_1 = rfc.predict(X_test)

# print("Accuracy is :", accuracy_score(prediction_1, Y_test))
# print("=================================================")
# print(classification_report(prediction_1, Y_test))

clf_2 = RandomForestClassifier(n_estimators=10, max_features=4, oob_score=True)
clf_2.fit(X_train, Y_train)
prediction_2 = rfc.predict(X_test)

# print(prediction_2==Y_test)
# print("Accuracy is :", accuracy_score(prediction_2, Y_test))
# print("=================================================")
# print(classification_report(prediction_2, Y_test))
#뭔가 이상함. 실습보면서 확인. 아닐수도있음

for feature, imp in zip(iris.feature_names, clf_2.feature_importances_):
    print(feature,imp)
