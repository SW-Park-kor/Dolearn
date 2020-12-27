from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


###iris 데이터로 분석
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target

# df = pd.DataFrame(X, columns = iris.feature_names)

# # print("< Iris Data >")
# # print("The number of sample data : " + str(len(df)))
# # print("The number of features of the data : " + str(len(df.columns)))
# # print("The labels of the data : " + str(np.unique(y)))
# # df

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42) # randomstate는 시드값

# # print("The number of train data set : %d " %len(X_train))
# # print("The number of test data set : %d " %len(X_test))

# estimator = KNeighborsClassifier(n_neighbors=3) #모델생성
# estimator.fit(X_train, y_train) #모델학습
# label_predict = estimator.predict(X_test) #test데이터 예측
# # print("The accuracy score of classification: %.9f" %accuracy_score(y_test, label_predict)) #모델 성능평가




#### breast cancer데이터로 분석
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# df = pd.DataFrame(X, columns = breast_cancer.feature_names)
# df
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# print("The number of train data set : %d " %len(X_train))
# print("The number of test data set : %d " %len(X_test))

estimator = KNeighborsClassifier(n_neighbors=5, weights = 'distance') #모델 생성
estimator.fit(X_train, y_train) # 모델 학습
label_predict = estimator.predict(X_test) # 테스트데이터 예측
# print("The accuracy score of classification: %.9f" %accuracy_score(y_test, label_predict))




myList = list(range(1,100))
neighbors = [ x for x in myList if x % 2 != 0]
# print(neighbors)
# print("The number of neighbors k is %d" %len(neighbors))

cv_scores = []
for k in neighbors:
    # print("< k = %d >" %k)
    estimator = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(estimator, X_train, y_train, cv = 10, scoring = 'accuracy')
    # print("The scores of classification are \n" + str(scores))
    cv_scores.append(scores.mean()) # average error 
    # print("The average score of scores is %.9f \n" %scores.mean())



####최적 k값 찾기

MSE = [1 - x for x in cv_scores] #오분류율 : 1-정확도

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Misclassification Error")
plt.show()

# determining best k 
min_MSE = min(MSE) # 최소 오분류비율값
index_of_min_MSE = MSE.index(min_MSE) # 최소값의 인덱스 찾기
optimal_k = neighbors[index_of_min_MSE] # 최적의 k값
print ("The optimal number of neighbors i is %d" % optimal_k)


## 최적의 K값으로 분류
estimator = KNeighborsClassifier(n_neighbors=13)
estimator.fit(X_train, y_train)
label_predict = estimator.predict(X_test)
print("The accuracy score of classification: %.9f"  
      %accuracy_score(y_test, label_predict))
