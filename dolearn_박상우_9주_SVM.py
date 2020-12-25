#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
# print(iris.keys())
# print(iris.data.shape)
# print(iris.feature_names)

# print(iris.DESCR) #describe

x = iris.data[:, :2]
y = iris.target
SVM = svm.SVC(kernel='linear', C=1).fit(x, y)  #SVM모델 학습

x_min, x_max = x[:, 0].min()-1, x[:, 0].max()+1
y_min, y_max = x[:, 1].min()-1, x[:, 1].max()+1

plot_unit = 0.025 # (x_max/x_min)/100
# arrange로 plot unit값만큼 균등하게 간격을 둔 1차원 배열 형태의 데이터를 만든 후, meshgrid로 2차원 배열 형태로 변환
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_unit), np.arange(y_min, y_max, plot_unit)) # 모눈종이생성

# z = SVM.predict(np.c_[xx.ravel(), yy.ravel()]) # xx,yy데이터로 하나의 배열을 만든 후 c_로 열을 추가하고 SVM모델로 분류한 값을 z에 저장
# z = z.reshape(xx.shape) # z의 차원을 xx와 같은 차원으로 재형성
# plt.pcolormesh(xx, yy, z, alpha=0.1) # 3개의 데이터를 그래프로 표현, z값에 따라 색상을 다르게 적용, alpha:투명도
# plt.scatter(x[:, 0],x[:, 1],c=y) # x축, y축 = x[:, 0],x[:, 1] // 산점도, c값에 따라 색상을 다르게 적용
# plt.xlabel('Sepal length') 
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.title('Support Vector Machine')
# plt.show()
# print('정확도 : ',SVM.score(X = x, y = y))




# SVM = svm.SVC(kernel='rbf', C=1).fit(x, y)
# z = SVM.predict(np.c_[xx.ravel(), yy.ravel()])
# z = z.reshape(xx.shape)
# plt.pcolormesh(xx, yy, z,alpha=0.1)
# plt.scatter(x[:, 0],x[:, 1],c=y)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.title('Support Vector Machine')
# plt.show()
# print('정확도 : ',SVM.score(X = x, y = y))





# SVM = svm.SVC(kernel='rbf', C=1, gamma=10).fit(x, y) # gamma가 높아지면 overfitting
# z = SVM.predict(np.c_[xx.ravel(), yy.ravel()])
# z = z.reshape(xx.shape)
# plt.pcolormesh(xx, yy, z, alpha=0.1)
# plt.scatter(x[:, 0],x[:, 1],c=y)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
# plt.xlim(xx.min(), xx.max())
# plt.title('Support Vector Machine')
# plt.show()
# print('정확도 : ',SVM.score(X = x, y = y))

SVM = svm.SVC(kernel='rbf', C=100).fit(x, y) #C값이 높아지면 overfitting
z = SVM.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.pcolormesh(xx, yy, z,alpha=0.1)
plt.scatter(x[:, 0],x[:, 1],c=y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Machine')
plt.show()
print('정확도 : ',SVM.score(X = x, y = y))


#%%