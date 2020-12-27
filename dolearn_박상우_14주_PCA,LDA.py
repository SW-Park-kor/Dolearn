#%%
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time
import pandas as pd


# data = datasets.load_breast_cancer()
data = datasets.load_iris()
# print(len(data.feature_names))
# print(data.feature_names)
# print(len(data.target_names))
# print(data.target_names)

x = data.data[:, :2] #첫번째, 두번째 속성 선택
y = data.target
target_names = data.target_names
# target_names

plt.figure(figsize=(10, 10))
colors = ['red', 'blue']

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(x[y == i, 0], x[y == i, 1], color=color, label=target_name)

plt.legend()
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

####PCA 적용
x = data.data
y = data.target
target_names = data.target_names

pca = PCA(n_components=2) # 모듈 생성
x_p = pca.fit(x).transform(x)  # 모델 훈련, 차원축소
# print('가장 큰 주성분 두 개에 대한 분산: %s' % str(pca.explained_variance_ratio_))

plt.figure(figsize=(10, 10))
colors = ['red', 'blue']

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(x_p[y == i, 0], x_p[y == i, 1], color=color, label=target_name)

plt.legend()
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()


####LDA 적용
x = data.data
y = data.target
target_names = data.target_names

lda = LinearDiscriminantAnalysis(solver='eigen', n_components=2)
x_l = lda.fit(x, y).transform(x) # 모델훈련, 차원축소

plt.figure(figsize=(10, 10))
colors = ['red', 'blue']

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(x_l[y == i, 0], x_l[y == i, 1], color=color, label=target_name)

plt.legend()
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.show()






#%%