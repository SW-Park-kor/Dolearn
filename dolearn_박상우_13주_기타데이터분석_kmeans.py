#%%
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0) # 0을 시드값으로 랜덤데이터 생성

centers = [[1, 1], [0, 0], [2, -1]]
data, labels_true = make_blobs(n_samples = 2000, centers = centers, cluster_std = 0.7)

# print(data)
# print()
# print(labels_true)
# print(np.unique(labels_true)) #중복된값 제외하고 출력 : 클러스터 확인가능


# 데이터 2차원 평면에 산점도로 나타내기
# plt.figure(figsize=(15,10))
# plt.scatter(data[:,0], data[:,1])


estimator = KMeans(init = 'k-means++', n_clusters = 3, n_init = 10)
estimator.fit(data) # labels_ 라는 클래스 변수에 클러스터 결과 저장

labels_predict = estimator.labels_
np.unique(labels_predict) # 중복제거

cm = plt.cm.get_cmap('jet') # color map
scaled_labels = (labels_predict - np.min(labels_predict)) 
scaled_labels = scaled_labels /(np.max(labels_predict) - np.min(labels_predict))
np.unique(scaled_labels)

plt.figure(figsize=(15,10))
plt.scatter(data[:,0], data[:,1], c = cm(scaled_labels)) # 클라스터 라벨값에 따른 RGBA값으로 색표현


#####성능평가
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import pandas as pd 

digits = load_digits()
data = digits.data
# print("< Before scaling >")
# print(data)

data = scale(data) # 각 feature들이 동일하게 영향을 주게하기위해 스케일링 
# print("< After scaling >")
# print(data)


labels_true = digits.target
n_samples, n_features = data.shape
clusters = np.unique(labels_true)
n_clusters = len(clusters)

# print("n_samples : " + str(n_samples))
# print("n_features : " + str(n_features))
# print("n_clusters : " + str(n_clusters))
# print("clusters : " + str(clusters))

estimator1 = KMeans(init = 'k-means++', n_clusters = n_clusters, n_init = 10)
estimator1.fit(data)

estimator2 = KMeans(init = 'random', n_clusters = n_clusters, n_init = 10)
estimator2.fit(data)

labels_predict1 = estimator1.labels_
labels_predict2 = estimator2.labels_

### 두 모델 성능평가 및 비교
from sklearn import metrics

print("< clustering performance evaluation >\n")
print("1. clustering with initializing first centroids of clusters with k-means++ function ")
print('homogenity score : %.3f' %(metrics.homogeneity_score(labels_true, labels_predict1)))
print('completeness score : %.3f' %(metrics.completeness_score(labels_true, labels_predict1)))
print('v-measure score : %.3f \n' %(metrics.v_measure_score(labels_true, labels_predict1)))

print("2. clustering with initializing first centroids of clusters randomly ")
print('homogenity score : %.3f' %(metrics.homogeneity_score(labels_true, labels_predict2)))
print('completeness score : %.3f' %(metrics.completeness_score(labels_true, labels_predict2)))
print('v-measure score : %.3f \n' %(metrics.v_measure_score(labels_true, labels_predict2)))


#%%