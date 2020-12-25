from sklearn.datasets import load_iris

iris = load_iris()
# print(iris.keys())
# print(iris.feature_names)
# print(iris['data'].shape)
# print(iris['data'][0:10])
# print(iris['target']) # iris의 종류

X = iris['data']
y = iris['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y) #X:데이터 y:타겟, default 비율 1:3

from sklearn.preprocessing import StandardScaler # 데이터의 범위를 평균 0, 표준편차 1의 범위로 정규화

scaler = StandardScaler() 

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




from sklearn.neural_network import MLPClassifier #다중인공신경망(MLP) 분류 알고리즘
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10))

mlp.fit(X_train, y_train) # 이때 나오는 warning은 최적화를 할때 주의하라는 내용, 무시 가능


predictions = mlp.predict(X_test) #타겟데이터로 학습

from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, predictions)) #예측값 비교
print(classification_report(y_test, predictions))