from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

tennis_data = pd.read_csv('playtennis.csv')
# print(tennis_data)

tennis_data.Outlook = tennis_data.Outlook.replace('Sunny', 0)
tennis_data.Outlook = tennis_data.Outlook.replace('Overcast', 1)
tennis_data.Outlook = tennis_data.Outlook.replace('Rain', 2)

tennis_data.Temperature = tennis_data.Temperature.replace('Hot', 3)
tennis_data.Temperature = tennis_data.Temperature.replace('Mild', 4)
tennis_data.Temperature = tennis_data.Temperature.replace('Cool', 5)

tennis_data.Humidity = tennis_data.Humidity.replace('High', 6)
tennis_data.Humidity = tennis_data.Humidity.replace('Normal', 7)

tennis_data.Wind = tennis_data.Wind.replace('Weak', 8)
tennis_data.Wind = tennis_data.Wind.replace('Strong', 9)

tennis_data.PlayTennis = tennis_data.PlayTennis.replace('No', 10)
tennis_data.PlayTennis = tennis_data.PlayTennis.replace('Yes', 11)

# print(tennis_data)

X = np.array(pd.DataFrame(tennis_data, columns = ['Outlook', 'Temperature', 'Humidity', 'Wind']))
y = np.array(pd.DataFrame(tennis_data, columns = ['PlayTennis']))
X_train, X_test, y_train, y_test = train_test_split(X, y) # 랜덤으로 데이터 선별

# print('X_train :', X_train)
# print('X_test :', X_test)
# print('y_train :', y_train)
# print('y_test :', y_test)

gnb_clf = GaussianNB()   # 가우시안나이브베이즈모듈 생성
gnb_clf = gnb_clf.fit(X_train, y_train) #트레인 데이터로 모듈 학습

gnb_prediction = gnb_clf.predict(X_test) #테스트값을 예측해서 prediction에 저장
# print(gnb_prediction)

####성능평가

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# print(confusion_matrix(y_test, gnb_prediction))

# print(classification_report(y_test, gnb_prediction))

fmeasure = round(f1_score(y_test, gnb_prediction, average = 'weighted'), 2)

accuracy = round(accuracy_score(y_test, gnb_prediction, normalize = True), 2)

df_nbclf = pd.DataFrame(columns=['Classifier', 'F-Measure', 'Accuracy'])
df_nbclf.loc[len(df_nbclf)] = ['Naive Bayes', fmeasure, accuracy]

print(df_nbclf)