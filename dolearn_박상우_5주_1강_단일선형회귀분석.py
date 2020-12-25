
# import pandas as pd
# import numpy as np

# myseries = pd.Series([100, 200, 300, 400, 500], index = ['2020-12-15', '2020-12-16', '2020-12-17', '2020-12-18', '2020-12-19'])
# print(type(myseries))
#<class 'pandas.core.series.Series'>

# print(myseries[1:])
# 2020-12-16    200
# 2020-12-17    300
# 2020-12-18    400
# 2020-12-19    500
# dtype: int64


#next#


# myseries2 = pd.DataFrame([100, 200, 300, 400, 500], columns=['expenses'], index = ['a', 'b', 'c', 'd', 'e'])

# print(myseries2)
# print(myseries2.index)
# print(myseries2.columns)


#next# 3강


# myseries3 = pd.DataFrame([100, 200, 300, 400, 500], columns=['expenses'], index = ['a', 'b', 'c', 'd', 'e'])

# print(myseries3.sum())
# print(myseries3.expenses**2)

#column 추가
# myseries3['values'] = (10, 20, 30, 40, 50) 
# print(myseries3)
# #자동정렬
# myseries3['values'] = pd.DataFrame(["2nd", "1st", "4nd", "3rd", "5nd"], index = ['b', 'a', 'd', 'c', 'e']) 
# print(myseries3)

#column삭제
# del myseries3['values']
# print(myseries3)


#next# 


# df1 = pd.DataFrame(['1', '2', '3'], columns = ["A"])
# df2 = pd.DataFrame(['4', '5', '6', '7'], columns = ["B"])
# df = df1.join(df2, how = 'outer')
# print(df)


#next# 


# #5행 3열의 랜덤데이터 생성
# df3 = pd.DataFrame(np.random.randn(5,3))
# df3.columns = ['a', 'b', 'c']
# print(df3)


#next# 


# print(df3.min())
# print(df3.mean()) #평균
# print(df3.std()) #표준편자
# print(df3.cumsum()) #누적합


#next# 

# print(df3.describe())

#next# 

# df3['division'] = ['X', 'Y', 'X', 'Y', 'Z']
# df4 = df3.groupby(['division']).mean()
# print(df4)


#next# 4강

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# %matplotlib inline

# value = np.random.standard_normal(40) #x축 값 생성
# # print(value)

# index = range(len(value)) #x축 값 개수
# plt.plot(index, value) #x,y로 선과 마커를 호출
# plt.xlim(0, 40) #x축 범위 설정
# plt.ylim(np.min(value)-1, np.max(value)+1) #y축 범위 설정

# plt.figure(figsize = (7, 4))
# plt.plot(value.cumsum(), 'b', lw=1.5) #블루, line width
# plt.plot(value.cumsum(), 'ro') #빨간색원형
# plt.xlabel('index') #x축 이름
# plt.ylabel('value') #y축 이름
# plt.title('Line Plot 1') #그래프 이름


#next#


# value = np.random.standard_normal((30,2))
# print(value)

# plt.figure(figsize = (10, 4))
# plt.plot(value[:,0], 'b', lw=1.5, label='1st') #블루, line width
# plt.plot(value[:,1], 'r', lw=1.5, label='2nd')
# plt.plot(value, 'ro') #빨간색원형
# plt.grid(True)
# plt.legend(loc=0)
# plt.xlabel('index') #x축 이름
# plt.ylabel('value') #y축 이름
# plt.title('Line Plot 2') #그래프 이름


#next#


# plt.figure(figsize = (10, 5))

# plt.subplot(211) #2x1 의 서브플롯중 첫번째 서브플롯
# plt.plot(value[:,0], lw=1.5, label='1st') #블루, line width
# plt.plot(value[:,0], 'ro') #빨간색원형
# plt.grid(True)
# plt.legend(loc=0)
# plt.ylabel('value1') #y축 이름
# plt.title('Line Plot 3') #그래프 이름

# plt.subplot(212) #2x1 의 서브플롯중 두번째 서브플롯
# plt.plot(value[:,1], lw=1.5, label='2nd') #블루, line width
# plt.grid(True)
# plt.legend(loc=0)
# plt.axis('tight')
# plt.ylabel('value2') #y축 이름


#next#


# plt.figure(figsize = (13, 5))

# plt.subplot(231)
# plt.plot(value[:,0], lw=1.5, label='1st')
# plt.plot(value[:,0], 'co') #coral? 오렌지색
# plt.grid(True)
# plt.legend(loc=0)
# plt.ylabel('value') #y축 이름
# plt.title('Line Plot 3') #그래프 이름

# plt.subplot(232)
# plt.plot(value[:,0], 'g--', lw=1.5, label='1st') #초록점선
# plt.grid(True)
# plt.legend(loc=0)
# plt.axis('tight')
# plt.ylabel('value') #y축 이름

# plt.subplot(233)
# plt.plot(value[:,0], 'g', lw=1.5, label='1st') #초록색
# plt.plot(value[:,0], 'bs') #blue sqare
# plt.grid(True)
# plt.legend(loc=0)
# plt.axis('tight')
# plt.ylabel('value')

# plt.subplot(234)
# plt.plot(value[:,1], '*', lw=1.5, label='2st') #별표시
# plt.grid(True)
# plt.legend(loc=0)
# plt.axis('tight')
# plt.ylabel('value')

# plt.subplot(235)
# plt.plot(value[:,1], 'b', lw=1.5, label='2st') #파란색
# plt.plot(value[:,1], 'ms') # 보라색 sqare
# plt.grid(True)
# plt.legend(loc=0)
# plt.axis('tight')
# plt.ylabel('value')

# plt.subplot(236)
# plt.plot(value[:,1], 'r--', lw=1.5, label='2st') #빨강 점선
# plt.grid(True)
# plt.legend(loc=0)
# plt.axis('tight')
# plt.ylabel('value')


#next#


# value = np.random.standard_normal((500,2))
# plt.plot(value[:,0], value[:,1], 'ro')
# plt.grid(False)
# plt.xlabel("val1")
# plt.ylabel("val2")
# plt.title('Scatter Plot1')

# plt.figure(figsize=(7,5))
# plt.scatter(value[:,0], value[:,1], marker='o')
# plt.xlabel("val1")
# plt.ylabel("val2")
# plt.title('Scatter Plot2')

# color = np.random.randint(0,10, len(value))
# plt.figure(figsize=(10,5))
# plt.scatter(value[:,0], value[:,1], c=color, marker='o')
# plt.xlabel("val1")
# plt.ylabel("val2")
# plt.title('Scatter Plot3')



#next#


# plt.figure(figsize=(12, 7))
# plt.hist(value, label=['1st', '2nd'], bins=30, color=('yellow', 'green'))
# plt.grid(True)
# plt.legend(bbox_to_anchor = (1.1, 1), loc='upper right')
# plt.xlabel("value")
# plt.ylabel("frequency")
# plt.title('Histogram')





# value = np.random.standard_normal(500)
# cm = plt.cm.get_cmap('Spectral')
# plt.figure(figsize=(12,6))

# n, bins, patches = plt.hist(value, bins = 30, color='yellow')
# print('1. bins:' + str(bins))
# print('2. the length of bins :' + str(len(bins)))

# bin_centers = 0.5*(bins[:-1] + bins[1:])
# print('3.bin_centers :'+str(bin_centers))
# #정규화 = 현재값-최소값 / 최대값-최소값
# col = (bin_centers - min(bin_centers))/(max(bin_centers)-min(bin_centers))
# print("4. col:"+str(col))

# for c, p in zip(col, patches):
#     plt.setp(p, 'facecolor', cm(c))

# plt.xlabel('value')
# plt.ylabel('frequency')
# plt.title('Histogram3')
# plt.show()



# import seaborn as sns

# data2 = sns.load_dataset('flights')

# pivoted_data = data2.pivot('year', 'month', 'passengers') #index, col, val

# sns.set(context='poster', font='monospace')
# #(데이터, 각 셀에 데이터값 표시할것인지, 10진수포맷, 선두께)
# sns.heatmap(pivoted_data, annot=False, fmt='d', linewidth=3)


# sns.set()
# sns.reset_orig()
# sns.heatmap(pivoted_data, annot=False, fmt='d', linewidth=3)


# from pandas.plotting import scatter_matrix

# value = np.random.randn(500,4)
# df = pd.DataFrame(value, columns=['val1', 'val2', 'val3', 'val4'])
# #(데이터, 투명도, 그림사이즈, 시각화종류)
# scatter_matrix(df, alpha=0.1, figsize=(6,6), diagonal='hist')






#next# 4주차



from sklearn import datasets
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# #train_test_split = 데이터셋을 Training set 과 Test Set 으로 분리
# # StratifiedKFold = 각 fold 내 데이터의 클래스 비율을 일정하게 유지하기 위해 cross validation 을 사용하기 위해
# # cross_val_score = cross validation 결과의 정확도를 측정하기 위해
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, mean_squared_error
# # classification_report = recall, precision, fmeasur를 측정하기 위해
# # roc_auc_score = roc curve 아래 면적 auc를 측정하기 위해


# data = datasets.load_breast_cancer()
# X = data.data #속성 데이터
# y = data.target #클래스 데이터



#####holdout 방법
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) # train 8 : 2 test 로 분류

# clf = DecisionTreeClassifier() #모델 선언, 할당
# clf.fit(X_train, y_train) #모델 학습

# y_pred = clf.predict(X_test)

# print('confusion matrix')
# print(confusion_matrix(y_test, y_pred))

# print('accuracy')
# print(accuracy_score(y_test, y_pred, normalize=True)) #normalize=False 시 제대로 분류된 데이터들이 출력

# print('classfication report')
# print(classification_report(y_test, y_pred))

# print('AUC')
# print(roc_auc_score(y_test, y_pred))

# print("mean squared error")
# print(mean_squared_error(y_test, y_pred))




##### K fold cross validation 방법

# skf = StratifiedKFold(n_splits=10) #10개의 폴드로 실험 진행
# skf.get_n_splits(X,y) #x,y를 10개의 폴드로 나눔

# for train_index, test_index in skf.split(X,y): #실험데이터 출력으로 확인
#     print("train set :", train_index)
#     print("test set :", test_index)

# clf = DecisionTreeClassifier()
# scores = cross_val_score(clf, X, y, cv=skf)

# print('K fold cross validation score')
# print(scores)
# print('average accuracy')
# print(scores.mean())


# skf_sh = StratifiedKFold(n_splits=10, shuffle=True) #10개의 폴드로 실험 진행, 셔플
# skf_sh.get_n_splits(X,y)

# clf = DecisionTreeClassifier()
# scores = cross_val_score(clf, X, y, cv=skf_sh)

# print('K fold cross validation score')
# print(scores)
# print('average accuracy')
# print(scores.mean())



# next # 5주차

#%%
from sklearn import linear_model #회귀분석
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.style.use('ggplot')

# data ={'x':[13, 19, 16, 14, 15, 14], 'y':[40, 83, 62, 48, 58, 43]}
# data = pd.DataFrame(data)
# print(data)

# data.plot(kind='scatter', x='x', y='y', figsize=(5,5), color='black')

# linear_regression = linear_model.LinearRegression() # 선형회귀모델 생성
# linear_regression.fit(X = pd.DataFrame(data['x']), y = data['y']) #데이터학습, 독립변수 x, 종속변수 y
# print('a value =', linear_regression.intercept_) #선형회귀식의 세로축 절편
# print('b value =', linear_regression.coef_) #선형 회귀식의 기울기

####적합도 검증

# prediction = linear_regression.predict(X=pd.DataFrame(data['x'])) # y값 예측
# residual = data['y']-prediction #잔차
# residual.describe()

# SSE = (residual**2).sum() #잔차 제곱의 합
# SST = ((data['y']-data['y'].mean())**2).sum()
# R_squared = 1-(SSE/SST) #결정계수
# print('R squared =', R_squared)

# data.plot(kind='scatter', x='x', y='y', figsize=(5,5), color='black')
# plt.plot(data['x'], prediction, color='blue')

####성능평가
from sklearn.metrics import mean_squared_error

# print('score =', linear_regression.score(X = pd.DataFrame(data['x']), y=data['y'])) #모듈을 통해 결정계수 구하기 = R_squared값과 같음
# print('mean squared error =', mean_squared_error(prediction, data['y']))
# print('rmse =', mean_squared_error(prediction, data['y'])**0.5)



#next#



# boston_house_prices = datasets.load_boston()
# print(boston_house_prices.keys)
# print(boston_house_prices.data.shape)
# print(boston_house_prices.feature_names)

# print(boston_house_prices.DESCR) #데이터셋 각 feature별 정보보기

# data_frame = pd.DataFrame(boston_house_prices.data)
# print(data_frame.tail()) #마지막 5개 데이터 출력

# data_frame['Price'] = boston_house_prices.target #종속변수를 콜럼에 추가
# print(data_frame.tail()) #마지막 5개 데이터 출력

# data_frame.plot(kind='scatter', x=5, y='Price', figsize=(6,6), color='black', xlim = (4, 8), ylim = (10, 45))

# linear_regression = linear_model.LinearRegression()
# linear_regression.fit(X=pd.DataFrame(data_frame[5]), y=data_frame['Price'])
# prediction = linear_regression.predict(X=pd.DataFrame(data_frame[5]))
# print('a value =', linear_regression.intercept_)
# print('b value =', linear_regression.coef_)


####적합도검증

# residuals = data_frame['Price'] - prediction
# # print(residuals.describe())

# SSE = (residuals**2).sum() #잔차 제곱의 합
# SST = ((data_frame['Price']-data_frame['Price'].mean())**2).sum()
# R_squared = 1-(SSE/SST)
# # print('R squared =', R_squared) #Price 가 예측에 영향을 주는 정도 확인

# data_frame.plot(kind='scatter', x=5, y='Price', figsize=(6,6), color='black', xlim=(4,8), ylim=(10,45))
# plt.plot(data_frame[5], prediction, color='blue')

# ####성능평가

# print('score =', linear_regression.score(X = pd.DataFrame(data_frame[5]), y=data_frame['Price']))
# print('mean squared error =', mean_squared_error(prediction, data_frame['Price']))
# print('rmse =', mean_squared_error(prediction, data_frame['Price'])**0.5)







# next # 2강






# data = {'x1':[13, 18, 17, 20, 22, 21],
# 'x2':[9, 7, 17, 11, 8, 10],
# 'y':[20, 22, 30, 27, 35, 32]}
# data = pd.DataFrame(data)
# X = data[['x1','x2']]
# y = data['y']
# # print(data)

# linear_regression = linear_model.LinearRegression()
# linear_regression.fit(X=pd.DataFrame(X), y=y)
# prediction = linear_regression.predict(X=pd.DataFrame(X))
# print('a value =', linear_regression.intercept_)
# print('b value =', linear_regression.coef_)

####적합도 검증

# residuals = y - prediction

# SSE = (residuals**2).sum() #잔차 제곱의 합
# SST = ((y-y.mean())**2).sum()
# R_squared = 1-(SSE/SST)
# print('R squared =', R_squared) #Price 가 예측에 영향을 주는 정도 확인

####성능평가
# print('score =', linear_regression.score(X = pd.DataFrame(X), y=y))
# print('mean squared error =', mean_squared_error(prediction, y))
# print('rmse =', mean_squared_error(prediction, y)**0.5) #오차가 작은것 확인가능


#next#

# boston_house_prices = datasets.load_boston()
# X = pd.DataFrame(boston_house_prices.data)
# y = pd.DataFrame(boston_house_prices.target)
# # print(X.tail())

# linear_regression = linear_model.LinearRegression()
# linear_regression.fit(X=pd.DataFrame(X), y=y)
# prediction = linear_regression.predict(X=pd.DataFrame(X))

# ####적합도 검증
# residuals = y - prediction
# # print(residuals.describe())

# SSE = (residuals**2).sum() #잔차 제곱의 합
# SST = ((y-y.mean())**2).sum()
# R_squared = 1-(SSE/SST)
# # print('R squared =', R_squared)

# ####성능평가
# print('score =', linear_regression.score(X = pd.DataFrame(X), y=y))
# print('mean squared error =', mean_squared_error(prediction, y))
# print('rmse =', mean_squared_error(prediction, y)**0.5) #오차가 작은것 확인가능





#%%