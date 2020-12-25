#%%
from sklearn import linear_model #다중회귀분석
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import datasets

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



#next#









#%%