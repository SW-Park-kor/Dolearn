#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##### 2015 미세먼지 데이터
data=pd.read_csv('2015_data.csv',encoding="cp949") # CSV파일을 불러오는 함수를 이용
# print(data.tail())

data_seoul=data[data.Location.isin(['서울'])] #location 서울인 데이터만 저장
# print(data_seoul.tail())
# print(data_seoul.corr()) # 상관관계 분석

import seaborn as sns
# sns.set(font_scale=1.5)
# f, ax = plt.subplots(figsize=(20,10))

# seaborn라이브러리를 이용해 heatmap표현 : annot=각 변수 표현, cmap RdBu= 양-붉은색, 음-푸른색
# sns_heatmap=sns.heatmap(data_seoul.corr(),annot=True, fmt=".2f", linewidths=.5, cmap="RdBu_r")


###### 날씨 데이터

data=pd.read_csv('total_weather.csv', encoding="cp949")
# print("데이터의 총 수", len(data))
# print(data.tail())
data_seoul=data[data.Location.isin(['서울'])]
date_dict = {"01":"겨울","02":"겨울","03":"봄", "04":"봄", "05":"봄", "06":"여름", "07":"여름", "08":"여름", "09":"가을","10":"가을","11":"가을","12":"겨울"}
data_seoul['Season']=data_seoul.Date.str[5:7].map(date_dict)
print(data_seoul.tail())

data_seoul=data_seoul[data_seoul.Season.isin(['겨울'])]
# # data_seoul.tail()

sns.set(font_scale=1.5)
f, ax = plt.subplots(figsize=(20,10))
sns_heatmap = sns.heatmap(data_seoul.corr(), annot=True, fmt=".2f", linewidths=.5, cmap="RdBu_r")




#%%