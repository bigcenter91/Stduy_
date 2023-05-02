import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


#1. 데이터
path = './_data/시험/'
path_save = './_save/samsung/'

s_data = pd.read_csv(path + '삼성전자 주가2.csv', 
                    index_col='일자', encoding='EUC-KR', delimiter=',')

h_data = pd.read_csv(path + '현대자동차.csv', index_col='일자',
                     encoding='EUC-KR', delimiter=',')

print(s_data.shape, h_data.shape) # (3260, 16) (3140, 16)

print(s_data.columns)
print(h_data.columns)

sh = pd.merge(s_data, h_data, on='일자', how='inner')

# 필요한 특성 추출
x = sh[['시가_x', '고가_x', '저가_x', '종가_x', '시가_y', '고가_y', '저가_y', '종가_y']]


# 결측치와 이상치 처리
x = x.fillna(method='ffill')
x = x.replace([np.inf, -np.inf], np.nan).dropna()
y = sh['종가_x']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

print(x_train)



# sh = pd.concat([s_data, h_data], axis=1, join='inner')
# print(sh)

# X = sh.drop('price', axis=1)
# y = sh['price']



# - 결측치 처리
# 스케일링
# 데이터 분리


#2. 모델 구성


#3. 컴파일, 훈련


#4. 평가, 예측