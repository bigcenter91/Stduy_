# 회귀로 맹그러
# 회귀 데이터 올인 for문
# scaler 6개

# 정규분포로 만들고, 분위수를 기준으로 0~1사이로 만들기 때문에
# 이상치에 자유롭다

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 스탠다드 + 민맥스 : 좋을 수도 있고 나쁠 수도 있다 // 0~1 사이 // 사용법은 똑같다
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split


#1. 데이터
# x, y = load_iris(return_X_y=True)
datalist = [load_iris(return_X_y=True), 
            fetch_california_housing(return_X_y=True)]

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, random_state=337, shuffle=True, train_size=0.8,
#     stratify=y,
# )

scaler_list = [StandardScaler(), 
               MinMaxScaler(), 
               MaxAbsScaler(), 
               QuantileTransformer(), 
               PowerTransformer()]

model_list = [RandomForestClassifier()]

data_name_list = ['아이리스: ' ,
                  '캘리포니아: ']

scaler_name_list = ['스탠다드 스케일러 : ',
                    '민맥스 스케일러 : ',
                    '맥스앱스 스케일러 : ',
                    '퀀타일 트랜스포머 : ',
                    '파워 트랜스포머 : ']
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer(method='box-cox')
# scaler = PowerTransformer(method='yeo-johnson')


# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델
# model = RandomForestClassifier()

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# print("결과 :", round(model.score(x_test, y_test)))


for i, v in enumerate(datalist):
    x, y = v
    print("===================================")
    print(data_name_list[i])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=337)
    
    for k, scaler in enumerate(scaler_list):
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
    print(scaler_name_list[k])
    for j, v2 in enumerate(model_list):
        model = v2
        
        #3. 컴파일, 훈련
        model.fit(x_train, y_train)
        
        #4. 평가, 예측
        print("결과 :", round(model.score(x_test, y_test)))


