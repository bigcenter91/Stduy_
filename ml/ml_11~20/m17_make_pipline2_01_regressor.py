# 삼중포문!!!
#1. 데이터셋
#2. 스케일러
#3. 모델

import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import pandas as pd


ddarung_path = './_data/ddarung/'
kaggle_bike_path = './_data/kaggle_bike/'

ddarung_train = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike_train = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

#1. 데이터
data_list = [load_boston(return_X_y=True),
             fetch_california_housing(return_X_y=True),
             ddarung_train,
             kaggle_bike_train]

scaler_list = [StandardScaler(),
               MinMaxScaler()]

model_list = [DecisionTreeRegressor(),
              RandomForestRegressor()]

data_name_list = ['보스턴 : ',
                  '캘리포니아 : ',
                  '따릉이 : ',
                  '캐글바이크 : ']

scaler_name_list = ['스탠다드 스케일러 : ',
                    '민맥스 스케일러 : ']

model_name_list = ['DecisionTreeRegressor : ',
                  'RandomForestRegressor : ']

#2. 모델
for i, v in enumerate(data_list):
    x, y = v
    print("===================================")
    print(data_name_list[i])
    for k, v3 in enumerate(scaler_list):
        scaler = v3
        x_scaled = scaler.fit_transform(x)
        print(scaler_name_list[k])
        for j, v2 in enumerate(model_list):
            model = v2
            
            #3. 컴파일, 훈련
            model.fit(x_scaled, y)
            
            #4. 평가, 예측
            results = model.score(x_scaled, y)
            print(model_name_list[j], results)
            y_predict = model.predict(x_scaled)
            r2 = r2_score(y, y_predict)
            print(model_name_list[j], "r2_score:", r2)