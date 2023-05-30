# 이중포문!! 
#1. 데이터셋
#2. 스케일러
#3. 모델

import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,load_wine
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler,StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn.utils import all_estimators
warnings.filterwarnings(action = 'ignore')
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
#1. 데이터

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)
             ,]

data_name_list = ['iris : ',
                  'breast_cancer :',
                  'digits :',
                  'load_wine',
                  ]

scaler_list = [MinMaxScaler(),
               StandardScaler(),
               MaxAbsScaler(),
               RobustScaler()]

model_list = [SVC(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]

scaler_name_list = ['MinMaxScaler',
               'StandardScaler',
               'MaxAbsScaler',
               'RobustScaler']

model_name_list = ['SVC',
              'DecisionTreeClassifier',
              'RandomForestClassifier']

n_splits=5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 413)

max_score = 0
max_name = '최대값'
max_scaler = 'max'
#1. 데이터
for index, value in enumerate(data_list):
    x, y = value 
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, train_size=0.8, shuffle=True, random_state=27)
    for i, value2 in enumerate(scaler_list):
        scaler = value2
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        #2. 모델구성
        allAlgorithms = all_estimators(type_filter='classifier')
        max_score = 0
        max_name = '최대값'
        max_scaler = 'max'
        for j, value3 in enumerate(model_list):
            model = value3 #j가 들어가면 데이터값이 나와버림
            #3. 컴파일, 훈련
            model.fit(x_train, y_train)
            #4. 평가, 예측
            results = model.score(x_test, y_test)
            print(model_name_list[j], 'model.score :' ,results)
            y_pred = model.predict(x)
            acc = accuracy_score(y,y_pred)
            print(model_name_list[j], 'accuaracy_score :', acc)

print("==============" ,data_name_list[index], "======================")
print('최고모델 :', max_scaler, max_score)
print("================================================")