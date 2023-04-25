#분류모델 - pipline사용
#삼중for문 : 1. 데이터셋, 2.스케일러, 3.모델

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline


#1. 데이터 
datasets = [(load_diabetes(return_X_y=True),'diabetes'),
            (fetch_california_housing(return_X_y=True), 'california')] 
dataname = ['diabetes','california']


# 2. 모델구성
scalers = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]
models = [DecisionTreeRegressor(), RandomForestRegressor()]

# max_score = 0

for data, data_name in datasets: 
    x, y = data
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337)
    # print(f'Data: {data_name}')
    max_score = 0
    for scaler in scalers:
        for model in models:
            pipeline = make_pipeline(scaler, model)
            pipeline.fit(x_train, y_train)            
            score = pipeline.score(x_test, y_test)
            # print(f'{scaler.__class__.__name__} + {model.__class__.__name__} Score: {score:.4f}')

            y_pred = pipeline.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            # print(f'{scaler.__class__.__name__} + {model.__class__.__name__} Accuracy: {acc:.4f}')
            if max_score < r2:
                max_score = r2
                max_name = f'{scaler.__class__.__name__} + {model.__class__.__name__}'
    print('\n')
        #dataset name , 최고모델, 성능
    print('========', data_name,'========')        
    print('최고모델:', max_name, max_score)
    print('================================')  