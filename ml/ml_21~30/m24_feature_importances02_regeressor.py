# 10개 데이터셋
# 10개의 파일을 만든다
# [실습/과제] 피처를 한개씩 삭제하고 성능비교
# 모델 RF로만 한다

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline

#1. 데이터 
datasets = [load_diabetes,
            fetch_california_housing] 
dataname = ['diabetes','california']


model = [DecisionTreeRegressor(), RandomForestRegressor()]
modelname = ['DTR', 'RFR']

for j, v1 in enumerate(datasets):
    x, y = v1(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=337)
    # print(f'Data: {data_name}')
    for i, v in enumerate(model):
        model = v
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
    print(type(model).__name__, ":", "r2:", r2)
    print(type(model).__name__, ":", "컬럼별 중요도", model.feature_importances_)
    print('-------------------------------------------')