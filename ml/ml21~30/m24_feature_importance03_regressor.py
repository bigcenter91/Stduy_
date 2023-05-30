import numpy as np
from sklearn.datasets import load_boston,fetch_california_housing,load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

#1.데이터


model_list = [DecisionTreeRegressor,
RandomForestRegressor,
GradientBoostingRegressor,
XGBRegressor]

data_list = [load_boston(return_X_y=True),
             fetch_california_housing(return_X_y=True),
             load_diabetes(return_X_y=True),
             ]

data_name_list = ['boston : ',
                  'breast_cancer :',
                  'diabetes :'
                  ]

scaler_list = [MinMaxScaler,
               StandardScaler,
               MaxAbsScaler,
               RobustScaler]

for index, v in enumerate(data_list):
    x,y = v
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27)
    for i, v2 in enumerate(scaler_list):
        scaler = v2()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        for i, value in enumerate(model_list):
            model = value()
            # 3. 훈련
            model.fit(x_train,y_train)

            #4. 평가, 예측
            result = model.score(x_test,y_test)
            print("acc : ", result)

            y_predict = model.predict(x_test)
            acc = r2_score(y_test,y_predict)
            print("r2_score : ", acc)
            print(type(model).__name__, ":", model.feature_importances_)
            print("=====================================================")