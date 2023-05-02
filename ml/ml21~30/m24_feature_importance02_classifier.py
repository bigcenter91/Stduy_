#m17-1카피
import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1.데이터
x,y = load_iris(return_X_y=True)

#2. 모델
model_list = [DecisionTreeClassifier,
RandomForestClassifier,
GradientBoostingClassifier,
XGBClassifier]

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

scaler_list = [MinMaxScaler,
               StandardScaler,
               MaxAbsScaler,
               RobustScaler]

# model_name_list = ['DecisionTreeClassifier',
# 'RandomForestClassifier',
# 'GradientBoostingClassifier',
# 'XGBClassifier']
for index, v in enumerate(data_list):
    x,y = v
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27)
    for i, v2 in enumerate(scaler_list):
        scaler = v2()
        x_scaler = scaler.fit_transform(x)
        for i, value in enumerate(model_list):
            model = value()
            # 3. 훈련
            model.fit(x_train,y_train)

            #4. 평가, 예측
            result = model.score(x_test,y_test)
            print("acc : ", result)

            y_predict = model.predict(x_test)
            acc = accuracy_score(y_test,y_predict)
            print("accuracy_score : ", acc)
            print(type(model).__name__, ":", model.feature_importances_)
            print("=====================================================")