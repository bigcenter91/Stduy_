# 삼중포문!!!
#1. 데이터셋
#2. 스케일러
#3. 모델

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


#1. 데이터
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_wine(return_X_y=True),
             load_digits(return_X_y=True)]

scaler_list = [StandardScaler(),
               MinMaxScaler()]

model_list = [LinearSVC(),
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]

data_name_list = ['아이리스 : ',
                  '브레스트 캔서 : ',
                  '와인 : ',
                  '디지츠 : ']

scaler_name_list = ['스탠다드 스케일러 : ',
                    '민맥스 스케일러 : ']

model_name_list = ['LinearSVC : ',
                  'LogisticRegression : ',
                  'DecisionTreeClassifier : ',
                  'RandomForestClassifier : ']

#2. 모델
for i, v in enumerate(data_list):
    x, y = v
    print("===================================")
    print(data_name_list[i])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    scaler = scaler_list[i]
    for j, v2 in enumerate(model_list):
        model = v2
        pipe = make_pipeline(scaler, model)
        
        #3. 컴파일, 훈련
        pipe.fit(x_train, y_train)
        
        #4. 평가, 예측
        results = pipe.score(x_test, y_test)
        print(model_name_list[j], results)
        y_predict = pipe.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(model_name_list[j], "accuracy_score:", acc)