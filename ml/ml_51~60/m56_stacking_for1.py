import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_diabetes, fetch_covtype, load_wine, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

#1. 데이터
data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes]
# lr = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor()
rf = RandomForestClassifier()

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=123, train_size=0.8, shuffle=True)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #2. 모델
    if i <5:
        model = StackingClassifier(
        estimators=[('LR',rf), ('KNN', knn), ('DT',dt)],
        )
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        
        #4. 평가, 예측
        print('score : ', model.score(x_test, y_test))
        print('Acc : ', accuracy_score(y_test, y_predict))
    else:
        model = StackingRegressor(
        estimators=[('XG',xg), ('LG', lg), ('CAT',cat)],
        )
        
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        
        #4. 평가, 예측
        print('score : ', model.score(x_test, y_test))
        print('r2 : ', r2_score(y_test, y_predict))