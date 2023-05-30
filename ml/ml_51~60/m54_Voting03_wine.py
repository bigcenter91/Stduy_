import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor # 리프노드의 차이다_xgb와 차이 
from catboost import CatBoostRegressor

#1. 데이터
x, y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
lr = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()

model = VotingClassifier(
    estimators=[('LR',lr), ('KNN', knn), ('DT',dt)],
    voting='soft' # 디폴트 하드야!
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('score : ', model.score(x_test, y_test))
print('voting Acc : ', accuracy_score(y_test, y_pred))

classfiers = [lr, knn, dt]
for model2 in classfiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print("{0}정확도 : {1:.4f}".format(class_name, score2))