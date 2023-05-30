import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing, fetch_covtype, load_digits
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
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True,
    # stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0) # 과정이 쫘악 나오기 때문에 0으로 한겨

model = VotingRegressor(
    estimators=[('XG',xg), ('LG', lg), ('CAT',cat)],
    # voting='hard' # 디폴트 하드야!
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('score : ', model.score(x_test, y_test))
print('voting r2 : ', r2_score(y_test, y_pred))

regressor = [xg, lg, cat]
for model2 in regressor:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print("{0}정확도 : {1:.4f}".format(class_name, score2))
# print(model.__class__.__name__)

# score :  0.8790459100605676
# voting r2 :  0.8790459100605676
# XGBRegressor정확도 : 0.8245
# LGBMRegressor정확도 : 0.8627
# CatBoostRegressor정확도 : 0.8879