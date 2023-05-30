import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold,StratifiedKFold
import warnings
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import time
import random
from sklearn.model_selection import train_test_split

#1.데이터
path = './_data/kaggle_bike/' #맨뒤에/오타

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# print(train_csv.shape) #(10886, 11)
# print(test_csv.shape) #(6493, 8)

#결측치 제거

#print(train_csv.isnull().sum()) #결측치 없음

x = train_csv.drop(['count','holiday','workingday','weather'], axis =1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state = 43,
)

#2. 모델
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = r2_score(y_test,y_predict)
print("r2_score : ", acc)
print("=====================================================")
print(type(model).__name__, ":", model.feature_importances_)

#지우기전
# [2.16821774e-05 2.26652413e-06 6.29248968e-06 1.85303435e-05
#  6.96736615e-05 6.32864019e-05 1.03122331e-04 6.97051685e-05
#  4.89357260e-02 9.50709715e-01] r2_score :  0.9996300398577245
# import heapq
# small_nums = heapq.nsmallest(3, model.feature_importances_)
# print(small_nums)

#지운후
# [2.94089750e-05 7.14189536e-05 6.88611421e-05 9.24561853e-05
#  7.19073108e-05 4.95626596e-02 9.50103288e-01] 0.9996351551785794