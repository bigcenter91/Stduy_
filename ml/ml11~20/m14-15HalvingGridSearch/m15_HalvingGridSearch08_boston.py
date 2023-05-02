import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import time
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state = seed,
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, random_state= 56, shuffle= True)

parameters = [{'n_estimators' : [100, 200, 300]}, {'max_depth' : [6, 10, 15, 12]}, 
            {'min_samples_leaf' : [3, 10]},
    {'min_samples_split' : [2, 3, 10]}, 
    {'max_depth' : [6, 8, 12]}, 
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_estimators' : [100, 200, 400]},
    {'min_samples_split' : [2, 3, 10]},
]

#2.모델
model = HalvingGridSearchCV(RandomForestRegressor(),parameters,
                     cv=kfold,
                     verbose=1,
                     refit= True,# True는 최상의 파라미터 출력.
                     factor= 3.5,
                     n_jobs=-1)

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)

print("최적의 매개변수 : " , model.best_estimator_) #내가 쓴것만 나옴.

print("최적의 파라미터 : " , model.best_params_) #전체

print("최적의 점수 : " , model.best_score_)

print("model_score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('r2_score :', r2_score(y_test,y_predict))

y_predict_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', r2_score(y_test,y_predict_best))

print(f'runtime : {time.time()-start}')

# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=3)
# 최적의 파라미터 :  {'min_samples_split': 3}
# 최적의 점수 :  0.6918946830603492
# model_score : 0.8537913690613159
# r2_score : 0.8537913690613159
# 최적 튠 ACC : 0.8537913690613159
# runtime : 13.26182222366333