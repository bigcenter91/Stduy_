import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.svm import SVC
import time
import random
import pandas as pd

seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state = seed,
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, random_state= 56, shuffle= True)

Parameters = [
    {"C":[1,10,100,1000], "kernel" : ['linear'], 'degree':[3,4,5]}, #12
    {"C":[1,10,100], "kernel" : ['rbf','linear'], 'gamma':[0.001, 0.0001]}, #12
    {"C":[1,10,100,1000], "kernel" : ['sigmoid'], 
     'gamma':[0.01,0.001, 0.0001],'degree':[3,4]},#24
    {"C":[0.1,1],'gamma' : [1,10]} #4
]     #총 48번 돈다.

#2.모델
model = HalvingRandomSearchCV(SVC(),Parameters,
                     cv=4,
                     verbose=1,
                     refit= True, # True는 최상의 파라미터 출력.
                     #refit= False, #로하면 에러나옴. -> 디폴트 False하게 되면 가장마지막값만 냄. 
                     #AttributeError: 'GridSearchCV' object has no attribute 'best_estimator_' 가 없다.
                     factor= 3.5, #디폴트는 10, 디폴트일 경우 10번 *cv만큼 훈련
                     n_jobs=-1)

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)


print("최적의 매개변수 : " , model.best_estimator_) #내가 쓴것만 나옴.

print("최적의 파라미터 : " , model.best_params_) #전체

print("최적의 점수 : " , model.best_score_)

print("model_score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test,y_predict))

y_predict_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', accuracy_score(y_test,y_predict_best))

print(f'runtime : {time.time()-start}')

# 최적의 매개변수 :  SVC(C=1000, degree=5, kernel='linear')
# 최적의 파라미터 :  {'kernel': 'linear', 'degree': 5, 'C': 1000}
# 최적의 점수 :  0.9537627551020408
# model_score : 0.9415204678362573
# accuracy_score : 0.9415204678362573
# 최적 튠 ACC : 0.9415204678362573
# runtime : 21.53453516960144