import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

parameters = [{'n_estimators' : [100, 200, 300]}, {'max_depth' : [6, 10, 15, 12]}, 
            {'min_samples_leaf' : [3, 10]},
    {'min_samples_split' : [2, 3, 10]}, 
    {'max_depth' : [6, 8, 12]}, 
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_estimators' : [100, 200, 400]},
    {'min_samples_split' : [2, 3, 10]},
]

#2.모델
model = GridSearchCV(RandomForestClassifier(),parameters,
                     cv=kfold,
                     verbose=1,
                     refit= True, # True는 최상의 파라미터 출력.
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

# 최적의 매개변수 :  RandomForestClassifier()
# 최적의 파라미터 :  {'n_estimators': 100}
# 최적의 점수 :  0.9598417721518988
# model_score : 0.9590643274853801
# accuracy_score : 0.9590643274853801
# 최적 튠 ACC : 0.9590643274853801
# runtime : 9.376745700836182