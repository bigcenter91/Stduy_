import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,HalvingRandomSearchCV #N빵
import time
import random
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import RobustScaler
seed = 135 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)
#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state = seed,
)
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
model = HalvingRandomSearchCV(RandomForestClassifier(),parameters,
                     cv=kfold,
                     verbose=1,
                     refit= True, # True는 최상의 파라미터 출력.
                     factor=3.5, #디폴트 3, 소수도 가능.
                     n_jobs=-1
                     )


#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
print(f'runtime : {time.time()-start}')
#그리드서치 :12초
#할빙그리드 : 4초
#print(x.shape, x_train.shape) #(1797, 64) (1437, 64)
#훈련의 전체의 max_resources(최대자원 1437=x_train 값.), max_resources(최소자원)
#factor로 나눠서 52 / 3(factor) = 18(상위) -> resources100 *3 (factor) -> 300
#factor : default = 3, 

print("최적의 매개변수 : " , model.best_estimator_) #내가 쓴것만 나옴.

print("최적의 파라미터 : " , model.best_params_) #전체

print("최적의 점수 : " , model.best_score_)

print("model_score :", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test,y_predict))

y_predict_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', accuracy_score(y_test,y_predict_best))

print(f'runtime : {time.time()-start}')

path = './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending= True)\
    .to_csv(path + 'm13GridSearch3.csv')

# n_iterations: 2                    #2번 전체 훈련
# n_required_iterations: 3
# n_possible_iterations: 2
# min_resources_: 102.0              #최소 훈련 데이터 갯수
# max_resources_: 1257               #최대 훈련 데이터 갯수
# aggressive_elimination: False
# factor: 3.5                        #n_candidates/3.5, n_resources * 3.5
# ----------
# iter: 0
# n_candidates: 25                   #전체 파라미터 개수
# n_resources: 102                   #0번째 훈련때 쓸 훈련데이터 개수
# Fitting 5 folds for each of 25 candidates, totalling 125 fits
# ----------
# iter: 1
# n_candidates: 8                   #전체 파라미터 개수/factor
# n_resources: 357                  #min_resources * factor
# Fitting 5 folds for each of 8 candidates, totalling 40 fits
# runtime : 15.404804468154907

# 최적의 매개변수 :  RandomForestClassifier(n_estimators=400)
# 최적의 파라미터 :  {'n_estimators': 400}
# 최적의 점수 :  0.952112676056338
# model_score : 0.9814814814814815
# accuracy_score : 0.9814814814814815
# 최적 튠 ACC : 0.9814814814814815
# runtime : 15.654845237731934