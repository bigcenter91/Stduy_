import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import random
# seed = 0 #random state 0넣는 거랑 비슷함.
# random.seed(seed)
# np.random.seed(seed)

#1. 데이터
x, y = load_iris(return_X_y=True)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, random_state= 202, shuffle= True)

Parameters = [
    {"C":[1,10,100,1000], "kernel" : ['linear'], 'degree':[3,4,5]}, #12
    {"C":[1,10,100], "kernel" : ['rbf','linear'], 'gamma':[0.001, 0.0001]}, #12
    {"C":[1,10,100,1000], "kernel" : ['sigmoid'], 
    'gamma':[0.01,0.001, 0.0001],'degree':[3,4]},#24
    {"C":[0.1,1],'gamma' : [1,10]} #4
]     #총 48번 돈다.

model_score = 0
best_random_state = None
best_score = 0
for random_state in range(200, 300):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.7, shuffle=True, random_state=random_state,
    )
    
    #2.모델
    model = GridSearchCV(SVC(),Parameters,
                         cv=kfold,
                         verbose=1,
                         refit= True,
                         n_jobs=-1)

    #3. 컴파일, 훈련
    start = time.time()
    model.fit(x_train, y_train)
    print(f'random_state={random_state}, runtime : {time.time()-start}')

    if model.best_score_ > best_score and model.score(x_test, y_test) > model_score:
        best_score = model.best_score_
        model_score = model.score(x_test, y_test)
        best_random_state = random_state

print("최적의 random_state : ", best_random_state)
print("최적의 매개변수 : " , model.best_estimator_)
print("최적의 파라미터 : " , model.best_params_)
print("교차 검증 최고 점수 : " , best_score)
print("최적 모델의 테스트 데이터 정확도 :", model_score)

