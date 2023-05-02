import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold,StratifiedKFold
import warnings
import pandas as pd
from sklearn.model_selection import GridSearchCV
import time
import random
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action = 'ignore')

seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.3 결측지 제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())

# 1.4 x, y 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

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
model = GridSearchCV(RandomForestRegressor(),parameters,
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
print('r2_score :', r2_score(y_test,y_predict))

y_predict_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', r2_score(y_test,y_predict_best))

print(f'runtime : {time.time()-start}')

# 최적의 매개변수 :  RandomForestRegressor()
# 최적의 파라미터 :  {'min_samples_split': 2}
# 최적의 점수 :  0.7618256402095326
# model_score : 0.7291481616042721
# r2_score : 0.7291481616042721
# 최적 튠 ACC : 0.7291481616042721
# runtime : 15.202845573425293