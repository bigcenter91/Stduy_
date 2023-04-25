import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV   
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline

#1. 데이터
path = './_data/dacon_ddarung/'
path_save = './_save/dacon_ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

###결측치제거### 
train_csv = train_csv.dropna() 
# print(train_csv.isnull().sum())
# print(train_csv.info())
# print(train_csv.shape)  #(1328, 10)

###데이터분리(train_set)###
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)


parameters = [
    {'rf__n_estimators' : [100,200], 'rf__max_depth' : [6,8,10,12], 'rf__min_samples_leaf' : [3,5,7,10]},
    {'rf__max_depth' : [6,8,10,12], 'rf__min_samples_leaf' : [3,5,7,10]},
    {'rf__min_samples_leaf' : [3,5,7,10], 'rf__min_samples_split' : [2,3,5,10]},
    {'rf__min_samples_split' : [2,3,5,10]}]

#2. 모델구성
pipe = Pipeline([('std', StandardScaler()), ('rf', RandomForestRegressor())])   
model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)


#3. 컴파일, 훈련 
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수:", model.best_estimator_) 
print("최적의 파라미터:", model.best_params_)
print("best_score:", model.best_score_)
print("model.score:", model.score(x_test, y_test))
print("걸린시간 :", round(end_time-start_time,2), "초")

#4. 평가, 예측
y_predict = model.predict(x_test)
print("r2_score:", r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)            
print("최적 튠 r2:", r2_score(y_test, y_pred_best))