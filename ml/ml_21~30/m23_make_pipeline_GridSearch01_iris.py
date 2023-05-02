# 랜덤서치, 그리드서치, 할빙그리드서치
# for문으로 한방에
# 단, 패치코드타입처럼 느린놈은 랜덤이나 할빙

import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score

#1. 데이터 
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, test_size=0.2
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = [
    {'randomforestclassifier__n_estimators' : [100,200], 'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},
    {'randomforestclassifier__min_samples_leaf' : [3,5,7,10], 'randomforestclassifier__min_samples_split' : [2,3,5,10]},
    {'randomforestclassifier__min_samples_split' : [2,3,5,10]}]

#2. 모델구성
# pipe = Pipeline([('std', StandardScaler()), ('rf', RandomForestClassifier())])   
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
models = [GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1), 
          RandomizedSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1), 
          HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1), 
          HalvingRandomSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)]
modelsname = ['그리드서치', '랜덤서치', '할빙그리드서치', '할빙랜덤서치']


best_modelsname = 'max_model'

for i, v in enumerate(models): 
    max_score = 0
    models=v
    models.fit(x_train, y_train)
    y_predict = models.predict(x_test)
    acc= accuracy_score(y_test, y_predict)
    print("accuracy_score:", accuracy_score(y_test, y_predict))
    if max_score< acc:
        max_score = acc
        best_modelsname = modelsname[i]
print("============================")
print('최고모델:', best_modelsname, max_score)
print("============================")