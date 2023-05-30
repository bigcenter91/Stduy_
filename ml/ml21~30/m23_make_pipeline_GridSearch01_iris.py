# 랜덤서치, 그리드서치, 할빙그리드서치를
#for 문으로 한방에하기
# 단, 패치코브타입처럼 느림놈은 랜덤이나 할빈

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV,HalvingGridSearchCV,RandomizedSearchCV
warnings.filterwarnings(action='ignore')
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# 1. 데이터
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)]

scaler_list = [MinMaxScaler(),
               StandardScaler(),
               MaxAbsScaler(),
               RobustScaler()]

sklearn_list = [HalvingRandomSearchCV(),
                HalvingGridSearchCV(),
                RandomizedSearchCV()]

model_list = [SVC(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]

scaler_name_list = ['MinMaxScaler',
                    'StandardScaler',
                    'MaxAbsScaler',
                    'RobustScaler']

model_name_list = ['SVC',
                   'DecisionTreeClassifier',
                   'RandomForestClassifier']

sklearn_name_list = ['HalvingRandomSearchCV',
                'HalvingGridSearchCV',
                'RandomizedSearchCV']

data_name_list = ['iris : ',
                  'breast_cancer :',
                  'digits :',
                  'load_wine']
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=413)

max_score = 0
max_name = '최대값'
max_scaler = 'max'

parameters = [{'rf__n_estimators' : [100, 200, 300]}, {'rf__max_depth' : [6, 10, 15, 12]}, 
            {'rf__min_samples_leaf' : [3, 10]},
    {'rf__min_samples_split' : [2, 3, 10]}, 
    {'rf__max_depth' : [6, 8, 12]}, 
    {'rf__min_samples_leaf' : [3, 5, 7, 10]},
    {'rf__n_estimators' : [100, 200, 400]},
    {'rf__min_samples_split' : [2, 3, 10]},
]
for index, value in enumerate(data_list):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, shuffle=True, random_state=27)
    
    for i, value2 in enumerate(scaler_list):
        scaler = value2
        model_score_list = []
        
        for j, value3 in enumerate(model_list):
            model = value3
            pipeline = make_pipeline(scaler, model)
            scores = cross_val_score(pipeline, x_train, y_train, cv=kfold)
            model_score = np.mean(scores)
            model_score_list.append(model_score)
            for k, value4 in enumerate(sklearn_list):
                search = value4
                search_name = sklearn_name_list[k]            
            if model_score > max_score:
                max_score = model_score
                max_name = model_name_list[j]
                max_scaler = scaler_name_list[i]

        print(f"\nData: {data_name_list[index]}, Scaler: {scaler_name_list[i]}")
        for k, model_score in enumerate(model_score_list):
            print(f"{model_name_list[k]}: {model_score:.4f}")
        print(f"Data: {data_name_list[index]} Best Model: {max_name} with a score of {max_score:.4f} using {max_scaler} scaling")
        