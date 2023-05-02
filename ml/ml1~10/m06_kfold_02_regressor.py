import numpy as np
from sklearn.datasets import load_boston,fetch_california_housing,load_diabetes
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler,StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn.utils import all_estimators
warnings.filterwarnings(action = 'ignore')
#1. 데이터

data_list = [load_boston(return_X_y=True),
             fetch_california_housing(return_X_y=True),
             load_diabetes(return_X_y=True)
             ,]

data_name_list = ['boston',
                  'california',
                  'diabetes',
                  ]

scaler_list = [MinMaxScaler(),
               StandardScaler(),
               MaxAbsScaler(),
               RobustScaler()]

scaler_name_list = ['MinMaxScaler',
               'StandardScaler',
               'MaxAbsScaler',
               'RobustScaler']

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=413)

for index, value in enumerate(data_list):
    x, y = value
    print("==================", data_name_list[index], "======================")
    for i, value2 in enumerate(scaler_list):
        scaler = value2
        max_score = 0
        max_name = '최대값'
        max_scaler = 'max'
        for name, algorithm in all_estimators(type_filter='regressor'):
            try:
                model = algorithm()
                scores = cross_val_score(model, scaler.fit_transform(x), y, cv=kfold)
                results = round(np.mean(scores), 4)
                if max_score < results:
                    max_score = results
                    max_name = name
                    max_scaler = scaler_name_list[i]
            except:
                continue
        print('Scaler:', max_scaler, 'Model:', max_name, 'Score:', max_score)
    print("======================================================")