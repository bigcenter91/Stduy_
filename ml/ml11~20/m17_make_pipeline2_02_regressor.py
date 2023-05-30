import numpy as np
from sklearn.datasets import load_boston,fetch_california_housing,load_diabetes
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
warnings.filterwarnings(action='ignore')
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# 1. 데이터
data_list = [load_boston(return_X_y=True),
             #fetch_california_housing(return_X_y=True),
             load_diabetes(return_X_y=True)
             ,]

data_name_list = ['boston',
                  #'california',
                  'diabetes',
                  ]

scaler_list = [MinMaxScaler(),
               StandardScaler(),
               MaxAbsScaler(),
               RobustScaler()]

model_list = [SVR(),
              DecisionTreeRegressor(),
              RandomForestRegressor()]

scaler_name_list = ['MinMaxScaler',
                    'StandardScaler',
                    'MaxAbsScaler',
                    'RobustScaler']

model_name_list = ['LinearSVR',
              'DecisionTreeRegressor',
              'RandomForestRegressor']

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=413)

max_score = 0
max_name = '최대값'
max_scaler = 'max'


for index, value in enumerate(data_list):
        data_name = data_name_list[index]
        print(f"Data Name: {data_name}")
        x, y = value
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.8, shuffle=True, random_state=27)
        for i, scaler in enumerate(scaler_list):
            scaler_name = scaler_name_list[i]
            model_score_list = []
            for j, model in enumerate(model_list):
                model_name = model_name_list[j]
                pipeline = make_pipeline(scaler, model)
                scores = cross_val_score(pipeline, x_train, y_train, cv=kfold)
                model_score = np.mean(scores)
                model_score_list.append(model_score)
                print(f"Data: {data_name}, Scaler: {scaler_name}, Model: {model_name}, Score: {model_score:.4f}")
            if model_score > max_score:
                max_score = model_score
                max_scaler_name = scaler_name
                max_model_name = model_name

        print(f"Data: {data_name_list[index]}, "
              f"Best Scaler: {max_scaler_name}, "
              f"Best Model: {max_model_name}, "
              f"Score: {max_score:.4f}")   

# Data Name: boston
# Data: boston, Scaler: MinMaxScaler, Model: LinearSVR, Score: 0.5536
# Data: boston, Scaler: MinMaxScaler, Model: DecisionTreeRegressor, Score: 0.6825
# Data: boston, Scaler: MinMaxScaler, Model: RandomForestRegressor, Score: 0.8402
# Data: boston, Scaler: StandardScaler, Model: LinearSVR, Score: 0.6096
# Data: boston, Scaler: StandardScaler, Model: DecisionTreeRegressor, Score: 0.7023
# Data: boston, Scaler: StandardScaler, Model: RandomForestRegressor, Score: 0.8375
# Data: boston, Scaler: MaxAbsScaler, Model: LinearSVR, Score: 0.4560
# Data: boston, Scaler: MaxAbsScaler, Model: DecisionTreeRegressor, Score: 0.6974
# Data: boston, Scaler: MaxAbsScaler, Model: RandomForestRegressor, Score: 0.8400
# Data: boston, Scaler: RobustScaler, Model: LinearSVR, Score: 0.5626
# Data: boston, Scaler: RobustScaler, Model: DecisionTreeRegressor, Score: 0.7047
# Data: boston, Scaler: RobustScaler, Model: RandomForestRegressor, Score: 0.8356
# Data: boston, Best Scaler: MinMaxScaler, Best Model: RandomForestRegressor, Score: 0.8402
# Data Name: diabetes
# Data: diabetes, Scaler: MinMaxScaler, Model: LinearSVR, Score: 0.0996
# Data: diabetes, Scaler: MinMaxScaler, Model: DecisionTreeRegressor, Score: -0.0466
# Data: diabetes, Scaler: MinMaxScaler, Model: RandomForestRegressor, Score: 0.4565
# Data: diabetes, Scaler: StandardScaler, Model: LinearSVR, Score: 0.1243
# Data: diabetes, Scaler: StandardScaler, Model: DecisionTreeRegressor, Score: -0.0859
# Data: diabetes, Scaler: StandardScaler, Model: RandomForestRegressor, Score: 0.4605
# Data: diabetes, Scaler: MaxAbsScaler, Model: LinearSVR, Score: 0.0891
# Data: diabetes, Scaler: MaxAbsScaler, Model: DecisionTreeRegressor, Score: -0.0425
# Data: diabetes, Scaler: MaxAbsScaler, Model: RandomForestRegressor, Score: 0.4513
# Data: diabetes, Scaler: RobustScaler, Model: LinearSVR, Score: 0.1349
# Data: diabetes, Scaler: RobustScaler, Model: DecisionTreeRegressor, Score: -0.1415
# Data: diabetes, Scaler: RobustScaler, Model: RandomForestRegressor, Score: 0.4482
# Data: diabetes, Best Scaler: MinMaxScaler, Best Model: RandomForestRegressor, Score: 0.8402