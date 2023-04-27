import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# #1. 데이터 
# datasets = load_diabetes()
# x = datasets['data']
# y = datasets.target

# print(datasets.feature_names)
# features = datasets['feature_names']
# # features =['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, random_state=337, train_size=0.8, #stratify=y
# )

# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# parameters = {'n_estimators' : 10000,
#               'learning_rate' : 0.01,
#               'max_depth': 3,
#               'gamma': 0,
#               'min_child_weight': 0,
#               'subsample': 0.4,
#               'colsample_bytree': 0.8,
#               'colsample_bylevel': 0.7,
#               'colsample_bynode': 0,
#               'reg_alpha': 1,
#               'reg_lambda': 1,
#               'random_state' : 123,
#             #   'eval_metric' : 'error'
#               }



parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.3,
              'max_depth' : 3,
              'gamma' : 0,
              'min_child_weight' : 0,
              'subsample' : 0.2,
              'colsample_bytree' :0.8,
              'colsample_bylevel' : 0.8,
              'colsample_bynode' : 0,
              'reg_alpha' : 1,
              'reg_lambda' : 1,
              'random_state' : 337,
}

scaler = RobustScaler()

model = XGBRegressor(**parameters)

def Runmodel(a,x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=123, train_size=0.8)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_test, y_test)], early_stopping_rounds = 10, verbose=0, eval_metric='rmse')
    results = model.score(x_test, y_test)
    print(a ,'results :' , results)

    y_predict =  model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    print(a,'r2: ', r2)

    mse = mean_squared_error(y_test, y_predict)
    print(a, 'rmse :', np.sqrt(mse))


x, y = load_diabetes(return_X_y=True)
Runmodel('remain all', x, y)


for i in range(x.shape[1]-1):
    a = model.feature_importances_
    b = np.argmin(a, axis=0)
    c = pd.DataFrame(pd.DataFrame(x).drop(b, axis=1).values)
    Runmodel(f'remain {9-i} column', x, y)