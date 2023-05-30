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
from sklearn.feature_selection import SelectFromModel


#1. 데이터 
x, y = load_diabetes(return_X_y=True)


# features =['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, #stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 10000,
              'learning_rate' : 0.01,
              'max_depth': 3,
              'gamma': 0,
              'min_child_weight': 0,
              'subsample': 0.4,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.7,
              'colsample_bynode': 0,
              'reg_alpha': 1,
              'reg_lambda': 1,
              'random_state' : 123,
            #   'eval_metric' : 'error'
              }

#2. 모델
model = XGBRegressor(**parameters)
model.set_params(early_stopping_rounds =100, eval_metric = 'rmse', **parameters)    


# train the model
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose =0
          )

#4. 평가
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
print(f"R2 score: {r2}")
print(f"RMSE: {rmse}")
print("======================================")

##################################################
print(model.feature_importances_)
# [0.06504052 0.02900812 0.20614523 0.10602617 0.06462499 0.06759115 0.10111099 0.08934037 0.16957761 0.10153492]
thresholds = np.sort(model.feature_importances_)        #list형태로 들어가있음. #np.sort오름차순 정렬
print(thresholds)
# [0.02900812 0.06462499 0.06504052 0.06759115 0.08934037 0.10111099 0.10153492 0.10602617 0.16957761 0.20614523]



for i in thresholds :
    selection = SelectFromModel(model, threshold=i, prefit=True) # False면 다시 훈련
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print('변형된 x_train:', select_x_train.shape, '변형된 x_test:', select_x_test.shape)
    
    selection_model = XGBRegressor()
    selection_model.set_params(early_stopping_rounds = 10, **parameters, eval_metric='rmse') 
    selection_model.fit(select_x_train, y_train,
                        eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
                        verbose=0)
    
    select_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    
    print("Tresh=%.3f, n=%d, R2: %.2f%%" %(i, select_x_train.shape[1], score*100))
    # f는 부동소수 %.3f 소수 3번째자리까지 빼라
    # %.3f = i / %d(정수형) = select_x_train.shape[1] / %.2f%% = score*100
    
# Tresh=0.029, n=10, R2: 47.94%
# Tresh=0.065, n=9, R2: 44.45%
# Tresh=0.065, n=8, R2: 46.38%
# Tresh=0.068, n=7, R2: 46.64%
# Tresh=0.089, n=6, R2: 49.03% *****
# Tresh=0.101, n=5, R2: 48.64%
# Tresh=0.101, n=4, R2: 45.39%
# Tresh=0.106, n=3, R2: 46.33%
# Tresh=0.170, n=2, R2: 43.52%
# Tresh=0.206, n=1, R2: 27.46%