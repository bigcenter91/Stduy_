import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, load_digits
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
x, y = load_digits(return_X_y=True)

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

thresholds = np.sort(model.feature_importances_)        #list형태로 들어가있음. #np.sort오름차순 정렬
print(thresholds)




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
    
    
    
# [0.05375443 0.04031569 0.0199406  0.05584563 0.02971438 0.09087079
#  0.16398476 0.04534812 0.07892078 0.04699575 0.09083203 0.13588329
#  0.14759369]
# [0.0199406  0.02971438 0.04031569 0.04534812 0.04699575 0.05375443
#  0.05584563 0.07892078 0.09083203 0.09087079 0.13588329 0.14759369
#  0.16398476]
# Tresh=0.020, n=13, R2: 89.38%
# Tresh=0.030, n=12, R2: 89.95%
# Tresh=0.040, n=11, R2: 89.67%
# Tresh=0.045, n=10, R2: 90.03%
# Tresh=0.047, n=9, R2: 90.46% *****
# Tresh=0.054, n=8, R2: 89.46%
# Tresh=0.056, n=7, R2: 88.62%
# Tresh=0.079, n=6, R2: 85.74%
# Tresh=0.091, n=5, R2: 84.77%
# Tresh=0.091, n=4, R2: 84.68%
# Tresh=0.136, n=3, R2: 85.25%
# Tresh=0.148, n=2, R2: 79.99%
# Tresh=0.164, n=1, R2: 70.39%