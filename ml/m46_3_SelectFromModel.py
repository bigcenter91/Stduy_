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
from sklearn.linear_model import LinearRegression

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

model = LinearRegression(**parameters)

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
    
########################################################
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




print(model.feature_importances_)
# [0.08909006 0.03688963 0.10926993 0.07085406 0.10310002 0.07881135
#  0.06157534 0.04914304 0.3419893  0.05927724]
thresholds = np.sort(model.feature_importances_)
print(thresholds)


for i in thresholds :
    selection = SelectFromModel(model, threshold=i, prefit=True) # False면 다시 훈련
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print('변형된 x_train:', select_x_train.shape, '변형된 x_test:', select_x_test.shape)
    
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
    
