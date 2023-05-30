import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
#1. 데이터
path = 'C:/study_data/_data/dacon_book/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

x = train_csv.drop(['Book-Rating'], axis = 1)
y = train_csv['Book-Rating']


cat_features = ['User-ID', 'Book-ID', 'Location', 'Book-Title', 'Book-Author', 'Publisher']
x[cat_features] = x[cat_features].astype('category')
# x_test[cat_features] = x_test[cat_features].astype('category')
test_csv[cat_features] = test_csv[cat_features].astype('category')

sample_submission = pd.read_csv(path + 'sample_submission.csv')



cvK = KFold(n_splits=5, shuffle=True, random_state=24)
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

min_rmse = np.inf

for k in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)

    def objective(trial, x_train, y_train, x_test, y_test, min_rmse):
        param = {
            'iterations': trial.suggest_int('iterations', 500, 1000),
            'depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate',  0.0005,0.5),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.001, 10),
            'one_hot_max_size' : trial.suggest_int('one_hot_max_size',24, 64),
            # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0, 1),
            'bagging_temperature': trial.suggest_int('bagging_temperature', 0.5, 1),
            'random_strength': trial.suggest_float('random_strength', 0.5, 1),
            # 'border_count': trial.suggest_int('border_count', 64, 128),
        }
        model = CatBoostRegressor(cat_features = cat_features, **param, verbose=0, task_type="GPU")
        model.fit(x_train, y_train)
        val_y_pred = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, val_y_pred))
        print('rmse :',rmse)
        y_pred = model.predict(test_csv)
        y_pred = np.round(y_pred, 3)
        min_rmse = np.inf

        sample_submission[sample_submission.columns[-1]]=y_pred
        if rmse < min_rmse:
            min_rmse = rmse
            print('min rmse :', min_rmse)
            sample_submission.to_csv('C:/study_data/_save/dacon_books/' + f'{np.round(rmse,4)}' + '.csv', index=True)
        return rmse
        
    opt = optuna.create_study(direction='minimize')
    opt.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, min_rmse), n_trials=100)
    print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)