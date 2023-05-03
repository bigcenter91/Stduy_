import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')
# *UserWarning: 'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. 
# Pass 'early_stopping()' callback via 'callbacks' argument instead.

#1. 데이터 
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#hyperopt----------------------------------------------------------#
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK

search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.01, 1),          #uniform(실수형태) : 정규분포형태에 따라서 제공하겠다 (->중심부(0.5부분)를 더 많이 제공하게 됨) =>> q값(범위값) 줄 필요 xx
    'depth' : hp.quniform('depth',3, 16, 1),               #quniform(정수형태) /// 1단위이지만, 사실은 1.0 => int해줘야함/ 그래도 1단위이므로 round는 안해도됨
    'one_hot_max_size' : hp.quniform('one_hot_max_size',24, 64, 1),          
    'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 10, 200, 2), 
    'bagging_temperature' : hp.uniform('bagging_temperature', 0.5, 1),
    'random_strength' : hp.uniform('random_strength', 0.5, 1),         
    'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0.01, 30)
}

# Catboost param 디폴트 
# iterations: 1000
# learning_rate: 0.03
# depth: 6
# l2_leaf_reg: 1
# border_count: 254
# random_strength: 1
# bagging_temperature: 1
# early_stopping_rounds: None

#모델 정의 
def cat_hamsu(search_space):
    params = {
        'iterations' : 10,
        'learning_rate' : search_space['learning_rate'],
        'depth' : int(search_space['depth']),
        'l2_leaf_reg' : search_space['l2_leaf_reg'],
        'bagging_temperature' : search_space['bagging_temperature'],
        'random_strength' : search_space['random_strength'],
        'one_hot_max_size' : int(search_space['one_hot_max_size']),
        'min_data_in_leaf' : int(search_space['min_data_in_leaf']),
        'task_type' : 'CPU',
        'logging_level' : 'Silent',
        
    }

    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train)
            #   eval_set=[(x_train, y_train), (x_test, y_test)],
            #   eval_metric='AUC',
            #   verbose=0,
            #   early_stopping_rounds=50
            #   )
    y_predict = model.predict(x_test)
    return_value = -(accuracy_score(y_test, y_predict))

    return return_value

trial_val = Trials()   #hist보기위해


best = fmin(
    fn= cat_hamsu,                            #함수
    space= search_space,                        #파라미터
    algo=tpe.suggest,                           #알고리즘 정의(디폴트) // 베이지안 최적화와의 차이점..
    max_evals=50,                               #베이지안 최적화의 n_iter와 동일(훈련 10번)
    trials=trial_val,                           #결과값 저장
    rstate = np.random.default_rng(seed=10)    #random_state와 동일
)


print("best:", best)


results = [aaa['loss'] for aaa in trial_val.results]   #trial_val.results의 값을 aaa반복해라 / aaa의 ['loss']만 반복해라 
df = pd.DataFrame({
        'learning_rate' : trial_val.vals['learning_rate'],
        'depth' : trial_val.vals['depth'],
        'l2_leaf_reg' : trial_val.vals['l2_leaf_reg'],
        'bagging_temperature' : trial_val.vals['bagging_temperature'],
        'random_strength' : trial_val.vals['random_strength'],
        'one_hot_max_size' : trial_val.vals['one_hot_max_size'],
        'min_data_in_leaf' : trial_val.vals['min_data_in_leaf'],
        'results': results
                   })

print(df)

### results칼럼에 최솟값이 있는 행을 출력 ###
min_row = df.loc[df['results'] == df['results'].min()]
print("최소 행",'\n' , min_row)

### results칼럼에 최솟값이 있는 행에서 results만 출력 ###
min_results = df.loc[df['results'] == df['results'].min(), 'results']
print(min_results.values)  