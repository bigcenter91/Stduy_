from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import time
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK
import pandas as pd


#1. 데이터
x, y = load_diabetes(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    shuffle = True,
                                                    random_state = 1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 정수형태면 q, 실수형태면 그냥
search_space = { 
                        'learning_rate' : hp.uniform('learning_rate', 0.001, 1), # 뒤에 1이 사실은 1.0야
                        'max_depth' : hp.quniform ('max_depth', 3, 16, 1),
                        'num_leaves' : hp.quniform ('num_leaves', 24, 64, 1),
                        # 'min_child_samples' : hp.quniform ('min_child_samples', 10, 200, 1),
                        # 'min_child_weight' : hp.quniform ('min_child_weight', 1, 50, 1),
                        'subsample' : hp.uniform('subsample', 0.5, 1), # 0 ~ 1 사이
                        # 'colsample_bytree' : hp.uniform ('colsample_bytree', 0.5, 1),
                        # 'max_bin' : hp.quniform('max_bin', 10, 500, 1),
                        # 'reg_lambda' : hp.uniform('reg_lambda', -0.001, 10, 1), # 
                        # 'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50, 1)
}
# hp.quniform(label, low, high, q) : 최소부터 최대까지 q 간격
# hp.uniform(label, low, high, ) : 최소부터 최대까지 정규분포 간격유지
# hp.randint(label, upper) : 0부터 최대값 upper까지 random한 정수값
# hp.loguniform(label, low, high) : exp(uniform(uniform(low, high))) 지수 변환// 값 반환 / 이거 역시 정규분포


def lgb_hamsu(search_space):
    params = {'n_estimators' : 1000, # 무조건 정수형
              'learning_rate' : search_space['learning_rate'],
              'max_depth' : int(search_space['max_depth']), # 무조건 정수형 // 라운드 처리할 필요는 없지만 인트처리는 해야해
              'num_leaves' : int(search_space['num_leaves']), # 무조건 정수형
            #   'min_child_samples' : int(round(min_child_samples)), # 무조건 정수형
            #   'min_child_weight' : int(round(min_child_weight)), # 무조건 정수형
              'subsample' : search_space['subsample'], # 0 ~ 1 사이 min() 1보다 작은값, max() 0보다 큰값 // min은 안전빵으로 한거야
            #   'colsample_bytree' : colsample_bytree,
            #   'max_bin' : max(int(round(max_bin)), 10), # max_bin와 10을 비교해서 가장 높은 값을 뽑아준다. # 무조건 정수형
            #   'reg_lambda' : max(reg_lambda, 0), # 무조건 양수만 나온다.
            #   'reg_alpha' : reg_alpha
              }
    
    model = LGBMRegressor(**params)
    
    model.fit(x_train, y_train,
              eval_set = [(x_train, y_train), (x_test, y_test)],
              eval_metric = 'rmse',
              verbose = 0,
              early_stopping_rounds = 50)
    
    y_predict = model.predict(x_test)
    results = mean_squared_error(y_test, y_predict) # mse는 낮을수록 좋지?
    return results

trial_val = Trials() # hist를 보기위해

# lgb_bo = BayesianOptimization(f = lgb_hamsu,
#                               pbounds = lgbm_bayesian_params,
#                               random_state = 1234)

best = fmin(
    fn = lgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)

print('best : ', best)
##### pandas dataframe
# results 컬럼에 최소값이 있는 행 출력
a = pd.DataFrame(trial_val.vals)
print(a)

results = [aaa['loss']for aaa in trial_val.results]

# for aaa in trial_val.results : 
#     losses.append(aaa['loss']) 위 for문과 동일하다

# df = pd.DataFrame({'x1' : trial_val.vals['x1'],
#                    'x2' : trial_val.vals['x2'],
#                    'results' : results})

df = pd.DataFrame({
        'learning_rate' : trial_val.vals['learning_rate'],
        'max_depth' : trial_val.vals['max_depth'],
        'num_leaves' : trial_val.vals['num_leaves'],
        'min_child_samples' : trial_val.vals['min_child_samples'],
        'min_child_weight' : trial_val.vals['min_child_weight'],
        'subsample' : trial_val.vals['subsample'],
        'colsample_bytree' : trial_val.vals['colsample_bytree'],
        'max_bin' : trial_val.vals['max_bin'],
        'reg_lambda' : trial_val.vals['reg_lambda'],
        'reg_alpha' : trial_val.vals['reg_alpha'],
         'results': results
                   })
print(df)


### results칼럼에 최솟값이 있는 행을 출력 ###
min_row = df.loc[df['results'] == df['results'].min()]
print("최소 행",'\n' , min_row)


### results칼럼에 최솟값이 있는 행에서 results만 출력 ###
min_results = df.loc[df['results'] == df['results'].min(), 'results']
print(min_results.values)