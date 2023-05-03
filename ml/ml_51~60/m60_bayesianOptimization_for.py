import numpy as np
from sklearn.datasets import load_diabetes
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

#1 데이터
x, y = load_diabetes(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)

# def lgbm_function (max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
#     model = LGBMRegressor(max_depth = int(max_depth), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
#                           num_leaves = int(num_leaves), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
#                           min_child_samples = int(min_child_samples), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
#                           min_child_weight = min_child_weight,
#                           subsample = subsample,
#                           colsample_bytree = colsample_bytree,
#                           max_bin = int(max_bin), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
#                           reg_lambda = reg_lambda,
#                           reg_alpha = reg_alpha,
#                           random_state = 1234)
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     score = r2_score(y_test, y_predict)
#     return score

# def xgb_function (max_depth, learning_rate , n_estimators , min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
#     model = XGBRegressor(max_depth = int(max_depth), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
#                           learning_rate  = learning_rate,
#                           n_estimators  = int(n_estimators), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
#                           min_child_weight = min_child_weight,
#                           subsample = subsample,
#                           colsample_bytree = colsample_bytree,
#                           max_bin = int(max_bin), # 실수값 전달로 인해 에류 발생 정수값으로 변환해야한다.
#                           reg_lambda = reg_lambda,
#                           reg_alpha = reg_alpha,
#                           random_state = 1234)
#     model.fit(x_train, y_train)
#     y_predict = model.predict(x_test)
#     score = r2_score(y_test, y_predict)
#     return score

def cat_function (max_depth, iterations, learning_rate, l2_leaf_reg, border_count, subsample, colsample_bylevel, random_strength):
    model = CatBoostRegressor(max_depth = int(max_depth),
                              learning_rate = learning_rate,
                              iterations   = int(iterations),
                              l2_leaf_reg = l2_leaf_reg,
                              border_count  = int(border_count),
                              subsample  = subsample,
                              colsample_bylevel  = colsample_bylevel,
                              random_strength  = random_strength,
                              random_state = 1234)
    model.fit(x_train, y_train, verbose = 0)
    y_predict = model.predict(x_test)
    score = r2_score(y_test, y_predict)
    return score

# lgbm_bayesian_params = {'max_depth' : (3, 16),
#                   'num_leaves' : (24, 64),
#                   'min_child_samples' : (10, 200),
#                   'min_child_weight' : (1, 50),
#                   'subsample' : (0.5, 1),
#                   'colsample_bytree' : (0.5, 1),
#                   'max_bin' : (10, 500),
#                   'reg_lambda' : (0.001, 10),
#                   'reg_alpha' : (0.01, 50)}

# xgb_bayesian_params = {'max_depth' : (3, 16),
#                        'learning_rate' : (0.3, 0.7),
#                        'n_estimators' : (100, 500),
#                        'min_child_weight' : (1, 50),
#                        'subsample' : (0.5, 1),
#                        'colsample_bytree' : (0.5, 1),
#                        'max_bin' : (10, 500),
#                        'reg_lambda' : (0.001, 10),
#                        'reg_alpha' : (0.01, 50)}

cat_bayesian_params = {'max_depth' : (3, 16),
                       'iterations' : (50, 500),
                       'learning_rate' : (0.1, 0.3),
                       'l2_leaf_reg' : (1, 15),
                       'border_count' : (32, 100),
                       'subsample' : (0.5, 1),
                       'colsample_bylevel' : (0.5, 1),
                       'random_strength' : (0.001, 1)}

# optimizer = BayesianOptimization(
#     f = lgbm_function,
#     pbounds = lgbm_bayesian_params,
#     random_state = 1234
# )

# optimizer = BayesianOptimization(
#     f = xgb_function,
#     pbounds = xgb_bayesian_params,
#     random_state = 1234
# )

optimizer = BayesianOptimization(
    f = cat_function,
    pbounds = cat_bayesian_params,
    random_state = 1234
)

optimizer.maximize(init_points = 5,
                   n_iter = 20)

print(optimizer.max)