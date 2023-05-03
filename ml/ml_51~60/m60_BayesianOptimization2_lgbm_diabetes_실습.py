from sklearn.datasets import load_diabetes
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from lightgbm import LGBMRegressor

bayesian_params = {
    'max_depth' : (3, 16),
    'num_leaves' : (24, 64),
    'min_child_sample' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (10, 500),
    'reg_lambda' : (0.001, 10),
    'reg_alpha' : (0.01, 50)
}

def lgbm_cv(max_depth, num_leaves, min_child_sample, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'max_depth': int(max_depth),
        'num_leaves': int(num_leaves),
        'min_child_sample': int(min_child_sample),
        'min_child_weight': int(min_child_weight),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'max_bin': int(max_bin),
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'random_state': 123
    }
    model = LGBMRegressor(**params)
    
    
    
    lgbm_bo = BayesianOptimization(lgbm_cv, bayesian_params, random_state=123)
    lgbm_bo.maximize(init_points=5, n_iter=20)

    print(lgbm_bo.max)
        
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2:.4f}")
        
    
    