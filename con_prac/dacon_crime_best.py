import pandas as pd
import numpy as np
import random
import os
# import optuna
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(777)  # Seed 고정

path = 'c:/_study/_data/_dacon_crime/'
save_path = 'c:/_study/_save/dacon_crime/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

train['날씨'] = train['강수량(mm)'] + train['강설량(mm)'] + train['적설량(cm)']
test['날씨'] = test['강수량(mm)'] + test['강설량(mm)'] + test['적설량(cm)']

# x_train = train.drop(['ID', 'TARGET'], axis = 1)
x_train = train.drop(['ID', 'TARGET'], axis = 1)
y_train = train['TARGET']
x_test = test.drop('ID', axis = 1)

le = LabelEncoder()

# '요일'과 '범죄발생지' 특성에 대해 Label Encoding 진행
for feature in ['요일', '범죄발생지']:
    # Train 데이터에 대해 fit과 transform을 동시에 수행
    x_train[feature] = le.fit_transform(x_train[feature])
    # Test 데이터에 대해 transform만 수행
    x_test[feature] = le.transform(x_test[feature])

ordinal_features = ['요일', '범죄발생지']

# Create a new feature 'is_weekend'
x_train['is_weekend'] = x_train['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)
x_test['is_weekend'] = x_test['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)

# Create a new feature 'is_night'
x_train['is_night'] = x_train['시간'].apply(lambda x: 1 if 0 <= x < 6 else 0)
x_test['is_night'] = x_test['시간'].apply(lambda x: 1 if 0 <= x < 6 else 0)

# Create a new feature 'is_weekend_night'
x_train['is_weekend_night'] = x_train['is_weekend'] * x_train['is_night']
x_test['is_weekend_night'] = x_test['is_weekend'] * x_test['is_night']

# Feature Engineering: one-hot encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
x_train_ohe = ohe.fit_transform(x_train[ordinal_features])
x_test_ohe = ohe.transform(x_test[ordinal_features])

x_train = pd.concat([x_train, pd.DataFrame(x_train_ohe, columns=ohe.get_feature_names(ordinal_features))], axis=1)
x_test = pd.concat([x_test, pd.DataFrame(x_test_ohe, columns=ohe.get_feature_names(ordinal_features))], axis=1)

# Scaling the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Handle Imbalanced Data
smote = SMOTE(random_state= 42, k_neighbors = 15)
x_train, y_train = smote.fit_resample(x_train, y_train)
# PCA, 랜덤오버샘플링
xgb_model = XGBClassifier(random_state= 42, use_label_encoder=False)

 # Define parameters for GridSearchCV
params_xgb = {'colsample_bytree': np.linspace(0.5, 1.0, 6),
              'eval_metric': ['logloss'],
              'learning_rate': np.linspace(0.01, 0.2, 20),
              'max_bin': list(range(10, 101, 10)),
              'max_depth': list(range(5, 16, 2)),
              'min_child_weight': list(range(1, 11, 1)),
              'n_estimators': list(range(50, 201, 10)),
              'reg_alpha': np.linspace(0.01, 0.1, 10),
              'reg_lambda': list(range(1, 5, 1)),
              'subsample': np.linspace(0.5, 1.0, 6)}

rand_cv_xgb = RandomizedSearchCV(xgb_model, param_distributions=params_xgb, n_iter=100, cv=2, n_jobs=-1, verbose=2, random_state=42)

# Fit the model
rand_cv_xgb.fit(x_train, y_train)

# Print best parameters and score for the model
print('XGBoost 최적 하이퍼파라미터: ', rand_cv_xgb.best_params_)
print('XGBoost 최고 예측 정확도: ', rand_cv_xgb.best_score_)

# Get the best model
xgb_best = rand_cv_xgb.best_estimator_

# Predict
pred = xgb_best.predict(x_test)

# 제출 파일을 읽어옵니다.
submit = pd.read_csv(path + 'sample_submission.csv')

# 예측한 값을 TARGET 컬럼에 할당합니다.
submit['TARGET'] = pred

#time
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# 예측한 결과를 파일로 저장합니다. index 인자의 값을 False로 설정하지 않으면 제출이 정상적으로 진행되지 않습니다.
submit.to_csv(save_path + date + '_submit.csv', index= False)
# 0.5198933614920027 best 제출점수 : 0.54094