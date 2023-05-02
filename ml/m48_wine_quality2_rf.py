# outlier 확인
# outlier 처리 
# 돌려

# keras29 참조

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.utils import to_categorical



#1. 데이터 
path = 'c:/stduy_/_data/dacon_wine/'
path_save = 'c:/stduy_/_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(train_csv.shape, test_csv.shape) # (5497, 13) (1000, 12)

le =LabelEncoder()
le.fit(train_csv['type'])
aa = le.transform(train_csv['type'])
print(aa)
print(type(aa))
train_csv['type'] = aa
test_csv['type'] = le.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis=1)
print(x.shape)                       #(5497, 12)
y = train_csv['quality']

print('y의 라벨값 : ', np.unique(y)) # [3 4 5 6 7 8 9]
print(np.unique(y, return_counts=True))

# y=pd.get_dummies(y)
# y = np.array(y)
y=pd.get_dummies(y)
y = np.array(y)

print(y.shape)   
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))
# (5497, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.75, stratify=y
)

scaler = Normalizer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)


parameters = {'n_estimators' : 20000,
              'learning_rate' : 0.5,
              'max_depth': 7,
              'gamma': 0,
              'min_child_weight': 0,
              'subsample': 0.5,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.7,
              'colsample_bynode': 1,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'random_state' : 3333,
            #   'eval_metric' : 'error'
              }



model = XGBClassifier(**parameters, n_jobs=1)
model.set_params(early_stopping_rounds =500, eval_metric = 'error', **parameters)    


# train the model
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose = 0
          )

#4. 평가
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
print(f"accuracy_score: {acc}")
print(f"RMSE: {rmse}")
print("======================================")

# #내보내기
# submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# y_submit = model.predict(test_csv)

# y_submit = np.argmax(y_submit, axis=1)
# y_submit+=3 # 라벨 값하고 맞춰줄려고
# #print(y_submit)

# submission['quality'] = y_submit

# submission.to_csv(path_save + 'submit_0428_1218.csv')