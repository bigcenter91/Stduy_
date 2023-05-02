import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold,StratifiedKFold
import warnings
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import time
import random
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action = 'ignore')

seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.3 결측지 제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())

# 1.4 x, y 분리
x = train_csv.drop(['count','hour_bef_precipitation','hour_bef_windspeed'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state = seed,
)

#2. 모델
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = r2_score(y_test,y_predict)
print("r2_score : ", acc)
print("=====================================================")
print(type(model).__name__, ":", model.feature_importances_)
import pandas as pd
#argmin = np.argmin(model.feature_importances_, axis = 0)
# n = 3
# argmin = np.argsort(model.feature_importances_)[:n]

# x_drop = pd.DataFrame(x).drop(argmin, axis = 1)

# x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 123)

# model.fit(x_train1, y_train1)

# result = model.score(x_test1, y_test1)
# print( 'result : ', result)

# y_predict1 = model.predict(x_test1)

# acc1 = r2_score(y_test1, y_predict1)
# print( 'acc1 : ', acc1)

# 지우기전
#[0.59946997 0.17489426 0.01352734 0.02875394 0.03836196 0.03571764, 0.042839   0.04201751 0.02441838]
#0.7291481616042721

#지운후 2개
#[0.60386043 0.18058421 0.04742648 0.04269097 0.04808729 0.04846351 ,0.02888711]
# 0.7321273851334174