import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
import random

#1.데이터
seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)


# 1.4 x, y 분리
x = train_csv.drop(['count','holiday','workingday','weather'], axis =1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle= True, random_state=27
)

#2. 모델
model_list = [RandomForestRegressor(),
GradientBoostingRegressor(),
DecisionTreeRegressor(),
XGBRegressor()]

# 3. 훈련
for model, value in enumerate(model_list) :
    model = value
    model.fit(x_train,y_train)

    #4. 평가, 예측
    result = model.score(x_test,y_test)
    print("acc : ", result)

    y_predict = model.predict(x_test)
    acc = r2_score(y_test,y_predict)
    
    print("r2_score : ", acc)
    print(type(model).__name__, ":", \
        model.feature_importances_)
    print("=====================================================")
    
    # 하위 20-25%의 피처 제거 후 재학습
    # idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    #argmin = np.argpartition(model.feature_importances_, 4)[:4]
    # x_drop = pd.DataFrame(x).drop(idx, axis=1)
    #idx = np.argsort(-model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    #x_drop = pd.DataFrame(x).drop(x.columns[argmin], axis=1)
    
    n_drop = int(len(model.feature_importances_) * 0.25)
    idx = np.argsort(model.feature_importances_)[-n_drop:]
    x_drop = pd.DataFrame(x).drop(x.columns[idx], axis=1)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_drop, y, train_size=0.7, shuffle=True, random_state=123)

    model.fit(x_train1, y_train1)

    result = model.score(x_test1, y_test1)
    print("acc after feature selection: ", result)

    y_predict1 = model.predict(x_test1)
    acc1 = r2_score(y_test1, y_predict1)

    print("r2_score after feature selection: ", acc1)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")
    
