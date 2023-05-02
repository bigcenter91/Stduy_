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

#1.데이터
path = './_data/kaggle_bike/' #맨뒤에/오타

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# print(train_csv.shape) #(10886, 11)
# print(test_csv.shape) #(6493, 8)

#결측치 제거

#print(train_csv.isnull().sum()) #결측치 없음

x = train_csv.drop(['count','holiday','workingday','weather'], axis =1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state = 43,
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

import matplotlib.pyplot as plt
def plot_feature_importances(model):
    n_features = train_csv.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), train_csv.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    plt.title(model)
    
plot_feature_importances(model)
plt.show()