import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
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
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

n_splits=5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 413) #n_splits 는 분할갯수 5개면 20%
# kfold는 cross validation을 쓸 애
#2.모델구성
model = RandomForestRegressor()

#3,4. 컴파일, 훈련, 예측
scores = cross_val_score(model, x, y, cv = kfold)

print('ACC :', scores, 
      '\n cross_val_score average : ', round(np.mean(scores),4))

# ACC : [0.76997788 0.7976122  0.79447668 0.7963084  0.74337494] 
#  cross_val_score average :  0.7804