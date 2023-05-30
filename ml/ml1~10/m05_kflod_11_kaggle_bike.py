#index 년도, 일자, 시간 다 중요하지만, 일단은 뺀다.
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
#1.데이터
path = './_data/kaggle_bike/' #맨뒤에/오타

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# print(train_csv.shape) #(10886, 11)
# print(test_csv.shape) #(6493, 8)

#결측치 제거

#print(train_csv.isnull().sum()) #결측치 없음

x = train_csv.drop(['count','casual','registered'], axis =1)

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

# ACC : [0.29612159 0.25134117 0.26577187 0.32038811 0.28922033] 
#  cross_val_score average :  0.2846