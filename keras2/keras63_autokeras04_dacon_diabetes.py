import autokeras as ak
import numpy as np
from sklearn.datasets import load_wine
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터

path= "c:/stduy_/_data/dacon_diabetes/"


train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)


train_set = train_set.fillna(0)

x = train_set.drop(['Outcome'], axis=1)
y = train_set['Outcome']

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=524)

# AutoKeras 분류 모델 생성
model = ak.StructuredDataClassifier(max_trials=2, overwrite=False)  # 최대 시도 횟수 지정

# 모델 훈련
model.fit(x_train, y_train, epochs=200)


best_model = model.export_model()
print(best_model.summary())

# 모델 평가
results = model.evaluate(x_test, y_test)
print('결과:', results)