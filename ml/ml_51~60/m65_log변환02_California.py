from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

# 로그변환은 회귀 데이터

#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target # 타겟 컬럼이 y값이다
print(df)

# df.plot.box()
# plt.show()


df.info()
print(df.describe())

# # df['Population'].boxplot() 이거 안되
# df['Population'].plot.box() 이거 써
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

df['target'].hist(bins=50)
plt.show()

y = df['target']
x = df.drop(['target'], axis=1)

############# x population 로그변환 #############
x['Population'] = np.log1p(x['Population']) # 지수 변환 np.exp1m
# 모든 것에 0승은 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123#
    )

############# y 로그변환 #############
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
#####################################
model = RandomForestRegressor(random_state=222)

#. 컴파일 훈련
model.fit(x_train, y_train)
score= model.score(x_test, y_test)

#4. 평가, 예측
score = model.score(x_test, y_test)
print("로그 > 지수 r2 : ", r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))))
print('score :', score)

# score : 0.8103694286620738
# score : 0.8356273959629564
