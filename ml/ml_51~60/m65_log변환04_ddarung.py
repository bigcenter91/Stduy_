from sklearn.datasets import fetch_california_housing, load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression


#1. 데이터
path = './_data/ddarung/'
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

###결측치제거### 
train_csv = train_csv.dropna() 

###데이터분리(train_set)###
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

df =pd.DataFrame(x, y)
print(df)

df.plot.box()
plt.show()

df.info()
print(df.describe())

# # df['Population'].boxplot() 이거 안되
# df['bmi'].plot.box() #이거 써
# plt.show()

# df['Population'].hist(bins=50)
# plt.show()

df.hist(bins=50)
# df['target'].hist(bins=50)
plt.show()

# y = df['count']
# x = df.drop(['count'], axis=1)

############# x  로그변환 #############
x['hour_bef_visibility'] = np.log1p(x['hour_bef_visibility']) # 지수 변환 np.exp1m
# 모든 것에 0승은 1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
    )

############# y 로그변환 #############
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
#####################################
model = RandomForestRegressor(random_state=222)
# model = LinearRegression()


#. 컴파일 훈련
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

#4. 평가, 예측
score = model.score(x_test, y_test)
print("로그 > 지수 r2 : ", r2_score(np.expm1(y_test), np.expm1(model.predict(x_test))))
print('score :', score)

# 로그 > 지수 r2 :  -0.011363643395082557
# score : 0.5668035051521175