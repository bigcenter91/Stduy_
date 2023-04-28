## 그래프 그린다 ##
# 1. value_counts > 쓰지마
# 2. np.unique의 return_counts 쓰지마
# 3. groupby 써, count() 써

# plt.bar로 그린다 (quality) 컬럼

# 힌트
# 데이터개수(y축) = 데이터 갯수, 주저리 주저리...

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

print(train_csv.shape)

x = train_csv.drop(['quality'], axis=1)
print(x.shape)                       #(5497, 12)
y = train_csv['quality']

# print(x.shape, y.shape)      # (4898, 12)
# print(datasets.describe()) # std = 표준편차
# print(datasets.info())

count_data = train_csv.groupby('quality')['quality'].count()
print(count_data.index, count_data)
# 3      26
# 4     186
# 5    1788
# 6    2416
# 7     924
# 8     152
# 9       5


# import matplotlib.pyplot as plt
# plt.bar(count_data.index, count_data)
# plt.show()
# 권한이 있으면 가능한거야?