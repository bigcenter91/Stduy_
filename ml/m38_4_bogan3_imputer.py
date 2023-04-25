import numpy as np
import pandas as pd
import sklearn as sk
print(sk.__version__)

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
# 결측치에 대한 책임을 돌릴거 같애
# imputer = SimpleImputer()
# imputer = SimpleImputer(strategy='mean')                          # 평균값
# imputer = SimpleImputer(strategy='median')                        # 중위값
# imputer = SimpleImputer(strategy='most_frequent')                 # 최빈값 // 갯수가 같을 경우 가장 작은 값
# imputer = SimpleImputer(strategy='constant')                      # 0 들어갔다
# imputer = SimpleImputer(strategy='constant', fill_value=7777)        
# imputer = KNNImputer()                                            # KNN 알고리즘을 통해 평균값을 찾아낸거다
# imputer = IterativeImputer()
# imputer = IterativeImputer(estimator=DecisionTreeRegressor())
imputer = IterativeImputer(estimator=XGBRFRegressor())              # 작은 데이터는 크게 효과 없지만 큰 데이터에서는 어느정도 먹혀

data2 = imputer.fit_transform(data)
print(data2)

# 결측치 없는 데이터는 없다