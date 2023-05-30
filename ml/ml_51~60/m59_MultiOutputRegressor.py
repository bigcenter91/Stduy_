import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


x, y = load_linnerud(return_X_y=True)

print(x)
print(y)
print(x.shape, y.shape) # (20, 3) (20, 3)

#  예상 [138.  33.  68.]]
# model = Ridge()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       round(mean_absolute_error(y, y_pred),4)) # 스코어 : 0.2968777763173123
# print(model.predict([[2, 110, 43]]))

# model = XGBRegressor()
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       (mean_absolute_error(y, y_pred), 4)) # 스코어 : 0.9999999567184008
# print(model.predict([[2, 110, 43]])) # [[138.00215   33.001656  67.99831 ]]

# model = LGBMRegressor() # 에러
# model.fit(x, y)
# print("스코어 :", model.score(x, y)) 
# print(model.predict([[2, 110, 43]])) 
# # ValueError: y should be a 1d array, got an array of shape (20, 3) instead.

# model = MultiOutputRegressor(LGBMRegressor()) # 에러
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       round(mean_absolute_error(y,y_pred),4)) 
# print(model.predict([[2, 110, 43]])) 
# ValueError: y should be a 1d array, got an array of shape (20, 3) instead.

# model = CatBoostRegressor() # 에러
# model.fit(x, y)
# y_pred = model.predict(x)
# print(model.__class__.__name__, "스코어 : ",
#       round(mean_absolute_error(y, y_pred),4)) 
# print(model.predict([[2, 110, 43]]))

model = CatBoostRegressor(loss_function='MultiRMSE') # MultiMAE, MultiMSE는 안된다
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, "스코어 : ",
      round(mean_absolute_error(y, y_pred),4)) 
print(model.predict([[2, 110, 43]]))