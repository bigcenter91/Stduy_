import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# split data into train, test, and predict sets
x_train, x_test_predict, y_train, y_test_predict = train_test_split(
    datasets.drop(['T (degC)'], axis=1), datasets['T (degC)'],
    train_size=0.7, shuffle=False, random_state=123,
)

x_test, x_predict, y_test, y_predict = train_test_split(
    x_test_predict, y_test_predict,
    test_size=1/3, shuffle=False, random_state=123,
)

# reshape input data
x_train = np.reshape(x_train.values, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test.values, (x_test.shape[0], x_test.shape[1], 1))
x_predict = np.reshape(x_predict.values, (x_predict.shape[0], x_predict.shape[1], 1))

# print the shapes of the datasets
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_predict.shape)
print(y_predict.shape)


'''
# 2. 모델 구성

# 3. 컴파일, 훈련

# 4. 평가, 예측
# evaluate on test set
y_pred_test = model.predict(x_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
print('RMSE (test set):', rmse_test)
print('R2 score (test set):', r2_test)

# evaluate on predict set
y_pred_predict = model.predict(x_predict)
rmse_predict = np.sqrt(mean_squared_error(y_predict, y_pred_predict))
r2_predict = r2_score(y_predict, y_pred_predict)
print('RMSE (predict set):', rmse_predict)
print('R2 score (predict set):', r2_predict)

'''