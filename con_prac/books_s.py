import pandas as pd
import numpy as np
# from surprise import SVD, Dataset, Reader, accuracy
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

#1. 데이터
path = 'C:/study_data/_data/dacon_book/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

x = train_csv.drop(['Book-Rating'], axis = 1)
# print(x)     # [871393 rows x 8 columns]

y = train_csv['Book-Rating']
# print(y)    # Name: Book-Rating, Length: 871393, dtype: int64

# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 915
)

categorical_cols = ['User-ID', 'Book-ID', 'Location', 'Book-Title', 'Book-Author', 'Publisher']
x_train[categorical_cols] = x_train[categorical_cols].astype('category')
x_test[categorical_cols] = x_test[categorical_cols].astype('category')
test_csv[categorical_cols] = test_csv[categorical_cols].astype('category')

model = LGBMRegressor()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print("RMSE : ", np.sqrt(mse))


#time
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# Submission
save_path = 'C:/study_data/_save/dacon_books/'
y_sub=model.predict(test_csv)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
sample_submission_csv[sample_submission_csv.columns[-1]]=y_sub
sample_submission_csv.to_csv(save_path + 'book_' + date + '.csv', index=False, float_format='%.0f')