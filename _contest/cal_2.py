import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import datetime
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF



#1 데이터
path= "c:/study_data/_data/cal/"
save_path = 'c:/study_data/_save/cal/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

le_gender = LabelEncoder()
le_weight_status = LabelEncoder()

le_gender.fit(train_csv['Gender'])
le_weight_status.fit(train_csv['Weight_Status'])

train_csv['Gender'] = le_gender.transform(train_csv['Gender'])
train_csv['Weight_Status'] = le_weight_status.transform(train_csv['Weight_Status'])

test_csv['Gender'] = le_gender.transform(test_csv['Gender'])
test_csv['Weight_Status'] = le_weight_status.transform(test_csv['Weight_Status'])

train_csv['Total_Height_Inches'] = train_csv['Height(Feet)'] * 12 + train_csv['Height(Remainder_Inches)']
train_csv.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1, inplace=True)

test_csv['Total_Height_Inches'] = test_csv['Height(Feet)'] * 12 + test_csv['Height(Remainder_Inches)']
test_csv.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1, inplace=True)

x = train_csv.drop(['Calories_Burned'], axis = 1)
y = train_csv['Calories_Burned']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.8,
                                                    shuffle = True,
                                                    random_state = 3333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

n_splits = 30
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 3333)

parameters = [{'kernel': [1.0 * RBF(1.0), 2.0 * RBF(2.0)],
               'alpha': [1e-10, 1e-5, 1e-2, 1, 10, 100],
               'length_scale': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}]

model = HalvingRandomSearchCV(GaussianProcessRegressor(),
                     parameters,
                     cv = 5,
                     verbose = 1,
                     refit = True,
                     n_jobs = -1)

start = time.time()
model.fit(x_train, y_train)
end = time.time()

loss = model.score(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)

y_submit = model.predict(test_csv)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Calories_Burned'] = y_submit
submission.to_csv(save_path + 'Cal' + date + '.csv')
print("걸린시간 : ", round(end - start, 2),'초')


# GaussianProcessRegressor
# loss :  -2.119246222358624
# RMSE :  107.78273207398917
# 걸린시간 :  86.34 초
