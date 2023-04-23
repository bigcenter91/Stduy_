import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import datetime
import time

# Load data
path = "c:/study_data/_data/cal/"
save_path = 'c:/study_data/_save/cal/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# Preprocess data
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

x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# Train XGBoost model
params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
d_train = xgb.DMatrix(x_train, label=y_train)
d_test = xgb.DMatrix(x_test, label=y_test)
watchlist = [(d_train, 'train'), (d_test, 'test')]
bst = xgb.train(params, d_train, num_boost_round=1000, evals=watchlist, early_stopping_rounds=10, verbose_eval=10)

# Predict
y_pred = bst.predict(d_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Make submission
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

d_submit = xgb.DMatrix(test_csv)
y_submit = bst.predict(d_submit)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Calories_Burned'] = y_submit
submission.to_csv(save_path + 'Cal' + date + '.csv')
