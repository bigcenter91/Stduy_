import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import joblib
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.layers import Dense,LeakyReLU,Dropout,Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping as es
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder # SCALER
import matplotlib.pyplot as plt 
import datetime #시간을 저장해주는 놈
from tqdm import tqdm_notebook
import time
from sklearn.metrics import make_scorer, mean_squared_error, f1_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings(action='ignore')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

# Load train and test data
path='c:/study_data/_data/ai_factory/social/'
save_path= './study_data/_save/dust/social/'

train_path='c:/study_data/_data/dust/social/TRAIN/'
train_AWS_path='c:/study_data/_data/dust/social/TRAIN_AWS/'

test_INPUT_path = 'c:/study_data/_data/dust/social/TEST_INPUT/'
test_AWS_path = 'c:/study_data/_data/dust/social/TEST_AWS/'

test_gongju_data = pd.read_csv(test_INPUT_path + '공주.csv')
submission = pd.read_csv(path+'answer_sample.csv')

train_data = pd.read_csv(train_path+ '공주.csv', index_col=0)
train_aws_data = pd.read_csv(train_AWS_path+'공주.csv', index_col=0)

le = LabelEncoder()
for i in train_data.columns:
    if train_data[i].dtype =='object, float64' :
        print(f'Processing column {i}')
        train_data[i] = le.fit_transform(train_data[i])
for u in train_aws_data.columns:
    if train_aws_data[u].dtype =='object, float64':
        print(f'Processing column {u}')
        train_aws_data[u] = le.fit_transform(train_aws_data[u])
for u in train_aws_data.columns:
        if train_aws_data[u].dtype =='object, float64':
            print(f'Processing column {u}')
            train_aws_data[u] = le.fit_transform(train_aws_data[u])

x = train_aws_data.drop(['일시','지점'], axis=1)
y = train_data.drop(['측정소'], axis=1)
y = train_data['PM2.5']
# x["일시"] = x["일시"].str.replace(pat=r'[^\w]', repl=r'', regex=True)
test_gongju_data["일시"] = test_gongju_data["일시"].str.replace(pat=r'[^\w]', repl=r'', regex=True)

x = x.fillna(x.median())
y = y.fillna(y.median())

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75], axis=0)
    print('1사분위 : ', quartile_1) 
    print('q2 : ', q2) 
    print('3사분위 : ', quartile_3) 
    iqr = quartile_3 - quartile_1 
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5) 
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
outliers_loc = outliers(x)
outliers_loc_y = outliers(y)
print('이상치의 위치 : ', list((outliers_loc)))

model = RandomForestRegressor()
model.fit(x, y)

results = model.score(x, y)
print(results)

y_pred = model.predict(test_gongju_data)


y_submit = model.predict(test_gongju_data)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Calories_Burned'] = y_submit
submission.to_csv(save_path + '_submit.csv')