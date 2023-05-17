import random
import numpy as np
import pandas as pd
import os
import tensorflow as tf

# 0. Set random seed
seed = 91345
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

# 0.1 gpu 사용여부
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)        



# 1.0 train, test, answer data pathing & import
path = 'c:/study_data/_data/finedust/'

import glob
train_pm_path = glob.glob(path + 'TRAIN/*.csv')
test_pm_path = glob.glob(path + 'TEST_INPUT/*.csv')
train_aws_path = glob.glob(path + 'TRAIN_AWS/*.csv')
test_aws_path = glob.glob(path + 'TEST_AWS/*.csv')
submission = pd.read_csv(path + 'answer_sample.csv', index_col=0)

def bring(filepath:str)->pd.DataFrame:
    li = []
    for i in filepath:
        df = pd.read_csv(i, index_col=None, header=0, encoding='utf-8-sig')
        li.append(df)
    data = pd.concat(li, axis=0, ignore_index=True)
    return data

train_pm = bring(train_pm_path)
test_pm = bring(test_pm_path)
train_aws = bring(train_aws_path)
test_aws = bring(test_aws_path)

print('# 1.0 Done')



# 1.1 Local label encoding
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

train_pm['측정소'] = label.fit_transform(train_pm['측정소'])
test_pm['측정소'] = label.transform(test_pm['측정소'])
train_aws['지점'] = label.fit_transform(train_aws['지점'])
test_aws['지점'] = label.transform(test_aws['지점'].ffill())

print('# 1.1 Done')



# 1.2 Create hour columns (as period function) & remove year, date columns
def get_hourly_features(hour: int):
    """주어진 시간(hour)에 대한 사인과 코사인 함수 값을 반환"""
    radians_per_hour = 2 * np.pi * hour / 24.0
    return [np.sin(radians_per_hour), np.cos(radians_per_hour)]

train_pm['hour'] = train_pm['일시'].str[6:8].astype('int8')
train_pm['hour_sin'], train_pm['hour_cos'] = get_hourly_features(train_pm['hour'])
train_pm = train_pm.drop(['연도', '일시', 'hour'], axis=1)

test_pm['hour'] = test_pm['일시'].str[6:8].astype('int8')
test_pm['hour_sin'], test_pm['hour_cos'] = get_hourly_features(test_pm['hour'])
test_pm = test_pm.drop(['연도', '일시', 'hour'], axis=1)

train_aws = train_aws.drop(['연도', '일시'], axis=1)
test_aws = test_aws.drop(['연도', '일시'], axis=1)

print('# 1.2 Done')



# 1.3 Remove missing values of train_pm/aws ​​( using imputer )
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

imputer = IterativeImputer(XGBRegressor())

train_pm['PM2.5'] = imputer.fit_transform(train_pm['PM2.5'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['기온(°C)'] = imputer.fit_transform(train_aws['기온(°C)'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['풍향(deg)'] = imputer.fit_transform(train_aws['풍향(deg)'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['풍속(m/s)'] = imputer.fit_transform(train_aws['풍속(m/s)'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['강수량(mm)'] = imputer.fit_transform(train_aws['강수량(mm)'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['습도(%)'] = imputer.fit_transform(train_aws['습도(%)'].values.reshape(-1 , 1)).reshape(-1,)

print('# 1.3 Done')










# 2.0 awsmap, pmmap data pathing and import
from typing import Tuple

def load_aws_and_pm()->Tuple[pd.DataFrame, pd.DataFrame]:
    path='c:/study_data/_data/finedust/'
    path_list = os.listdir(path)

    meta='/'.join([path, path_list[1]])
    meta_list=os.listdir(meta)

    awsmap = pd.read_csv('/'.join([meta,meta_list[0]]))
    awsmap = awsmap.drop(awsmap.columns[-1], axis=1)
    pmmap = pd.read_csv('/'.join([meta,meta_list[1]]))
    pmmap = pmmap.drop(pmmap.columns[-1], axis=1)
    return awsmap, pmmap

awsmap, pmmap = load_aws_and_pm()

print('# 2.0 Done')



# 2.1 Local label encoding for awsmap and pmmap
awsmap['Location'] = label.fit_transform(awsmap['Location'])
pmmap['Location'] = label.fit_transform(pmmap['Location'])

print('# 2.1 Done')



# 2.2 Reorder awsmap, pmmap by area number (number encoding in alphabetical order)
awsmap = awsmap.sort_values(by='Location')
pmmap = pmmap.sort_values(by='Location')

print('# 2.2 Done')



# 2.3 Find the distance of the aws station from the pm station (17 x 30)
from haversine import haversine

def distance(awsmap:pd.DataFrame,pmmap:pd.DataFrame)->pd.DataFrame:
    '''pm과 ams관측소 사이의 거리'''
    a = []
    for i in range(pmmap.shape[0]):
        b=[]
        for j in range(awsmap.shape[0]):
            b.append(haversine((np.array(pmmap)[i, 1], np.array(pmmap)[i, 2]), (np.array(awsmap)[j, 1], np.array(awsmap)[j, 2])))
        a.append(b)
    distance = pd.DataFrame(np.array(a),index=pmmap['Location'],columns=awsmap['Location'])
    return distance

dist = distance(awsmap, pmmap)

print('# 2.3 Done')



# 2.4 Returns the index number and scaled weight of n (default=3) aws stations closest to the pm station
def scaled_score(distance:pd.DataFrame,pmmap:pd.DataFrame,near:int=3)->Tuple[pd.DataFrame,np.ndarray]:
    '''pm으로부터 가까운 상위 near개의 환산점수'''
    min_i=[]
    min_v=[]
    for i in range(distance.shape[0]):
        min_i.append(np.argsort(distance.values[i,:])[:near])
        min_v.append(distance.values[i, min_i[i]])

    min_i = np.array(min_i)
    min_v = pd.DataFrame(np.array(min_v),index=distance.index)
    
    for i in range(pmmap.shape[0]):
        for j in range(near):
            min_v.values[i, j]=min_v.values[i, j]**2
            
    sum_min_v = np.sum(min_v, axis=1)

    recip=[]
    for i in range(pmmap.shape[0]):
        recip.append(sum_min_v[i]/min_v.values[i, :])
    recip = np.array(recip)
    recip_sum = np.sum(recip, axis=1)
    coef = 1/recip_sum

    result = []
    for i in range(pmmap.shape[0]):
        result.append(recip[i, :]*coef[i])
    result = pd.DataFrame(np.array(result),index=distance.index)
    return result, min_i

result, min_i = scaled_score(dist, pmmap)
dist = dist.values
result = result.values

print('# 2.4 Done')



# 2.5 Calculate the weather at the pm station
train_pm = train_pm.values.reshape(17, -1, train_pm.shape[1])
train_aws = train_aws.values.reshape(30, -1, train_aws.shape[1])
test_pm = test_pm.values.reshape(17, -1, test_pm.shape[1])
test_aws = test_aws.values.reshape(30, -1, test_aws.shape[1])

train_pm_aws = []
for i in range(17):
    train_pm_aws.append(train_aws[min_i[i, 0], :, 1:]*result[0, 0] + train_aws[min_i[i, 1], :, 1:]*result[0, 1] + train_aws[min_i[i, 2], :, 1:]*result[0, 2])
train_pm_aws = np.array(train_pm_aws)

test_pm_aws = []
for i in range(17):
    test_pm_aws.append(test_aws[min_i[i, 0], :, 1:]*result[0, 0] + test_aws[min_i[i, 1], :, 1:]*result[0, 1] + test_aws[min_i[i, 2], :, 1:]*result[0, 2])

test_pm_aws = np.array(test_pm_aws)

print('# 2.5 Done')



# 2.6 Local column drop
train_pm = train_pm[:, :, 1:]
test_pm = test_pm[:, :, 1:]
 
print('# 2.6 Done')



# 2.7 Clustering into k regions with kMeans
from sklearn.cluster import KMeans

pm_means = [np.mean(train_pm[i, :, 0]) for i in range(17)]
pm_std = [np.std(train_pm[i, :, 0]) for i in range(17)]
cluster = np.array(list(zip(pm_means, pm_std)))

k = 3

kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=seed)
kmeans.fit(cluster)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print('# 2.7 Done')



# 2.8 One-hot encoding of label value of kMeans
from tensorflow.keras.utils import to_categorical

labels = to_categorical(labels)

print('# 2.8 Done')



# 2.9 Data Preparation
train_pm_onehot = []
for i in range(17):
    for j in range(train_pm.shape[1]):
        train_pm_onehot.append(np.concatenate([train_pm[i, j, :], labels[i, :]]))
        
train_pm_onehot = np.array(train_pm_onehot).reshape(17, -1, train_pm.shape[2]+k)
train_data = np.concatenate([train_pm_onehot, train_pm_aws], axis=2)

test_pm_onehot = []
for i in range(17):
    for j in range(test_pm.shape[1]):
        test_pm_onehot.append(np.concatenate([test_pm[i, j, :], labels[i, :]]))

test_pm_onehot = np.array(test_pm_onehot).reshape(17, -1, test_pm.shape[2]+k)
pm_col_num = 0

print('# 2.9 Done')










# 3.1 Split_x
timesteps = 6

def split_x(dt, ts, pred_date):
    a = []
    for j in range(dt.shape[0]):
        b = []
        for i in range(dt.shape[1]-ts-pred_date):
            c = dt[j, i:i+ts, :]
            b.append(c)
        a.append(b)
    return np.array(a)

for i in range(72):
    globals()[f'x{i+1}'] = split_x(train_data, timesteps, i).reshape(-1, timesteps, train_data.shape[2])

print('# 3.1 Done')



# 3.2 Split_y
for i in range(72):
    globals()[f'y{i+1}'] = []
    for j in range(train_data.shape[0]):
        globals()[f'y{i+1}'].append(train_data[j, timesteps+i:, pm_col_num].reshape(-1,))
    globals()[f'y{i+1}'] = np.array(globals()[f'y{i+1}']).reshape(-1,)

print('# 3.2 Done')



# 3.3 Train_test_split
from sklearn.model_selection import train_test_split

for i in range(72):
    globals()[f'x{i+1}_train'], globals()[f'x{i+1}_test'], globals()[f'y{i+1}_train'], globals()[f'y{i+1}_test'] = train_test_split(globals()[f'x{i+1}'], globals()[f'y{i+1}'], train_size=0.7, random_state=seed, shuffle=True)

print('# 3.3 Done')



# 3.4 Reduce File Size ( float 64 -> float 32 )
for i in range(72):
    globals()[f'x{i+1}_train']=globals()[f'x{i+1}_train'].reshape(-1, timesteps, globals()[f'x{i+1}'].shape[2]).astype(np.float32)
    globals()[f'x{i+1}_test']=globals()[f'x{i+1}_test'].reshape(-1, timesteps, globals()[f'x{i+1}'].shape[2]).astype(np.float32)
    globals()[f'y{i+1}_train']=globals()[f'y{i+1}_train'].astype(np.float32)
    globals()[f'y{i+1}_test']=globals()[f'y{i+1}_test'].astype(np.float32)

print('# 3.4 Done')










# 4.1 Model, Compile, Fit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    patience=5
)

rl = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
)

from tensorflow.keras.models import load_model

# for i in range(72):
#     globals()[f'model{i+1}'] = load_model(f"C:/AIA/finedust/pm2.5_code/Aiur_Submit_1353/Aiur_Submit{i+1}.h5")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

for i in range(72):
    globals()[f'model{i+1}'] = Sequential()
    globals()[f'model{i+1}'].add(LSTM(32, input_shape=(timesteps, x1_train.shape[2])))
    globals()[f'model{i+1}'].add(Dense(32, activation='relu'))
    globals()[f'model{i+1}'].add(Dense(8, activation='relu'))
    globals()[f'model{i+1}'].add(Dense(1))
    
    globals()[f'model{i+1}'].compile(loss = 'mae', optimizer = 'adam')
    
    globals()[f'model{i+1}'].fit(
        globals()[f'x{i+1}_train'], globals()[f'y{i+1}_train'],
        batch_size=1024,
        epochs=50,
        callbacks=[es,rl],
        validation_split=0.2)
    print(f'train {i+1} finished')

print('# 4.1 Done')

date








# 5.1 Predict
l=[]
for k in range(17):
    for j in range(64):
        for i in range(72):
            test_pm[k, 120*j+i+48, pm_col_num] = globals()[f'model{i+1}'].predict\
                (np.concatenate([test_pm_onehot[k, 120*j+48-timesteps:120*j+48, :], test_pm_aws[k, 120*j+48-timesteps:120*j+48, :]], axis=1)\
                    .reshape(-1, timesteps, globals()['x{}_train'.format(i+1)].shape[2]).astype(np.float32))
            print(f'model conversion state : {k} - {j} - {i}, {np.round((72*64*k+72*j+i+1)*100/(17*64*72), 5)}% complete')
            # print(test_pm[k, 120*j+48:120*j+120, pm_col_num])
        l.append(test_pm[k, 120*j+48:120*j+120, pm_col_num])

print('# 5.1 Done')


# 5.2 Post-processing 1
l = np.array(l).reshape(-1,)

quantization = 0.004
l = np.round(l/quantization)*quantization
l = l.reshape(17, -1)


# 5.3 Post-processing 2
# train data의 2일차까지를 train data로 만든 모델을 통해 predict한 결과값을
# 예측 시간 단위로 시각화하여 대략적으로 보정
# train data(3일차~5일차)의 실제값들의 평균(train_true)- train data(3일차~5일차)의 predict값들의 평균(train_predict)->plot
primary_complement_point = 12
secondary_complement_point = 56
tertiary_complement_point = 68

for j in range(17):
    for i in range(64):
        l[j, primary_complement_point+72*i:secondary_complement_point+72*i] = l[j, primary_complement_point+72*i:secondary_complement_point+72*i] - 1*quantization
        l[j, secondary_complement_point+72*i:tertiary_complement_point+72*i] = l[j, secondary_complement_point+72*i:tertiary_complement_point+72*i] - 2*quantization
        l[j, tertiary_complement_point+72*i:72+72*i] = l[j, tertiary_complement_point+72*i:72+72*i] - 3*quantization




# 5.4 Save Submit
l = l.reshape(-1,)
submission['PM2.5']=l

path_save = 'c:/study_data/_save/dust/'
submission.to_csv(path_save + '배환희seed91345' + date + '.csv')

print('# 5.2 Done')



# 5.5 Save weigths
for i in range(72):
    globals()[f'model{i+1}'].save(f'c:/study_data/_save/dust/{date}_{seed}_Submit{i+1}.h5')

print('# 5.3 Done')

