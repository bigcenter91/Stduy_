import os
from typing import Any
import numpy as np
import pandas as pd
import time
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input,Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
import glob

# 0. gpu 사용여부
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)        

# 1.0 train, test, answer 데이터 경로 지정 및 가져오기
path = 'c:/study_data/_data/finedust/'

train_pm_path = glob.glob(path + 'TRAIN/*.csv')
test_pm_path = glob.glob(path + 'TEST_INPUT/*.csv')
train_aws_path = glob.glob(path + 'TRAIN_AWS/*.csv')
test_aws_path = glob.glob(path + 'TEST_AWS/*.csv')
submission = pd.read_csv('c:/study_data/_data/finedust/answer_sample.csv', index_col=0)

from preprocess_3 import bring
train_pm = bring(train_pm_path)
test_pm = bring(test_pm_path)
train_aws = bring(train_aws_path)
test_aws = bring(test_aws_path)

print('# 1.0 Done')



# 1.1 지역 라벨인코딩
label = LabelEncoder()

train_pm['측정소'] = label.fit_transform(train_pm['측정소'])
test_pm['측정소'] = label.transform(test_pm['측정소'])
train_aws['지점'] = label.fit_transform(train_aws['지점'])
test_aws['지점'] = label.transform(test_aws['지점'].ffill())

print('# 1.1 Done')



# 1.2 month, hour 열 생성 & 일시 열 제거
from preprocess_3 import get_hourly_features

# train_pm['month'] = train_pm['일시'].str[:2].astype('int8')
train_pm['hour'] = train_pm['일시'].str[6:8].astype('int8')
train_pm['hour_sin'], train_pm['hour_cos'] = get_hourly_features(train_pm['hour'])
train_pm = train_pm.drop(['연도', '일시', 'hour'], axis=1)

# test_pm['month'] = test_pm['일시'].str[:2].astype('int8')
test_pm['hour'] = test_pm['일시'].str[6:8].astype('int8')
test_pm['hour_sin'], test_pm['hour_cos'] = get_hourly_features(test_pm['hour'])
test_pm = test_pm.drop(['연도', '일시', 'hour'], axis=1)

train_aws = train_aws.drop(['연도', '일시'], axis=1)
test_aws = test_aws.drop(['연도', '일시'], axis=1)

print('# 1.2 Done')



# 1.3 train_pm/aws, test_aws의 결측치 제거 ( 일단 imputer )
imputer = IterativeImputer(XGBRegressor())

train_pm['PM2.5'] = imputer.fit_transform(train_pm['PM2.5'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['기온(°C)'] = imputer.fit_transform(train_aws['기온(°C)'].values.reshape(-1 , 1)).reshape(-1,)
test_aws['기온(°C)'] = imputer.transform(test_aws['기온(°C)'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['풍향(deg)'] = imputer.fit_transform(train_aws['풍향(deg)'].values.reshape(-1 , 1)).reshape(-1,)
test_aws['풍향(deg)'] = imputer.transform(test_aws['풍향(deg)'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['풍속(m/s)'] = imputer.fit_transform(train_aws['풍속(m/s)'].values.reshape(-1 , 1)).reshape(-1,)
test_aws['풍속(m/s)'] = imputer.transform(test_aws['풍속(m/s)'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['강수량(mm)'] = imputer.fit_transform(train_aws['강수량(mm)'].values.reshape(-1 , 1)).reshape(-1,)
test_aws['강수량(mm)'] = imputer.transform(test_aws['강수량(mm)'].values.reshape(-1 , 1)).reshape(-1,)

train_aws['습도(%)'] = imputer.fit_transform(train_aws['습도(%)'].values.reshape(-1 , 1)).reshape(-1,)
test_aws['습도(%)'] = imputer.transform(test_aws['습도(%)'].values.reshape(-1 , 1)).reshape(-1,)

print('# 1.3 Done')


















# 2.0 awsmap, pmmap 경로 지정 및 가져오기
from preprocess_3 import load_aws_and_pm
awsmap, pmmap = load_aws_and_pm()

print('# 2.0 Done')



# 2.1 awsmap, pmmap의 지역 라벨인코딩
awsmap['Location'] = label.fit_transform(awsmap['Location'])
pmmap['Location'] = label.fit_transform(pmmap['Location'])

print('# 2.1 Done')



# 2.2 awsmap, pmmap을 지역 번호순으로 재정렬 ( 가나다 순서로 번호 인코딩 )
awsmap = awsmap.sort_values(by='Location')
pmmap = pmmap.sort_values(by='Location')

print('# 2.2 Done')



# 2.3 pm관측소로부터 aws관측소의 거리 구하기 ( 17개 x 30개 )
from preprocess_3 import distance
dist = distance(awsmap, pmmap)

print('# 2.3 Done')



# 2.4 pm관측소에서 가장 가까운 n(default=3)개의 aws관측소의 인덱스 번호와 환산 가중치 반환
from preprocess_3 import scaled_score
result, min_i = scaled_score(dist, pmmap)
dist = dist.values
result = result.values

print('# 2.4 Done')



# 2.5 pm관측소의 날씨 구하기
train_pm = train_pm.values.reshape(17, -1, train_pm.shape[1])
train_aws = train_aws.values.reshape(30, -1, train_aws.shape[1])
test_pm = test_pm.values.reshape(17, -1, test_pm.shape[1])
test_aws = test_aws.values.reshape(30, -1, test_aws.shape[1])

train_pm_aws = []
for i in range(17):
    train_pm_aws.append(train_aws[min_i[i, 0], :, 1:]*result[0, 0] + train_aws[min_i[i, 1], :, 1:]*result[0, 1] + train_aws[min_i[i, 2], :, 1:]*result[0, 2])
    
test_pm_aws = []
for i in range(17):
    test_pm_aws.append(test_aws[min_i[i, 0], :, 1:]*result[0, 0] + test_aws[min_i[i, 1], :, 1:]*result[0, 1] + test_aws[min_i[i, 2], :, 1:]*result[0, 2])

test_pm_aws = np.array(test_pm_aws)

print('# 2.5 Done')
    


# 2.6 지역 column drop
train_pm = train_pm[:, :, 1:]
test_pm = test_pm[:, :, 1:]

print('# 2.6 Done')



# 2.7 훈련 데이터 준비
train_data = np.concatenate([train_pm, train_pm_aws], axis=2)

print('# 2.7 Done')

# pm 의 열
# 괄호친 열은 제거
# (측정소) PM2.5 (momth) (hour) hour_sin hour_cos = ( None, 3 )

# pm_aws의 열
# (지역) 기온 풍향 풍속 강수량 습도 = ( None, 5 )


# train_data 의 열 ( 훈련 시킬 x값 )
# pm열 + pm_aws의 열
# (측정소) PM2.5 (month) (hour) hour_sin hour_cos + (지역) 기온 풍향 풍속 강수량 습도 = ( None, 8 )
#           0                       1       2               3   4    5    6     7

pm_col_num = 0














# 3.1 split_x
timesteps = 10

from preprocess_3 import split_x
for i in range(72):
    globals()['x{}'.format(i+1)] = split_x(train_data, timesteps, i).reshape(-1, timesteps, train_data.shape[2])

print('# 3.1 Done')



# 3.2 split_y
for i in range(72):
    globals()['y{}'.format(i+1)] = []
    for j in range(train_data.shape[0]):
        globals()['y{}'.format(i+1)].append(train_data[j, timesteps+i:, pm_col_num].reshape(-1,))
    globals()['y{}'.format(i+1)] = np.array(globals()['y{}'.format(i+1)]).reshape(-1,)

print('# 3.2 Done')



# 3.3 train_test_split
for i in range(72):
    globals()['x{}_train'.format(i+1)], globals()['x{}_test'.format(i+1)], globals()['y{}_train'.format(i+1)], globals()['y{}_test'.format(i+1)] = train_test_split(globals()['x{}'.format(i+1)], globals()['y{}'.format(i+1)], train_size=0.8, random_state=323, shuffle=True)

print('# 3.3 Done')



# 3.4 Scaler
# scaler = MinMaxScaler()

# for i in range(72):
#     globals()['x{}_train'.format(i+1)] = globals()['x{}_train'.format(i+1)].reshape(-1, globals()['x{}'.format(i+1)].shape[2])
#     globals()['x{}_test'.format(i+1)] = globals()['x{}_test'.format(i+1)].reshape(-1, globals()['x{}'.format(i+1)].shape[2])

#     globals()['x{}_train'.format(i+1)][:, pm_col_num+3:], globals()['x{}_test'.format(i+1)][:, pm_col_num+3:] = scaler.fit_transform(globals()['x{}_train'.format(i+1)][:, pm_col_num+3:]), scaler.transform(globals()['x{}_test'.format(i+1)][:, pm_col_num+3:])

# test_pm_aws = scaler.transform(test_pm_aws.reshape(-1, test_pm_aws.shape[2])).reshape(-1, timesteps, test_pm_aws.shape[2])

# train_data 의 열 ( 훈련 시킬 x값 )
# pm열 + pm_aws의 열
# (측정소) PM2.5 (month) (hour) hour_sin hour_cos + (지역) 기온 풍향 풍속 강수량 습도 = ( None, 8 )

print('# 3.4 Done')



# 3.5 데이터 용량 줄이기 ( float 64 -> float 32 )
for i in range(72):
    globals()['x{}_train'.format(i+1)]=globals()['x{}_train'.format(i+1)].reshape(-1, timesteps, globals()['x{}'.format(i+1)].shape[2]).astype(np.float32)
    globals()['x{}_test'.format(i+1)]=globals()['x{}_test'.format(i+1)].reshape(-1, timesteps, globals()['x{}'.format(i+1)].shape[2]).astype(np.float32)
    globals()['y{}_train'.format(i+1)]=globals()['y{}_train'.format(i+1)].astype(np.float32)
    globals()['y{}_test'.format(i+1)]=globals()['y{}_test'.format(i+1)].astype(np.float32)

print('# 3.5 Done')




















# 4.1 Model, Compile, fit
es = EarlyStopping(
    monitor='val_loss',
    restore_best_weights=True,
    patience=2
)

rl = ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
)

# newmodel1 = Sequential()
# newmodel1.add(Conv1D(128, 6, input_shape=(timesteps, x1_train.shape[2])))
# # newmodel1.add(Dropout(0.2))
# newmodel1.add(Flatten())
# newmodel1.add(Dense(128, activation='relu'))
# newmodel1.add(Dense(64, activation='relu'))
# newmodel1.add(Dense(32, activation='relu'))
# newmodel1.add(Dense(1))

# newmodel1.compile(loss='mae', optimizer='adam')
    
# newmodel1.fit(x1_train, y1_train,
# batch_size=1024,
# epochs=3,
# callbacks=[es,rl],
# validation_split=0.2)

for i in range(72):
    globals()['model{}'.format(i+1)] = Sequential()
    globals()['model{}'.format(i+1)].add(Conv1D(128, 6, input_shape=(timesteps, x1_train.shape[2])))
    # globals()['model{}'.format(i+1)].add(Dropout(0.2))
    globals()['model{}'.format(i+1)].add(Flatten())
    globals()['model{}'.format(i+1)].add(Dense(128, activation='relu'))
    globals()['model{}'.format(i+1)].add(Dense(64, activation='relu'))
    globals()['model{}'.format(i+1)].add(Dense(32, activation='relu'))
    globals()['model{}'.format(i+1)].add(Dense(1))

    globals()['model{}'.format(i+1)].compile(loss='mae', optimizer='adam')
    
    globals()['model{}'.format(i+1)].fit(
        globals()['x{}_train'.format(i+1)], globals()['y{}_train'.format(i+1)],
        batch_size=128,
        epochs=10,
        callbacks=[es,rl],
        validation_split=0.2)
    print(f'{i}번쨰 훈련 완료')

print('# 4.1 Done')
# print(y1_train.shape)
# print(np.concatenate([test_pm[0, 120*0+48-timesteps:120*0+48,:], test_pm_aws[0, 120*0+48-timesteps:120*0+48, :]], axis=1))
# print(np.concatenate([test_pm[0, 120*0+48-timesteps:120*0+48,:], test_pm_aws[0, 120*0+48-timesteps:120*0+48, :]], axis=1).shape) 
# print(y1_train[0])
# print(model.predict(np.concatenate([test_pm[0, 120*0+48-timesteps:120*0+48,:], test_pm_aws[0, 120*0+48-timesteps:120*0+48, :]], axis=1).reshape(1,10,8)))
# print(model.predict(np.concatenate([test_pm[0, 120*0+48-timesteps:120*0+48,:], test_pm_aws[0, 120*0+48-timesteps:120*0+48, :]], axis=1).reshape(1,10,8)).shape)











# 5.1 predict
l=[]
for j in range(17):
    for k in range(64):
        for i in range(72):
            test_pm[j, 120*k+i+48, pm_col_num] = globals()['model{}'.format(i+1)].predict\
                (np.concatenate([test_pm[j, 120*k+48-timesteps:120*k+48, :], test_pm_aws[j, 120*k+48-timesteps:120*k+48, :]], axis=1)\
                    .reshape(-1, timesteps, globals()['x{}_train'.format(i+1)].shape[2]).astype(np.float32))
            print(f'model 변환 진행중{j}의 {k}의 {i}번')
            print(test_pm[j, 120*k+48:120*k+120, pm_col_num])
        l.append(test_pm[j, 120*k+48:120*k+120, pm_col_num])

print('# 5.1 Done')



# 5.2 Save Submit
l = np.array(l).reshape(-1,)
l = np.round(l/0.004)*0.004
submission['PM2.5']=l
submission.to_csv('c:/study_data/_save/dust/_Submit_time1031.csv')

print('# 5.2 Done')



# 5.3 Save weigths
model1.save("c:/study_data/_save/dust/Aiur_Submit1.h5")
model2.save("c:/study_data/_save/dust/Aiur_Submit2.h5")
model3.save("c:/study_data/_save/dust/Aiur_Submit3.h5")
model4.save("c:/study_data/_save/dust/Aiur_Submit4.h5")
model5.save("c:/study_data/_save/dust/Aiur_Submit5.h5")
model6.save("c:/study_data/_save/dust/Aiur_Submit6.h5")
model7.save("c:/study_data/_save/dust/Aiur_Submit7.h5")
model8.save("c:/study_data/_save/dust/Aiur_Submit8.h5")
model9.save("c:/study_data/_save/dust/Aiur_Submit9.h5")
model10.save("c:/study_data/_save/dust/Aiur_Submit10.h5")
model11.save("c:/study_data/_save/dust/Aiur_Submit11.h5")
model12.save("c:/study_data/_save/dust/Aiur_Submit12.h5")
model13.save("c:/study_data/_save/dust/Aiur_Submit13.h5")
model14.save("c:/study_data/_save/dust/Aiur_Submit14.h5")
model15.save("c:/study_data/_save/dust/Aiur_Submit15.h5")
model16.save("c:/study_data/_save/dust/Aiur_Submit16.h5")
model17.save("c:/study_data/_save/dust/Aiur_Submit17.h5")
model18.save("c:/study_data/_save/dust/Aiur_Submit18.h5")
model19.save("c:/study_data/_save/dust/Aiur_Submit19.h5")
model20.save("c:/study_data/_save/dust/Aiur_Submit20.h5")
model21.save("c:/study_data/_save/dust/Aiur_Submit21.h5")
model22.save("c:/study_data/_save/dust/Aiur_Submit22.h5")
model23.save("c:/study_data/_save/dust/Aiur_Submit23.h5")
model24.save("c:/study_data/_save/dust/Aiur_Submit24.h5")
model25.save("c:/study_data/_save/dust/Aiur_Submit25.h5")
model26.save("c:/study_data/_save/dust/Aiur_Submit26.h5")
model27.save("c:/study_data/_save/dust/Aiur_Submit27.h5")
model28.save("c:/study_data/_save/dust/Aiur_Submit28.h5")
model29.save("c:/study_data/_save/dust/Aiur_Submit29.h5")
model30.save("c:/study_data/_save/dust/Aiur_Submit30.h5")
model31.save("c:/study_data/_save/dust/Aiur_Submit31.h5")
model32.save("c:/study_data/_save/dust/Aiur_Submit32.h5")
model33.save("c:/study_data/_save/dust/Aiur_Submit33.h5")
model34.save("c:/study_data/_save/dust/Aiur_Submit34.h5")
model35.save("c:/study_data/_save/dust/Aiur_Submit35.h5")
model36.save("c:/study_data/_save/dust/Aiur_Submit36.h5")
model37.save("c:/study_data/_save/dust/Aiur_Submit37.h5")
model38.save("c:/study_data/_save/dust/Aiur_Submit38.h5")
model39.save("c:/study_data/_save/dust/Aiur_Submit39.h5")
model40.save("c:/study_data/_save/dust/Aiur_Submit40.h5")
model41.save("c:/study_data/_save/dust/Aiur_Submit41.h5")
model42.save("c:/study_data/_save/dust/Aiur_Submit42.h5")
model43.save("c:/study_data/_save/dust/Aiur_Submit43.h5")
model44.save("c:/study_data/_save/dust/Aiur_Submit44.h5")
model45.save("c:/study_data/_save/dust/Aiur_Submit45.h5")
model46.save("c:/study_data/_save/dust/Aiur_Submit46.h5")
model47.save("c:/study_data/_save/dust/Aiur_Submit47.h5")
model48.save("c:/study_data/_save/dust/Aiur_Submit48.h5")
model49.save("c:/study_data/_save/dust/Aiur_Submit49.h5")
model50.save("c:/study_data/_save/dust/Aiur_Submit50.h5")
model51.save("c:/study_data/_save/dust/Aiur_Submit51.h5")
model52.save("c:/study_data/_save/dust/Aiur_Submit52.h5")
model53.save("c:/study_data/_save/dust/Aiur_Submit53.h5")    
model54.save("c:/study_data/_save/dust/Aiur_Submit54.h5")
model55.save("c:/study_data/_save/dust/Aiur_Submit55.h5")
model56.save("c:/study_data/_save/dust/Aiur_Submit56.h5")
model57.save("c:/study_data/_save/dust/Aiur_Submit57.h5")
model58.save("c:/study_data/_save/dust/Aiur_Submit58.h5")
model59.save("c:/study_data/_save/dust/Aiur_Submit59.h5")
model60.save("c:/study_data/_save/dust/Aiur_Submit60.h5")
model61.save("c:/study_data/_save/dust/Aiur_Submit61.h5")
model62.save("c:/study_data/_save/dust/Aiur_Submit62.h5")
model63.save("c:/study_data/_save/dust/Aiur_Submit63.h5")
model64.save("c:/study_data/_save/dust/Aiur_Submit64.h5")
model65.save("c:/study_data/_save/dust/Aiur_Submit65.h5")
model66.save("c:/study_data/_save/dust/Aiur_Submit66.h5")
model67.save("c:/study_data/_save/dust/Aiur_Submit67.h5")
model68.save("c:/study_data/_save/dust/Aiur_Submit68.h5")
model69.save("c:/study_data/_save/dust/Aiur_Submit69.h5")
model70.save("c:/study_data/_save/dust/Aiur_Submit70.h5")
model71.save("c:/study_data/_save/dust/Aiur_Submit71.h5")
model72.save("c:/study_data/_save/dust/Aiur_Submit72.h5")

print('# 5.3 Done')

