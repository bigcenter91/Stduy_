import os
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
from tensorflow.keras.layers import Input,Conv1D
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
path = 'c:/study_data/_data/dust/'

train_pm_path = glob.glob(path + 'TRAIN/*.csv')
test_pm_path = glob.glob(path + 'TEST_INPUT/*.csv')
train_aws_path = glob.glob(path + 'TRAIN_AWS/*.csv')
test_aws_path = glob.glob(path + 'TEST_AWS/*.csv')
submission = pd.read_csv('c:/study_data/_data/dust/answer_sample.csv', index_col=0)

from preprocess_ import bring
train_pm = bring(train_pm_path)
test_pm = bring(test_pm_path)
train_aws = bring(train_aws_path)
test_aws = bring(test_aws_path)



# 1.1 지역 라벨인코딩
label = LabelEncoder()

train_pm['측정소'] = label.fit_transform(train_pm['측정소'])
test_pm['측정소'] = label.transform(test_pm['측정소'])
train_aws['지점'] = label.fit_transform(train_aws['지점'])
test_aws['지점'] = label.transform(test_aws['지점'].ffill())



# 1.2 month, hour 열 생성 & 일시 열 제거
train_pm['month'] = train_pm['일시'].str[:2].astype('int8')
train_pm['hour'] = train_pm['일시'].str[6:8].astype('int8')
train_pm = train_pm.drop(['연도', '일시'], axis=1)

test_pm['month'] = test_pm['일시'].str[:2].astype('int8')
test_pm['hour'] = test_pm['일시'].str[6:8].astype('int8')
test_pm = test_pm.drop(['연도', '일시'], axis=1)

train_aws = train_aws.drop(['연도', '일시'], axis=1)

test_aws = test_aws.drop(['연도', '일시'], axis=1)



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



# 1.4 PCA 
from sklearn.decomposition import PCA
# pca = PCA(n_components=3)

# train_aws = pca.fit_transform(train_aws)
# test_aws = pca.transform(test_aws)


















# 2.0 awsmap, pmmap 경로 지정 및 가져오기
from preprocess_ import load_aws_and_pm
awsmap, pmmap = load_aws_and_pm()



# 2.1 awsmap, pmmap의 지역 라벨인코딩
awsmap['Location'] = label.fit_transform(awsmap['Location'])
pmmap['Location'] = label.fit_transform(pmmap['Location'])



# 2.2 awsmap, pmmap을 지역 번호순으로 재정렬 ( 가나다 순서로 번호 인코딩 )
awsmap = awsmap.sort_values(by='Location')
pmmap = pmmap.sort_values(by='Location')



# 2.3 pm관측소로부터 aws관측소의 거리 구하기 ( 17개 x 30개 )
from preprocess_ import distance
dist = distance(awsmap, pmmap)



# 2.4 pm관측소에서 가장 가까운 n(default=3)개의 aws관측소의 인덱스 번호와 환산 가중치 반환
from preprocess_ import scaled_score
result, min_i = scaled_score(dist, pmmap)
dist = dist.values
result = result.values



# 2.5 pm관측소의 날씨 구하기
train_pm = train_pm.values.reshape(17, -1, train_pm.shape[1])
train_aws = np.array(train_aws).reshape(30, -1, train_aws.shape[1])
test_pm = test_pm.values.reshape(17, -1, test_pm.shape[1])
test_aws = np.array(test_aws).reshape(30, -1, test_aws.shape[1])

train_pm_aws = []
for i in range(17):
    train_pm_aws.append(train_aws[min_i[i, 0], :, 1:]*result[0, 0] + train_aws[min_i[i, 1], :, 1:]*result[0, 1] + train_aws[min_i[i, 2], :, 1:]*result[0, 2])

train_data = np.concatenate([train_pm, train_pm_aws], axis=2)

test_pm_aws = []
for i in range(17):
    test_pm_aws.append(test_aws[min_i[i, 0], :, 1:]*result[0, 0] + test_aws[min_i[i, 1], :, 1:]*result[0, 1] + test_aws[min_i[i, 2], :, 1:]*result[0, 2])



# 2.6 역순 예측 데이터 생성
train_pm_reverse = np.flip(train_pm, axis=1)
train_pm_aws_reverse = np.flip(train_pm_aws, axis=1)
test_pm_reverse = np.flip(test_pm, axis=1)
test_pm_aws_reverse = np.flip(test_pm_aws, axis=1)

train_rev_data = np.concatenate([train_pm_reverse, train_pm_aws_reverse], axis=2)




















# 3.1 split_x
timesteps = 10

from preprocess_ import split_x
x = split_x(train_data, timesteps).reshape(-1, timesteps, train_data.shape[2])
x_rev = split_x(train_rev_data, timesteps).reshape(-1, timesteps, train_rev_data.shape[2])



# 3.2 split_y
y = []
for i in range(train_data.shape[0]):
    y.append(train_data[i, timesteps:, 1].reshape(-1,))
y = np.array(y).reshape(-1,)

y_rev=[]
for i in range(train_rev_data.shape[0]):
    y_rev.append(train_rev_data[i, timesteps:, 1].reshape(-1,))
y_rev = np.array(y_rev).reshape(-1,)



# 3.3 train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=555, shuffle=True)
x_rev_train, x_rev_test, y_rev_train, y_rev_test = train_test_split(x_rev, y_rev, train_size=0.8, random_state=555, shuffle=True)



# 3.4 Scaler
scaler = MinMaxScaler()

x_train = x_train.reshape(-1, x.shape[2])
x_test = x_test.reshape(-1, x.shape[2])
x_rev_train = x_rev_train.reshape(-1, x.shape[2])
x_rev_test = x_rev_test.reshape(-1, x.shape[2])

x_train[:, 2:], x_test[:, 2:] = scaler.fit_transform(x_train[:, 2:]), scaler.transform(x_test[:, 2:])
x_rev_train[:, 2:], x_rev_test[:, 2:] = scaler.fit_transform(x_rev_train[:, 2:]), scaler.transform(x_rev_test[:, 2:])
# 2: 를 하는 이유는 0번째는 지역, 1번째는 PM2.5
# 지역은 스케일 의미가 없을거같고, PM2.5는 predict해서 나온값을 계속 scale하기 번거로우니까


# 3.5.0 데이터 용량 줄이기 ( float 64 -> float 32 )
x_train=x_train.reshape(-1, timesteps, x.shape[2]).astype(np.float32)
x_test=x_test.reshape(-1, timesteps, x.shape[2]).astype(np.float32)
y_train=y_train.astype(np.float32)
y_test=y_test.astype(np.float32)



# 3.5.1 역방향 데이터도 적용
x_rev_train=x_rev_train.reshape(-1, timesteps, x.shape[2]).astype(np.float32)
x_rev_test=x_rev_test.reshape(-1, timesteps, x.shape[2]).astype(np.float32)
y_rev_train=y_rev_train.astype(np.float32)
y_rev_test=y_rev_test.astype(np.float32)



# pm 의 열
# 측정소 PM2.5 month hour = ( None, 4 )

# aws 의 열
# (지역) 기온 풍향 풍속 강수량 습도       에서 괄호친 열은 제거

# train_pm_aws의 열
# 기온 풍향 풍속 강수량 습도 = ( None, 5 )

# train_data 의 열 ( 훈련 시킬 x값 )
# pm열 + train_pm_aws의 열
# 측정소 PM2.5 month hour + 기온 풍향 풍속 강수량 습도 = ( None, 9 )




# 3.6 지역 컬런 드랍
x_train = x_train[:,:,1:]
x_test = x_test[:,:,1:]
x_rev_train =x_rev_train[:,:,1:]
x_rev_test =x_rev_test[:,:,1:]

# 4.1 Model
input1 = Input(shape=(timesteps, x_train.shape[2]))
conv1d1 = Conv1D(128,6)(input1)
drop1 = Dropout(0.2)(conv1d1)
lstm1 = LSTM(128, activation='relu', name='lstm1')(drop1)
drop2 = Dropout(0.2)(lstm1)
dense1 = Dense(128, activation='relu', name='dense1')(drop2)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(32, activation='relu', name='dense3')(dense2)
dense4 = Dense(16, activation='relu', name='dense4')(dense3)
output1 = Dense(1, name='output1')(dense4)

model1 = Model(inputs=input1, outputs=output1)
model2 = Model(inputs=input1, outputs=output1)



# 4.2 Compile, fit
model1.compile(loss='mae', optimizer='adam')
model2.compile(loss='mae', optimizer='adam')

es = EarlyStopping(monitor='val_loss',
                   restore_best_weights=True,
                   patience=5
                   )
rl = ReduceLROnPlateau(monitor='val_loss',
                       patience=4,
                       )



stt = time.time()
model1.fit(x_train, y_train, batch_size=128, epochs=100,
          callbacks=[es,rl],
          validation_split=0.2)

model2.fit(x_rev_train, y_rev_train, batch_size=128, epochs=100,
          callbacks=[es,rl],
          validation_split=0.2)

test_pm = np.array(test_pm)
test_pm_aws = np.array(test_pm_aws)




















# 5.1 predict
l=[]
for j in range(17):
    for k in range(64):
        for i in range(120):
            if np.isnan(test_pm[j, 120*k+i, 1]) and i<84:
                test_pm[j, 120*k+i, 1] = model1.predict(np.concatenate([test_pm[j, 120*k+i-1-timesteps:120*k+i-1, 1:], test_pm_aws[j, 120*k+i-1-timesteps:120*k+i-1, :]], axis=1).reshape(-1, timesteps, x_train.shape[2]).astype(np.float32))
            elif i>=84:
                test_pm[j, 120*k+204-i-1, 1] = model2.predict(np.flip(np.concatenate([test_pm[j, 120*k+204-i:120*k+204-i+timesteps, 1:], test_pm_aws[j, 120*k+204-i:120*k+204-i+timesteps, :]], axis=1), axis=0).reshape(-1, timesteps, x_rev_train.shape[2]).astype(np.float32))
            print(f'model 변환 진행중{j}의 {k}의 {i}번')
        l.append(test_pm[j, 120*k+48:120*k+120, 1])

l = np.array(l).reshape(-1,)



# 5.2 Save Submit
submission['PM2.5']=l
submission.to_csv('c:/study_data/_data/dust/Aiur_Submit_time502.csv')



# 5.3 Save Weight
# submission['PM2.5']= np.round(a, 3)
ett = time.time()
print('걸린시간 :', np.round((ett-stt),2),'초')
model1.save("c:/study_data/_save/Aiur_Submit5031.h5")
model2.save("c:/study_data/_save/Aiur_Submit6031.h5")





#PCA 제거, 