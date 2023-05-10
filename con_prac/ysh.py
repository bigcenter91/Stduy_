import math
import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import time
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
import datetime 
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 
def split_x(dataset, timesteps): # split_x라는 함수를 정의
    aaa = [] # aaa 에 빈칸의 리스트를 만든다.
    for i in range(len(dataset) - timesteps + 1): # for : 반복문, i 변수에, range 일정 간격으로 숫자를 나열 len 데이터의 길이 시작값 - 끝값 + 증가값, in 과 : 사이가 반복할 횟수 
        subset = dataset[i : (i + timesteps)] # subset 변수에 dataset i0 : (i0 + 5)
        aaa.append(subset) # aaa 리스트에 subset 값 이어붙힌다. aaa.i0 : (i0 + 5)
    return np.array(aaa) # 충족할때까지 반복한다.
col_name = ['연도','일시', '측정소', 'PM2.5']
le_col_name = ['일시', '측정소']
# path = "/content/drive/MyDrive/"
# path_save = "/content/drive/MyDrive/"
path = 'c:/study_data/_data/finedust/'
path_train = 'c:/study_data/_data/finedust/TRAIN/'
path_test = "c:/study_data/_data/finedust/TEST_INPUT/"
path_train_AWS = "c:/study_data/_data/finedust/TRAIN_AWS/"
path_test_AWS = "c:/study_data/_data/finedust/TEST_AWS/"
path_save = "c:/study_data/_save/aif/"
data_name_list = [
    '공주.csv','노은동.csv','논산.csv','대천2동.csv','독곶리.csv','동문동.csv',
    '모종동.csv','문창동.csv','성성동.csv','신방동.csv','신흥동.csv','아름동.csv','예산군.csv',
    '읍내동.csv','이원면.csv','정림동.csv','홍성읍.csv']
path_meta = 'c:/study_data/_data/finedust/META/'

# read in the csv files
pmmap_csv = pd.read_csv(path_meta+'pmmap.csv', index_col=False, encoding='utf-8')
awsmap_csv = pd.read_csv(path_meta+'awsmap.csv', index_col=False, encoding='utf-8')

# create a dictionary of places in awsmap.csv
places_awsmap = {}
for i, row in awsmap_csv.iterrows():
    places_awsmap[row['Description']] = (row['Latitude'], row['Longitude'])

# define a function to calculate the distance between two points
def distance(lat1, lon1, lat2, lon2):
    R = 6371 # earth radius in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.sin(dLon/2) * math.sin(dLon/2) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# loop over the locations in pmmap.csv
for i, row_a in pmmap_csv.iterrows():
    # find the closest places in awsmap.csv to the location in pmmap.csv
    closest_places = []
    closest_distances = []
    for j, row_b in awsmap_csv.iterrows():
        dist = distance(row_a['Latitude'], row_a['Longitude'], row_b['Latitude'], row_b['Longitude'])
        if len(closest_places) < 3:
            closest_places.append(row_b['Location'])
            closest_distances.append(dist)
        else:
            max_index = closest_distances.index(max(closest_distances))
            if dist < closest_distances[max_index]:
                closest_places[max_index] = row_b['Location']
                closest_distances[max_index] = dist
    
    # sort the distances in ascending order
    closest_places = [x for _, x in sorted(zip(closest_distances, closest_places))]
    closest_distances.sort()
    
    # print the closest places for the location in pmmap.csv with their distances
    print("Closest places to {} (in ascending order of distance):".format(row_a['Location']))
    for k in range(1):
        print("{}, distance: {:.2f} km".format(closest_places[k], closest_distances[k]))


#### 1. 데이터

train_files = glob.glob(path+'TRAIN/*.csv')
# train_aws_files = glob.glob(path+'TRAIN_AWS/*.csv')

test_input_files = glob.glob(path+'test_input/*.csv')
# test_aws_files = glob.glob(path+'test_aws/*.csv')

############################# train 폴더 #########################################

train_li=[]
for filename in train_files :
    df = pd.read_csv(filename,index_col=None, 
                     header=0, # 위에 컬럼에 대한 인식하게 함 
                     encoding='utf-8')
    train_li.append(df)
train_dataset= pd.concat(train_li,axis=0,
                         ignore_index=True # 원래 있던 인덱스는 사라지고 새로운 인덱스가 생성된다. 
                         )
# print(train_dataset)
print(train_dataset.shape) # (596088, 4)

#공주
gongjoo_train_csv=pd.read_csv(path_train_AWS +'공주.csv',encoding='utf-8',index_col=False)
# 노은동
gyeryong_train_csv=pd.read_csv(path_train_AWS +'계룡.csv',encoding='utf-8',index_col=False)
# 논산
nonsan_train_csv=pd.read_csv(path_train_AWS +'논산.csv',encoding='utf-8',index_col=False)
#대천2동
deacheon_train_csv=pd.read_csv(path_train_AWS +'대천항.csv',encoding='utf-8',index_col=False)
#독곶리
deasan_train_csv=pd.read_csv(path_train_AWS +'대산.csv',encoding='utf-8',index_col=False)
#동문동
teaan_train_csv=pd.read_csv(path_train_AWS +'태안.csv',encoding='utf-8',index_col=False)
#모종동
asan_train_csv=pd.read_csv(path_train_AWS +'아산.csv',encoding='utf-8',index_col=False)
#문창동
Oworld_train_csv=pd.read_csv(path_train_AWS +'오월드.csv',encoding='utf-8',index_col=False)
#성성동
sung_train_csv=pd.read_csv(path_train_AWS +'성거.csv',encoding='utf-8',index_col=False)
#신방동
sung02_train_csv=pd.read_csv(path_train_AWS +'성거.csv',encoding='utf-8',index_col=False)
#신흥동
seyun_train_csv=pd.read_csv(path_train_AWS +'세종연서.csv',encoding='utf-8',index_col=False)
#아름동
sego_train_csv=pd.read_csv(path_train_AWS +'세종고운.csv',encoding='utf-8',index_col=False)
#예산군
yesan_train_csv=pd.read_csv(path_train_AWS +'예산.csv',encoding='utf-8',index_col=False)
#읍내동
jangdong_train_csv=pd.read_csv(path_train_AWS +'장동.csv',encoding='utf-8',index_col=False)
#이원면
teaan02_train_csv=pd.read_csv(path_train_AWS +'태안.csv',encoding='utf-8',index_col=False)
#정림동
Oworld_train_csv02=pd.read_csv(path_train_AWS +'오월드.csv',encoding='utf-8',index_col=False)
#홍성읍
aan_train_csv=pd.read_csv(path_train_AWS +'홍북.csv',encoding='utf-8',index_col=False)

train_aws_li = [gongjoo_train_csv,gyeryong_train_csv,
    nonsan_train_csv,deacheon_train_csv,
    deasan_train_csv,teaan_train_csv,
    asan_train_csv,Oworld_train_csv,
    sung_train_csv,sung02_train_csv,
    seyun_train_csv,sego_train_csv,
    yesan_train_csv,jangdong_train_csv,
    teaan02_train_csv,Oworld_train_csv02,aan_train_csv
]

train_aws_dataset = pd.concat( train_aws_li, axis=0,ignore_index=True)
print(train_aws_dataset.shape)
############################# test 폴더 #########################################

test_li=[]
for filename in test_input_files :
    df = pd.read_csv(filename,index_col=None, 
                     header=0, # 위에 컬럼에 대한 인식하게 함 
                     encoding='utf-8')
    test_li.append(df)
test_input_dataset= pd.concat(test_li,axis=0,
                         ignore_index=True # 원래 잇던 인덱스는 사라지고 새로운 인덱스가 생성된다. 
                         )
print(test_input_dataset)

train_li=[]
for filename in train_files :
    df = pd.read_csv(filename,index_col=None, 
                     header=0, # 위에 컬럼에 대한 인식하게 함 
                     encoding='utf-8')
    train_li.append(df)
train_dataset= pd.concat(train_li,axis=0,
                         ignore_index=True # 원래 있던 인덱스는 사라지고 새로운 인덱스가 생성된다. 
                         )
# print(train_dataset)
print(train_dataset.shape) # (596088, 4)
gongjoo_test_csv=pd.read_csv(path_test_AWS +'공주.csv',encoding='utf-8',index_col=False)

sego_test_csv=pd.read_csv(path_test_AWS +'세종고운.csv',encoding='utf-8',index_col=False)
seyun_test_csv=pd.read_csv(path_test_AWS +'세종연서.csv',encoding='utf-8',index_col=False)
gyeryong_test_csv=pd.read_csv(path_test_AWS +'계룡.csv',encoding='utf-8',index_col=False)
Oworld_test_csv=pd.read_csv(path_test_AWS +'오월드.csv',encoding='utf-8',index_col=False)
jangdong_test_csv=pd.read_csv(path_test_AWS +'장동.csv',encoding='utf-8',index_col=False)
Oworld_test_csv02=pd.read_csv(path_test_AWS +'오월드.csv',encoding='utf-8',index_col=False)
nonsan_test_csv=pd.read_csv(path_test_AWS +'논산.csv',encoding='utf-8',index_col=False)
deacheon_test_csv=pd.read_csv(path_test_AWS +'대천항.csv',encoding='utf-8',index_col=False)
deasan_test_csv=pd.read_csv(path_test_AWS +'대산.csv',encoding='utf-8',index_col=False)
teaan_test_csv=pd.read_csv(path_test_AWS +'태안.csv',encoding='utf-8',index_col=False)
asan_test_csv=pd.read_csv(path_test_AWS +'아산.csv',encoding='utf-8',index_col=False)
sung_test_csv=pd.read_csv(path_test_AWS +'성거.csv',encoding='utf-8',index_col=False)
yesan_test_csv=pd.read_csv(path_test_AWS +'예산.csv',encoding='utf-8',index_col=False)
teaan02_test_csv=pd.read_csv(path_test_AWS +'태안.csv',encoding='utf-8',index_col=False)
aan_test_csv=pd.read_csv(path_test_AWS +'홍북.csv',encoding='utf-8',index_col=False)
sung02_test_csv=pd.read_csv(path_test_AWS +'성거.csv',encoding='utf-8',index_col=False)

test_csv_list = [gongjoo_test_csv,gyeryong_test_csv,
    nonsan_test_csv,deacheon_test_csv,
    deasan_test_csv,teaan_test_csv,
    asan_test_csv,Oworld_test_csv,
    sung_test_csv,sung02_test_csv,
    seyun_test_csv,sego_test_csv,
    yesan_test_csv,jangdong_test_csv,
    teaan02_test_csv,Oworld_test_csv02,aan_test_csv
]

for v in test_csv_list:
    mode = v['지점'].mode()[0]
    v['지점'] = v['지점'].fillna(mode)
print('Done.')

test_aws_dataset = pd.concat(test_csv_list,axis=0,ignore_index=True)

################### 일시 -> 월, 일, 시간으로 분리 ###############################
# 12-31 21 : 00 / 12와 21 추출 
# (596088, 4) (1051920, 8) (131376, 4) (131376, 8)
train_dataset = train_dataset.drop(['일시','연도'],axis=1)
train_aws_dataset = train_aws_dataset.drop(['지점'],axis=1)
train_all_dataset = pd.concat([train_dataset,train_aws_dataset],axis=1)
print(train_all_dataset.columns)
test_input_dataset = test_input_dataset.drop(['일시','연도'],axis=1)
test_aws_dataset = test_aws_dataset.drop(['지점'],axis=1)
test_all_dataset = pd.concat([test_input_dataset,test_aws_dataset],axis=1)
print(test_all_dataset.columns)
'''
Index(['측정소', 'PM2.5', '연도', '일시', '지점', '기온(°C)', '풍향(deg)', '풍속(m/s)',
       '강수량(mm)', '습도(%)'],
      dtype='object')
Index(['측정소', 'PM2.5', '연도', '일시', '지점', '기온(°C)', '풍향(deg)', '풍속(m/s)',
       '강수량(mm)', '습도(%)'],
      dtype='object')
'''


# 12-31 21 : 00 / 12와 21 추출 
# print(train_aws_dataset['일시'].info()) 
# # print(type(train_aws_dataset['일시'][0])) # <class 'str'>
# print(train_aws_dataset['일시'].dtype) # object
# train_dataset['month'] = train_dataset['일시'].str[:2]
# # print(train_aws_dataset['month'])
# train_dataset['day'] = train_dataset['일시'].str[3:5]
# # print(train_dataset['day'])
# train_dataset['hour'] = train_dataset['일시'].str[6:8]
# # print(train_aws_dataset['hour'])
# train_aws_dataset = train_aws_dataset.drop(['일시'],axis=1)
 
# train_aws_dataset['month'] = train_aws_dataset['month'].astype('Int16')
# train_aws_dataset['day'] = train_aws_dataset['day'].astype('Int16')
# train_aws_dataset['hour'] = train_aws_dataset['hour'].astype('Int16')

# (1051920, 10) (596088, 2) (1051920, 8) (131376, 4) (131376, 8)


train_all_dataset['month'] = train_all_dataset['일시'].str[:2]
# print(train_aws_dataset['month'])
train_all_dataset['day'] = train_all_dataset['일시'].str[3:5]
# print(train_dataset['day'])
train_all_dataset['hour'] = train_all_dataset['일시'].str[6:8]
# print(train_aws_dataset['hour'])
train_all_dataset = train_all_dataset.drop(['일시'],axis=1)

train_all_dataset['month'] = train_all_dataset['month'].astype('Int16')
train_all_dataset['day'] = train_all_dataset['day'].astype('Int16')
train_all_dataset['hour'] = train_all_dataset['hour'].astype('Int16')

test_all_dataset['month'] = test_all_dataset['일시'].str[:2]
test_all_dataset['day'] = test_all_dataset['일시'].str[3:5]
test_all_dataset['hour'] = test_all_dataset['일시'].str[6:8]
test_all_dataset['month'] = test_all_dataset['month'].astype('Int16')
test_all_dataset['day'] = test_all_dataset['day'].astype('Int16')
test_all_dataset['hour'] = test_all_dataset['hour'].astype('Int16')
print(test_all_dataset['month'])
test_all_dataset= test_all_dataset.drop(['일시'],axis=1)


############################# 라벨 인코더 #########################################

le=LabelEncoder()
train_all_dataset['locate'] = le.fit_transform(train_all_dataset['측정소'])
test_all_dataset['locate'] = le.transform(test_all_dataset['측정소'])

# train_dataset= train_dataset.drop(['측정소'],axis=1)
train_all_dataset= train_all_dataset.drop(['측정소'],axis=1)
test_all_dataset= test_all_dataset.drop(['측정소'],axis=1)


print('결측치')
print(train_all_dataset.shape,train_dataset.shape,train_aws_dataset.shape,test_input_dataset.shape,test_aws_dataset.shape)
print(train_all_dataset)
print('test_all_dataset : ',test_all_dataset.columns)
print('train_all_dataset : ',train_all_dataset.columns)
'''
test_all_dataset :  Index(['PM2.5', '연도', '기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)', '습도(%)',
       'month', 'day', 'hour', 'locate'],
      dtype='object')
train_all_dataset :  Index(['PM2.5', '연도', '기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)', '습도(%)',
       'month', 'day', 'hour', 'locate'],
      dtype='object')

'''

imputer_start = time.time()
############################# 결측치 처리 #########################################
imputer = IterativeImputer(estimator=XGBRegressor(
        n_jobs = -1,                        
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
    )) # knn알고리즘으로 평균값을 찾아낸 것이다. 
# train_dataset = train_dataset.interpolate(order=2)
# train_aws_dataset = train_aws_dataset.interpolate(order=2)
# test_aws_dataset = test_aws_dataset.interpolate(order=2)
train_all_dataset = imputer.fit_transform(train_all_dataset)
test_all_dataset = imputer.transform(test_all_dataset)
print('test_all_dataset : ',test_all_dataset)
print('train_all_dataset : ',train_all_dataset)
# for i in imputer_list : 
#     train_aws_dataset[i] = imputer.transform(train_aws_dataset[i])
#     test_aws_dataset[i] = imputer.transform(test_aws_dataset[i])
data_col = ['PM2.5', '연도', '기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)', '습도(%)',
       'month', 'day', 'hour', 'locate']
train_all_dataset = pd.DataFrame(train_all_dataset,columns=data_col)
test_all_dataset= pd.DataFrame(test_all_dataset,columns=data_col)
imputer_end = time.time()
print('imputer_time : ',round(imputer_end-imputer_start,2))

# true_test 모으기 
timesteps = 24
true_test_list = [test_all_dataset[24:48]]
a = 48
for i in range(1088) : # 3일치가 1088묶음으로 있다. 
    true_test_list.append(
        test_all_dataset[a + 48:a + 120] # 3일치
                          ) 
    a =+ 120 # 2일 
true_test = pd.concat(true_test_list,axis=0,ignore_index=True)
true_test = true_test.drop(['PM2.5'],axis=1)
true_test = split_x(true_test,timesteps)
true_test = true_test[:(len(true_test)-1)]

print('true_test.shape : ',true_test.shape) # (78337, 24, 10)
print('len(true_test) : ',len(true_test)) # 78337
# print(len(true_test)) # 78336


y = train_all_dataset['PM2.5']
x = train_all_dataset.drop(['PM2.5'],axis=1)

x = split_x(x,timesteps)
x = x[:(len(x)-1)]
y = y[timesteps:]
print('x:',x,'y:',y)
print(x.shape,y.shape)

x_train,x_test, y_train, y_test = train_test_split(
    x,y,test_size = 0.2, 
    # random_state=337,
    # shuffle=True
)

print('RobustScaler')

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

#2. 모델 

start= time.time()

model= Sequential()
model.add(LSTM(64,input_shape=(timesteps, 10)))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# model.summary()
model.save(path_save +'lstm_model.hdf5')


#3. 훈련 
model.compile(optimizer='adam', loss='mse')
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss',mode='min',restore_best_weights=True,patience=8)
model.fit(x_train,y_train,epochs=24,verbose=1,callbacks =[es])
end= time.time()
print('걸린 시간 : ',round(end -start,2),'초')

# 4. 평가, 예측

result = model.evaluate(x_test,y_test)
y_pred = model.predict(x_test)
print("model.score : ",result)
r2 = r2_score(y_test,y_pred)
print('r2 : ',r2)
mae = mean_absolute_error(y_test,y_pred)
print('mae : ',mae)


# 5. 제출
print('제출')

# print(true_test.head(50))
# print(true_test.shape) # (78336, 4)


submission_csv = pd.read_csv(path +'answer_sample.csv')
print(submission_csv.shape)
y_submit = model.predict(true_test)
submission_csv['PM2.5'] = y_submit
submission_csv.to_csv(path_save + '0510_LSTM_TIME.csv',encoding='utf-8')
print('완료')