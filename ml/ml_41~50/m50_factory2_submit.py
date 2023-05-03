import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error, r2_score
# / // \ \\ 다 같다_경로의 하위 디렉토리
# /n 줄바꿈?

path = 'c:/study_data/_data/dust/'
save_path = 'c:/study_data/_save/dust/'

#_data
# TRAIN
# TRAIN_AWS
# TEST_INPUT
# TEST_AWS
# META
# anser_sample.csv
# glob = 이 폴더 안에 있는 것을 모두 불러와 텍스트화 시켜준다

train_files = glob.glob(path + "TRAIN/*.csv")
# print(train_files)
test_input_files = glob.glob(path + 'test_input/*.csv')
print(test_input_files) # 가독성 있는 변수명을 지어야해

################### Train 폴더 ###################
li = [] # 리스트 만들어줘야해?
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig')
    li.append(df)
print(li) # [35064 rows x 4 columns]
print(len(li)) # 17 // 명확하게 리스트 형태로 나온게 아니야 concat 해줘야지?

train_dataset = pd.concat(li, axis=0,
                          ignore_index=True)
print(train_dataset) # [596088 rows x 4 columns]


################### Test 폴더 ###################
li = [] # 리스트 만들어줘야해?
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header=0,
                     encoding='utf-8-sig')
    li.append(df)
print(li) # [7728 rows x 4 columns]]
print(len(li)) # 17 // 명확하게 리스트 형태로 나온게 아니야 concat 해줘야지?

test_input_dataset = pd.concat(li, axis=0,
                          ignore_index=True)
print(test_input_dataset) # [131376 rows x 4 columns]

################### 측정소 라벨인코더 ###################
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소'])
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])
print(train_dataset) # [596088 rows x 5 columns]
print(test_input_dataset) # [131376 rows x 5 columns]
train_dataset = train_dataset.drop(['측정소'], axis=1) # 열을 삭제하니까 1 [596088 rows x 5 columns]
test_input_dataset = test_input_dataset.drop(['측정소'], axis=1) # [131376 rows x 5 columns]


################### 일시 > 월, 일, 시간으로 분리하라 ###################
# 12-31 21:00 > 12와 21 추출
# print(train_dataset.info()) 
#  #   Column  Non-Null Count   Dtype
# ---  ------  --------------   -----
#  0   연도      596088 non-null  int64
#  1   일시      596088 non-null  object
#  2   PM2.5   580546 non-null  float64
#  3   locate  596088 non-null  int32
# dtypes: float64(1), int32(1), int64(1), object(1)

# <class 'pandas.core.series.Series'>
# object 나오면 그냥 텍스트구나 스트링이구나 하면된다_dataset

################### train 변경 ###################
train_dataset['month'] = train_dataset['일시'].str[:2] # str[:2] 두번째부터 뽑을거야
# print(train_dataset['month'])
train_dataset['hour'] = train_dataset['일시'].str[6:8] # str[:2] 두번째부터 뽑을거야
# print(train_dataset['hour'])
train_dataset = train_dataset.drop(['일시'], axis=1)
print(train_dataset) # [596088 rows x 5 columns]
# print(train_dataset.info())
### str > int
# train_dataset['month'] = pd.to_numeric(train_dataset['month'])
# train_dataset['month'] = pd.to_numeric(train_dataset['month']).astype('int16')
train_dataset['month'] = train_dataset['month'].astype('int8')
train_dataset['hour'] = train_dataset['hour'].astype('int8')
# test_input_dataset['month'] = test_input_dataset['month'].astype('int8')
# test_input_dataset['hour'] = test_input_dataset['hour'].astype('int8')
print(train_dataset.info())
# print(test_input_dataset.info())


#################### test_input  변경 ###################
test_input_dataset['month'] = test_input_dataset['일시'].str[:2] # str[:2] 두번째부터 뽑을거야
# print(test_input_dataset['month'])
test_input_dataset['hour'] = test_input_dataset['일시'].str[6:8] # str[:2] 두번째부터 뽑을거야
# print(test_input_dataset['hour'])
test_input_dataset = test_input_dataset.drop(['일시'], axis=1)
print(test_input_dataset) # [596088 rows x 5 columns]
# print(test_input_dataset.info())
### str > int
# test_input_dataset['month'] = pd.to_numeric(test_input_dataset['month'])
# test_input_dataset['month'] = pd.to_numeric(test_input_dataset['month']).astype('int16')
test_input_dataset['month'] = test_input_dataset['month'].astype('int8')
test_input_dataset['hour'] = test_input_dataset['hour'].astype('int8')
# test_input_dataset['month'] = test_input_dataset['month'].astype('int8')
# test_input_dataset['hour'] = test_input_dataset['hour'].astype('int8')
print(test_input_dataset.info())
print(test_input_dataset.info())


# print(train_dataset.info())
# print(test_input_dataset.info())


#################### 결측치 제거 PM2.5에 15542개가 있다 ####################
# 전체 596085 > 580546으로 줄인다
train_dataset = train_dataset.dropna()
print(train_dataset.info())
# 0   연도      580546 non-null  int64
#  1   PM2.5   580546 non-null  float64
#  2   locate  580546 non-null  int32
#  3   month   580546 non-null  int8
#  4   hour    580546 non-null  int8
#################### 결측치 제거 PM2.5 15542 ####################
x_submit = test_input_dataset[test_input_dataset.isna().any(axis=1)]
### 결측치가 있는 데이터의 행들만 추출
print(x_submit)
# [78336 rows x 5 columns]
print(x_submit.info())
# PM2.5   0 non-null      float64 전부 nan이라는 얘기지?

#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   연도      78336 non-null  int64
#  1   PM2.5   0 non-null      float64
#  2   locate  78336 non-null  int32
#  3   month   78336 non-null  int8
#  4   hour    78336 non-null  int8
# dtypes: float64(1), int32(1), int64(1), int8(2)
x_submit = x_submit.drop(['PM2.5'], axis=1)
print(x_submit)
# [78336 rows x 4 columns]
#### 파생피쳐 상당히 중요하다 시즌 - 파생피처도 생각해봐야해 ####

y = train_dataset['PM2.5'] # 메모리 안전빵 때문에 y를 먼저 해준거다 하지만 큰 의미는 없다, yys의 세심한 성격?
x = train_dataset.drop(['PM2.5'], axis=1)
print(x, '\n', y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, random_state=555, shuffle=True # 시계열에서도 True로 놓아도 된다
)

parameters = {'n_estimators' : 10000,
              'learning_rate' : 0.01,
              'max_depth' : 7,
            #   'gamma' : 1,
            #   'min_child_weight' : 1,
              'subsample' : 1,
            #   'colsample_bytree' : 1,
            #   'colsample_bylevel' : 1,
            #   'colsample_bynode' : 1,
            #   'reg_alpha' : 0,
            #   'reg_lambda' : 1,
            #   'random_state' : 524,
            #   'verbose' : 0,
              'n_jobs' : -1,
              }


#2. 모델
model = XGBRegressor()

#3. 컴파일, 훈련
model.set_params(**parameters,
                 eval_metric='mae',
                 early_stopping_rounds=200,
                )

start_time = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_train, y_train), (x_test, y_test)]   
)

end_time= time.time()
print("걸린시간 : ", round(end_time - start_time, 2), "초")

#4. 평가, 예측
y_predic = model.predict(x_test)
y_submit = model.predict(x_submit)

results = model.score(x_test, y_test)
print("model.score: ", results)
 
r2 = r2_score(y_test, y_predic)
print("r2 스코어 :", r2)

mae = mean_absolute_error(y_test, y_predic)
print("mae 스코어 :", mae)

############### 제출 파일 맹글기 ###############
answer_sample_csv = pd.read_csv(path + 'answer_sample.csv',
                                index_col=None, header=0, encoding='utf-8-sig')

# print(answer_sample_csv) # [78336 rows x 4 columns]
# print(answer_sample_csv.info())

answer_sample_csv['PM2.5'] = y_submit
print(answer_sample_csv)

answer_sample_csv.to_csv(save_path + 'm50_factory2_submit.csv',
                         index=None)