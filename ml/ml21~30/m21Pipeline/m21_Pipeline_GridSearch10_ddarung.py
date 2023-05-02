import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold,StratifiedKFold
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 1.3 결측지 제거
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())

# 1.4 x, y 분리
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.7, shuffle=True, random_state = 34,
)

parameters = [{'rf__n_estimators' : [100, 200, 300]}, {'rf__max_depth' : [6, 10, 15, 12]}, 
            {'rf__min_samples_leaf' : [3, 10]},
    {'rf__min_samples_split' : [2, 3, 10]}, 
    {'rf__max_depth' : [6, 8, 12]}, 
    {'rf__min_samples_leaf' : [3, 5, 7, 10]},
    {'rf__n_estimators' : [100, 200, 400]},
    {'rf__min_samples_split' : [2, 3, 10]},
]

#2. 모델
pipe = Pipeline([("std", StandardScaler()),("rf", RandomForestClassifier())])  #->얘는 리스트로 받아놈.

model = GridSearchCV(pipe, parameters, cv =5, verbose=1, n_jobs= 3)
#pipe랑 parameters는 정상적인 파라미터가 아님.
#pipeline에 파라미터를 넣어야하는데 랜포 파라미터를 넣어서 오류가뜸. 해결책 rf__(정의한 것)을 넣으면됨.

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = r2_score(y_test,y_predict)
print("r2_score : ", acc)

# acc :  0.020050125313283207
# r2_score :  0.5032328122313818

