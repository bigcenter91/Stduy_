import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
import random
import pandas as pd
warnings.filterwarnings(action = 'ignore')

seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

#print(train_csv.shape) #(652, 9)
#print(test_csv.shape) #(116, 8)

#print(train_csv.isnull().sum()) # 결측치 x
x = train_csv.drop(['Outcome'],axis =1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
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
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)

# acc :  0.7404580152671756
# accuracy_score :  0.7404580152671756