import numpy as np 
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

###결측치제거### 
# print(train_csv.isnull().sum()) 
#결측치 없음

###데이터분리(train_set)###
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, #stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
model = BaggingRegressor(DecisionTreeRegressor(),  #배깅에 쓰는 모델 DTC/ 모델 훈련 10번
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=337,
                          bootstrap=True,            #통상 True/ #샘플의 중복을 허용 (디폴트=True)
                          )              



#3. 훈련 
model.fit(x_train, y_train)

#4. 평가
y_pred = model.predict(x_test)
print("model.score:", model.score(x_test, y_test))
print("r2:", r2_score(y_test, y_pred))