import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.experimental import enable_halving_search_cv
import pandas as pd
from sklearn.model_selection import HalvingRandomSearchCV
import time
import random
from sklearn.model_selection import train_test_split
#warnings.filterwarnings(action = 'ignore')

seed = 0 #random state 0넣는 거랑 비슷함.
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

#print(train_csv.shape) #(652, 9)
#print(test_csv.shape) #(116, 8)

#print(train_csv.isnull().sum()) # 결측치 x
x = train_csv.drop(['Outcome','Pregnancies',
                    'BloodPressure','SkinThickness',
                    'Insulin'],axis =1)
y = train_csv['Outcome']

# 필요 없는 특성 삭제
#x= np.delete(x, [1, 3, 4, 5], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

#2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)
print("=====================================================")
print(type(model).__name__, ":", model.feature_importances_)

#지우기전
#[0.08487798 0.24327991 0.09016173 0.07369281 0.07264017 0.16508998,0.12439203 0.14586539] accuracy_score :  0.7175572519083969

#지운후
# [0.34468159 0.25129734 0.19480203 0.20921903] accuracy_score :  0.732824427480916