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
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
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
model_list = [RandomForestClassifier(),
GradientBoostingClassifier(),
DecisionTreeClassifier(),
XGBClassifier()]

# 3. 훈련
for model, value in enumerate(model_list) :
    model = value
    model.fit(x_train,y_train)

    #4. 평가, 예측
    result = model.score(x_test,y_test)
    print("acc : ", result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    
    print("accuracy_score : ", acc)
    print(type(model).__name__, ":", \
        model.feature_importances_)
    print("=====================================================")
    
    # 하위 20-25%의 피처 제거 후 재학습
    #idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    # argmin = np.argpartition(model.feature_importances_, 4)[:4]
    # x_drop = pd.DataFrame(x).drop(argmin, axis=1)
    
    n_drop = int(len(model.feature_importances_) * 0.25)
    idx = np.argsort(model.feature_importances_)[-n_drop:]
    x_drop = pd.DataFrame(x).drop(x.columns[idx], axis=1) 
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_drop, y, train_size=0.7, shuffle=True, random_state=123)

    model.fit(x_train1, y_train1)

    result = model.score(x_test1, y_test1)
    print("acc after feature selection: ", result)

    y_predict1 = model.predict(x_test1)
    acc1 = accuracy_score(y_test1, y_predict1)

    print("accuracy_score after feature selection: ", acc1)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")