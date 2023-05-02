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
from sklearn.decomposition import PCA
#warnings.filterwarnings(action = 'ignore')

# 1. 데이터
# 1.1 경로, 가져오기
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

pca = PCA(n_components= 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#2. 모델

model = RandomForestClassifier(random_state=123)

#3. 훈련
model.fit(x_train,y_train)

#4. 결과
results = model.score(x_test, y_test)
print("결과는  :", results)