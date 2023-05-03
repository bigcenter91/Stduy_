import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
# 이상치 판별에 RobustScaler
from sklearn.metrics import accuracy_score
import pickle

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y
)


scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 - 피클 불러오겠지?
import pickle
path = 'c:/study_data/_save/pickle_test/'
model = pickle.load(open(path + 'm44_joblib_save.dat', 'rb'))
# 머신러닝은 pickle로 save, load하면 된다


#3. 평가, 예측
results = model.score(x_test, y_test)
print("최종 점수 : ", results)

y_predict =  model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)