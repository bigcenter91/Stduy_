import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
# 이상치 판별에 RobustScaler
from sklearn.metrics import accuracy_score


x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y
)


scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

# parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000] 디폴트 100 / 1~inf / 정수 : epochs
#               'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3 / 0~1 / eta
#               'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~inf / 정수 : 최대의 깊이를 잡겠다 / 3~7정도가 잘먹힌다
#               'gamma' : [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10, 100] 디폴트 0 / 0~inf
#               'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] 디폴트 1 / 0~inf : 최소의 
#               'subsample' : [0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1 : 데이터의 훈련 시키는 양 ex) 0.8이면 80%의 양으로 훈련 시키겠다
#               'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]디폴트 1 / 0~1
#               'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]디폴트 1 / 0~1
#               'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]디폴트 1 / 0~1
#               'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10]디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha_라쏘
#               'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda_리지
#               } reg_alpha/ lambda 레이어에 양수로 떨어뜨리겠다


parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.3,
              'max_depth' : 3,
              'gamma' : 1,
              'min_child_weight' : 1,
              'subsample' : 1,
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1.0,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,
              'reg_lambda' : 1,
              'random_state' : 337,
              'verbose' : 1,
              }

# xgboost.core.XGBoostError: Invalid Parameter format for colsample_bylevel expect float but value='[1]'
# 나는 float 형태를 기대했는데 리스트형태로줘서

#2. 모델
# model = XGBClassifier(**parameters)
model = XGBClassifier()


#3. 훈련
# model.set_params(early_stopping_rounds=10)
hist = model.fit(x_train, y_train, 
          eval_set=[(x_train, y_train),(x_test, y_test)], # validation 데이터가 되는거다
        #   early_stopping_rounds=10,
          verbose=1, # 0이냐 0이 아닌거냐
        #   eval_metric='logloss,'
          eval_metric='error', # 이진 분류
        #   eval_metric='auc',   # 이진
        # val_metric='merror',   # 이진
        #   eval_metric='rmse', mae, rmsle  # 회귀
        #  
                
          )



#4. 평가, 예측
# print("최상의 매개션수 : ", model.best_params_)
# print("최상의 점수 : ", model.best_score_)

results = model.score(x_test, y_test)
print("최종 점수 : ", results)

y_predict =  model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)
# score가 accuracy_score라는 얘기지

import pickle 
path = 'c:/study_data/_save/pickle_test/'
pickle.dump(model, open(path+'m43_pickle_save.dat','wb'))


# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,8))
# plt.plot(hist["validation_0"]["error"])
# plt.plot(hist["validation_1"]["error"])
# plt.xlabel('round')
# plt.ylabel('error')
# plt.title('XGBoost')
# plt.show()