import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time



# 아이리스 데이터하고 트레인테스트스플릿하고 KFold, cross_val_score 전처리하는구나를 알아야한다

#1. 데이터
# x, y = load_iris(return_X_y=True)
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2, #stratify=y
)
# 30개가 테스트가 될거다
# Kfold 5번 돌릴거야

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

parameters = [
    {"C":[1,10,100,1000], "kernel": ['linear'], 'degree':[3,4,5]}, # 12 
    {"C":[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},  # 12
    {"C":[1,10,100,1000], "kernel": ['sigmoid'],
     'gamma':[0.01, 0.001, 0.001, 0.0001], 'degree':[3, 4]}, # 24
    {"C" : [0.1,1],'gamma':[1,10]},
    # 총 48번 돌겠지// 리스트 안에 딕셔너리 형태 안에 다시 리스트
    # 그리드 써치는 그냥 무조건 다 돈다 파라미터 계수만큼
]

# Keyvalue형태로 받으면 편하겠지_딕셔너리 형태로 받게끔 할거야
#2. 모델
# model = GridSearchCV(SVC(), parameters, 
# model = RandomizedSearchCV(SVC(), parameters, 
model = HalvingGridSearchCV(SVC(), parameters, 
                    #  cv=kfold 
                     cv=5, # 분류의 디폴트는 Stratifiedkfold야
                     verbose=1,
                     refit=True, # refit=True면 최적의 파라미터로 훈련, 디폴트가 True
                     factor=3.5,#(factor 디폴트는 3)   # =False면 해당 파라미터로 훈련하지 않고 최종 파라미터로 훈련 = 최적 값을 보관하지 않는다
                     n_jobs=-1)  # n_jobs: cpu의 갯수를 몇개 사용할것인지
                    # 성능 차이가 factor에 따라 달라지지?
                    # 240번 돌겠지

#그리드 써치에서 연산량을 빼서 쓰겠다
# 52개 중에 10개를 쓰겠다 그 전에는 중복된 연산이 많았잖아?
# 나중엔 많으면 왠만하면 Random으로 쓸 것이다
# 데이터는 N빵으로 늘리고 하이퍼파라미터는 줄이고

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


print("걸린시간 : ", round(end_time - start_time, 2),'초')
# 로드 디지트 랜덤 : 12
# halbing 그리드 : 5

print(x.shape, x_train.shape) # (1797, 64) (1437, 64)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score_ : ',model.best_score_) # train best score
print('model_score : ',model.score(x_test, y_test)) # test best score

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score_ :  0.9916666666666668
# model_score :  1.0

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
# accuracy_score :  1.0

y_pred_best = model.best_estimator_.predict(x_test)
print("최적의 튠 ACC", accuracy_score(y_test, y_pred_best))
# 최적의 튠 ACC 1.0

print("걸린시간 : ", round(end_time - start_time, 2),'초')


########################################################
# print(pd.DataFrame(model.cv_results_))
print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)) #ascending 오름차순이 디폴트다 False가 내림이겠지?
print(pd.DataFrame(model.cv_results_).columns)

path = './temp/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score', ascending=True)\
    .to_csv(path + 'm14_HalvingGridSearch1.csv')
# 데이터를 csv를 맹그러놓을 때 to.csv

'''
min_resources_: 100                     # 최소 훈련 데이터 갯수
max_resources_: 1437                    # 최대 훈련 데이터 갯수
aggressive_elimination: False
factor: 3.5                             # 엔빵양
----------
iter: 0                                                     
n_candidates: 54                        # 전체 파라미터 갯수
n_resources: 100                        # 0번째 훈련에 쓸 훈련 데이터 갯수
Fitting 5 folds for each of 54 candidates, totalling 270 fits
----------
iter: 1
n_candidates: 16                        # 전체 파라미터 갯수 / factor
n_resources: 350                        # min_resources * factor
Fitting 5 folds for each of 16 candidates, totalling 80 fits
----------
iter: 2             
n_candidates: 5                         # 17 / factor
n_resources: 1225                       # 350 * factor
Fitting 5 folds for each of 5 candidates, totalling 25 fits
'''